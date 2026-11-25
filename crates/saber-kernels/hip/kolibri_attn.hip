#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define KA_BM 16
#define KA_BQ 4
#define KA_NQPK 4
#define KA_HEAD 128
#define KA_NW 4
#define KA_NT 128
#define KA_NF 8
#define KA_SSTR 20
#define KA_VSTR 20
#define KA_LDS_PER_WAVE (KA_BM * KA_SSTR + KA_BM + (KA_HEAD * KA_VSTR) / 2)
#define KA_ACC_ROW(lane, e) (((lane) >> 4) * 8 + (e))

typedef __bf16 ka_bf16x8 __attribute__((ext_vector_type(8)));
typedef float ka_floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int ka_uint2 __attribute__((ext_vector_type(2)));
typedef unsigned int ka_uint4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ ka_floatx8 ka_wmma(ka_bf16x8 a, ka_bf16x8 b, ka_floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

template <int BYTE>
__device__ __forceinline__ float ka_fp8(unsigned int packed) {
  return __builtin_amdgcn_cvt_f32_fp8(packed, BYTE);
}

__device__ __forceinline__ float ka_tile_max(const float *v) {
  float m = v[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) m = fmaxf(m, v[i]);
  return fmaxf(m, __shfl_xor(m, 16, 32));
}

__device__ __forceinline__ float ka_tile_sum(const float *v) {
  float s = v[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) s += v[i];
  return s + __shfl_xor(s, 16, 32);
}

extern "C" __global__ __launch_bounds__(KA_NT) void kolibri_attn_prefill_h128_gqa12_fp8(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const int *__restrict__ block_table,
    const float *__restrict__ kscales, const float *__restrict__ vscales,
    __bf16 *__restrict__ output, const long *__restrict__ positions,
    int rows, int nq, int nkv, int block_size, int ring_blocks, int window,
    float scale, int q_head_offset) {
  if (nq != 24 || nkv != 2) return;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int kvh = wave;
  if (kvh >= nkv) return;
  const int qblock = blockIdx.x;
  if (qblock * KA_BQ >= rows) return;
  const int seq_len = (int)positions[rows - 1] + 1;
  const int context = seq_len - rows;
  const int c = lane & 15;
  const int h = lane >> 4;
  __shared__ float lds[KA_NW * KA_LDS_PER_WAVE];
  float *scores = lds + wave * KA_LDS_PER_WAVE;
  float *alpha_row = scores + KA_BM * KA_SSTR;
  __bf16 *values = (__bf16 *)(alpha_row + KA_BM);

  ka_bf16x8 qfragments[KA_HEAD / 16];
  {
    const int m = c;
    const int qpos = qblock * KA_BQ + m / KA_NQPK;
    const int qhead = q_head_offset + kvh * 12 + m % KA_NQPK;
    const bool live = qpos < rows && qhead < nq;
    const __bf16 *qp = q + ((long)(live ? qpos : 0) * nq + (live ? qhead : 0)) * KA_HEAD + h * 8;
#pragma unroll
    for (int ks = 0; ks < KA_HEAD / 16; ++ks) {
      ka_bf16x8 value = *(const ka_bf16x8 *)(qp + ks * 16);
      if (!live)
#pragma unroll
        for (int i = 0; i < 8; ++i) value[i] = (__bf16)0.f;
      qfragments[ks] = value;
    }
  }

  ka_floatx8 accumulated[KA_NF];
#pragma unroll
  for (int f = 0; f < KA_NF; ++f)
#pragma unroll
    for (int e = 0; e < 8; ++e) accumulated[f][e] = 0.f;
  float maximum = -INFINITY;
  float denominator = 0.f;
  const int qpos_hi = min(qblock * KA_BQ + KA_BQ - 1, rows - 1);
  const int first_key = window > 0 ? max(0, context + qblock * KA_BQ - window + 1) : 0;
  const int first_tile = first_key / 16;
  const int last_tile = (context + qpos_hi) / 16;

  for (int tile = first_tile; tile <= last_tile; ++tile) {
    ka_floatx8 score_accumulator;
#pragma unroll
    for (int e = 0; e < 8; ++e) score_accumulator[e] = 0.f;
    const int token = tile * 16 + c;
    const int valid_token = token < seq_len ? token : 0;
    const long logical_block = valid_token / block_size;
    const long block = block_table[ring_blocks > 0 ? logical_block % ring_blocks : logical_block];
    const long slot = valid_token % block_size;
    const long cache_position = block * block_size + slot;
    const long base = (cache_position * nkv + kvh) * KA_HEAD;
    const float key_scale = kscales[cache_position * nkv + kvh];
#pragma unroll
    for (int ks = 0; ks < KA_HEAD / 16; ++ks) {
      const ka_uint2 raw = *(const ka_uint2 *)(kc + base + ks * 16 + h * 8);
      ka_bf16x8 key;
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        key[w * 4 + 0] = (__bf16)(ka_fp8<0>(raw[w]) * key_scale);
        key[w * 4 + 1] = (__bf16)(ka_fp8<1>(raw[w]) * key_scale);
        key[w * 4 + 2] = (__bf16)(ka_fp8<2>(raw[w]) * key_scale);
        key[w * 4 + 3] = (__bf16)(ka_fp8<3>(raw[w]) * key_scale);
      }
      if (token >= seq_len)
#pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = (__bf16)0.f;
      score_accumulator = ka_wmma(qfragments[ks], key, score_accumulator);
    }

#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = KA_ACC_ROW(lane, e);
      const int qpos = qblock * KA_BQ + m / KA_NQPK;
      const int qhead = q_head_offset + kvh * 12 + m % KA_NQPK;
      const int absolute = context + qpos;
      const bool keep = qpos < rows && qhead < nq && token < seq_len && token <= absolute
                        && (window == 0 || absolute - token < window);
      scores[m * KA_SSTR + c] = keep ? scale * score_accumulator[e] : -INFINITY;
    }
    __builtin_amdgcn_s_waitcnt(0);

    float score_values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) score_values[i] = scores[c * KA_SSTR + h * 8 + i];
    const float tile_maximum = ka_tile_max(score_values);
    float next_maximum = fmaxf(maximum, tile_maximum);
    next_maximum = next_maximum > -INFINITY ? next_maximum : 0.f;
    float probabilities[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) probabilities[i] = __builtin_expf(score_values[i] - next_maximum);
    const float tile_sum = ka_tile_sum(probabilities);
    const float alpha = __builtin_expf(maximum - next_maximum);
    denominator = denominator * alpha + tile_sum;
    maximum = next_maximum;
    if (h == 0) alpha_row[c] = alpha;
    __builtin_amdgcn_s_waitcnt(0);

    const int value_token = tile * 16 + c;
    const int value_valid = value_token < seq_len ? value_token : 0;
    const long value_logical_block = value_valid / block_size;
    const long value_block = block_table[ring_blocks > 0 ? value_logical_block % ring_blocks : value_logical_block];
    const long value_slot = value_valid % block_size;
    const long value_position = value_block * block_size + value_slot;
    const long value_base = (value_position * nkv + kvh) * KA_HEAD;
    const float value_scale = vscales[value_position * nkv + kvh];
#pragma unroll
    for (int d0 = 0; d0 < KA_HEAD; d0 += 32) {
      const int d = d0 + h * 16;
      const ka_uint4 raw = *(const ka_uint4 *)(vc + value_base + d);
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        values[(d + w * 4 + 0) * KA_VSTR + c] = value_token < seq_len ? (__bf16)(ka_fp8<0>(raw[w]) * value_scale) : (__bf16)0.f;
        values[(d + w * 4 + 1) * KA_VSTR + c] = value_token < seq_len ? (__bf16)(ka_fp8<1>(raw[w]) * value_scale) : (__bf16)0.f;
        values[(d + w * 4 + 2) * KA_VSTR + c] = value_token < seq_len ? (__bf16)(ka_fp8<2>(raw[w]) * value_scale) : (__bf16)0.f;
        values[(d + w * 4 + 3) * KA_VSTR + c] = value_token < seq_len ? (__bf16)(ka_fp8<3>(raw[w]) * value_scale) : (__bf16)0.f;
      }
    }
    __builtin_amdgcn_s_waitcnt(0);

#pragma unroll
    for (int f = 0; f < KA_NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        accumulated[f][e] *= alpha_row[KA_ACC_ROW(lane, e)];
    ka_bf16x8 probability_fragment;
#pragma unroll
    for (int i = 0; i < 8; ++i) probability_fragment[i] = (__bf16)probabilities[i];
#pragma unroll
    for (int f = 0; f < KA_NF; ++f) {
      const ka_bf16x8 value_fragment = *(const ka_bf16x8 *)(values + (f * 16 + c) * KA_VSTR + h * 8);
      accumulated[f] = ka_wmma(probability_fragment, value_fragment, accumulated[f]);
    }
    __builtin_amdgcn_s_waitcnt(0);
  }

  if (h == 0) alpha_row[c] = denominator;
  __builtin_amdgcn_s_waitcnt(0);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int m = KA_ACC_ROW(lane, e);
    const int qpos = qblock * KA_BQ + m / KA_NQPK;
    const int qhead = q_head_offset + kvh * 12 + m % KA_NQPK;
    if (qpos >= rows || qhead >= nq) continue;
    __bf16 *out = output + ((long)qpos * nq + qhead) * KA_HEAD + c;
#pragma unroll
    for (int f = 0; f < KA_NF; ++f) out[f * 16] = (__bf16)(accumulated[f][e] / alpha_row[m]);
  }
}
