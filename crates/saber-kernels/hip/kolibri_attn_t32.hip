#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define KAT_BQ 32
#define KAT_GQA 12
#define KAT_HEAD 128
#define KAT_TILE 48
#define KAT_KROW 136
#define KAT_VROW 56
#define KAT_NT 768
#define KAT_NF 8

using kat_bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using kat_floatx8 = float __attribute__((ext_vector_type(8)));
using kat_uint2 = unsigned int __attribute__((ext_vector_type(2)));
using kat_uint4 = unsigned int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ kat_floatx8 kat_wmma(
    kat_bf16x8 a, kat_bf16x8 b, kat_floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

template <int BYTE>
__device__ __forceinline__ float kat_fp8(unsigned int packed) {
  return __builtin_amdgcn_cvt_f32_fp8(packed, BYTE);
}

__device__ __forceinline__ kat_bf16x8 kat_fp8x8(kat_uint2 raw, float scale) {
  kat_bf16x8 result;
#pragma unroll
  for (int word = 0; word < 2; ++word) {
    result[word * 4 + 0] = (__bf16)(kat_fp8<0>(raw[word]) * scale);
    result[word * 4 + 1] = (__bf16)(kat_fp8<1>(raw[word]) * scale);
    result[word * 4 + 2] = (__bf16)(kat_fp8<2>(raw[word]) * scale);
    result[word * 4 + 3] = (__bf16)(kat_fp8<3>(raw[word]) * scale);
  }
  return result;
}

__device__ __forceinline__ float kat_other_half(float value) {
  return __shfl_xor(value, 16, 32);
}

__device__ __forceinline__ bool kat_wave_any(bool predicate) {
  return __builtin_amdgcn_ballot_w32(predicate) != 0u;
}

struct KatShared {
  __bf16 k[KAT_TILE * KAT_KROW];
  __bf16 v[KAT_HEAD * KAT_VROW];
};

extern "C" __global__ __launch_bounds__(KAT_NT) void kolibri_attn_prefill_h128_gqa12_fp8_t32(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const int *__restrict__ block_table,
    const float *__restrict__ kscales, const float *__restrict__ vscales,
    __bf16 *__restrict__ output, const long *__restrict__ positions,
    int rows, int nq, int nkv, int block_size, int ring_blocks, int window,
    float scale, int q_head_offset) {
  if (nq != 24 || nkv != 2 || q_head_offset != 0) return;
  __shared__ KatShared sm;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int c = lane & 15;
  const int hh = lane >> 4;
  const int qb = blockIdx.x;
  const int kvh = blockIdx.y;
  const int seq_len = (int)positions[rows - 1] + 1;
  const int context = seq_len - rows;
  const int qmax = min(qb * KAT_BQ + KAT_BQ - 1, rows - 1);
  const int first_key = window > 0 ? max(0, context + qb * KAT_BQ - window + 1) : 0;
  const int last_key = context + qmax;
  const int ntiles = (last_key - first_key + KAT_TILE) / KAT_TILE;

  const int r = wave * 16 + c;
  const int qil = r / KAT_GQA;
  const int hi = r - qil * KAT_GQA;
  const int qpos = qb * KAT_BQ + qil;
  const int qhead = kvh * KAT_GQA + hi;
  const bool live = qpos < rows;
  const int qrow = live ? qpos : rows - 1;
  const int absolute = context + qpos;
  const __bf16 *qp = q + ((long)qrow * nq + qhead) * KAT_HEAD + hh * 8;
  kat_bf16x8 qf[KAT_HEAD / 16];
#pragma unroll
  for (int t = 0; t < KAT_HEAD / 16; ++t) {
    qf[t] = *(const kat_bf16x8 *)(qp + t * 16);
    if (!live)
#pragma unroll
      for (int i = 0; i < 8; ++i) qf[t][i] = (__bf16)0.f;
  }

  kat_floatx8 acc[KAT_NF];
#pragma unroll
  for (int f = 0; f < KAT_NF; ++f)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[f][e] = 0.f;
  float m_ref = -3.0e38f;
  float offset = 3.0e38f;
  float denominator = 0.f;

  for (int tile = 0; tile < ntiles; ++tile) {
    __syncthreads();
    if (tid < KAT_TILE * (KAT_HEAD / 16)) {
      const int key = tid / (KAT_HEAD / 16);
      const int chunk = tid - key * (KAT_HEAD / 16);
      const int token = min(first_key + tile * KAT_TILE + key, last_key);
      const int logical_block = token / block_size;
      const int block = block_table[ring_blocks > 0 ? logical_block % ring_blocks : logical_block];
      const long cache_position = (long)block * block_size + token % block_size;
      const long base = (cache_position * nkv + kvh) * KAT_HEAD;
      const float key_scale = kscales[cache_position * nkv + kvh];
      const kat_uint4 raw = *(const kat_uint4 *)(kc + base + chunk * 16);
      *(kat_bf16x8 *)&sm.k[key * KAT_KROW + chunk * 16] =
          kat_fp8x8((kat_uint2){raw.x, raw.y}, key_scale);
      *(kat_bf16x8 *)&sm.k[key * KAT_KROW + chunk * 16 + 8] =
          kat_fp8x8((kat_uint2){raw.z, raw.w}, key_scale);
    }
    {
      const int key = tid / (KAT_HEAD / 8);
      const int dimension = (tid - key * (KAT_HEAD / 8)) * 8;
      const int token = min(first_key + tile * KAT_TILE + key, last_key);
      const int logical_block = token / block_size;
      const int block = block_table[ring_blocks > 0 ? logical_block % ring_blocks : logical_block];
      const long cache_position = (long)block * block_size + token % block_size;
      const long base = (cache_position * nkv + kvh) * KAT_HEAD;
      const float value_scale = vscales[cache_position * nkv + kvh];
      const kat_uint2 raw = *(const kat_uint2 *)(vc + base + dimension);
      const kat_bf16x8 value = kat_fp8x8(raw, value_scale);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sm.v[(dimension + i) * KAT_VROW + key] = value[i];
    }
    __syncthreads();

#pragma unroll 1
    for (int mt = 0; mt < 3; ++mt) {
      kat_floatx8 score = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const __bf16 *krow = &sm.k[(mt * 16 + c) * KAT_KROW + hh * 8];
#pragma unroll
      for (int t = 0; t < KAT_HEAD / 16; ++t) {
        const kat_bf16x8 key = *(const kat_bf16x8 *)(krow + t * 16);
        score = kat_wmma(key, qf[t], score);
      }
      const int key0 = first_key + tile * KAT_TILE + mt * 16 + hh * 8;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int token = key0 + e;
        const bool keep = token <= absolute && (window == 0 || absolute - token < window);
        score[e] = keep ? score[e] * scale : -INFINITY;
      }

      float smax = score[0];
#pragma unroll
      for (int e = 1; e < 8; ++e) smax = fmaxf(smax, score[e]);
      if (kat_wave_any(smax > m_ref + 14.0f)) {
        const float rmax = fmaxf(smax, kat_other_half(smax));
        const float next = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
        const float alpha = __builtin_expf(m_ref - next);
#pragma unroll
        for (int f = 0; f < KAT_NF; ++f) acc[f] *= alpha;
        denominator *= alpha;
        m_ref = next;
        offset = 0.f - next;
      }

      float probability[8];
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        probability[e] = __builtin_expf(score[e] + offset);
        sum += probability[e];
      }
      denominator += sum;
      kat_bf16x8 p;
#pragma unroll
      for (int e = 0; e < 8; ++e) p[e] = (__bf16)probability[e];
      const __bf16 *vrow = &sm.v[c * KAT_VROW + mt * 16 + hh * 8];
#pragma unroll
      for (int f = 0; f < KAT_NF; ++f) {
        const kat_bf16x8 value = *(const kat_bf16x8 *)(vrow + f * 16 * KAT_VROW);
        acc[f] = kat_wmma(value, p, acc[f]);
      }
    }
  }

  if (!live) return;
  const float norm = denominator + kat_other_half(denominator);
  const float inverse = norm > 0.f ? 1.f / norm : 0.f;
  __bf16 *out = output + ((long)qpos * nq + qhead) * KAT_HEAD + hh * 8;
#pragma unroll
  for (int f = 0; f < KAT_NF; ++f) {
    kat_floatx8 value = acc[f];
#pragma unroll
    for (int e = 0; e < 8; ++e) out[f * 16 + e] = (__bf16)(value[e] * inverse);
  }
}
