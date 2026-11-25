#include "ripple_spike.hip"

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_act_quant_bf16(
    const __bf16 *__restrict__ input, unsigned char *__restrict__ output,
    float *__restrict__ scales, int rows, int width) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || group * 128 >= width || lane >= 32) return;
  const int base = row * width + group * 128;
  float amax = 0.f;
  for (int e = 0; e < 4; ++e) amax = fmaxf(amax, fabsf((float)input[base + lane * 4 + e]));
  for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
  const float scale = fmaxf(amax * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) scales[row * (width / 128) + group] = scale;
  for (int e = 0; e < 4; ++e)
    output[base + lane * 4 + e] = rs_fp8(fminf(fmaxf((float)input[base + lane * 4 + e] / scale,
                                                     -448.f), 448.f));
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_draft_swiglu_quant_bf16(
    const __bf16 *__restrict__ gate_up, unsigned char *__restrict__ output,
    float *__restrict__ scales, int rows, int width) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || group * 128 >= width || lane >= 32) return;
  const int base = row * width + group * 128;
  const long gate_base = (long)row * 2 * width + group * 128;
  float amax = 0.f;
  float values[4];
  for (int e = 0; e < 4; ++e) {
    const int column = lane * 4 + e;
    const float gate = (float)gate_up[gate_base + column];
    const float up = (float)gate_up[width + gate_base + column];
    const float silu = (float)(__bf16)(gate * rs_sigmoid(gate));
    values[e] = (float)(__bf16)(silu * up);
    amax = fmaxf(amax, fabsf(values[e]));
  }
  for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
  const float scale = fmaxf(amax * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) scales[row * (width / 128) + group] = scale;
  for (int e = 0; e < 4; ++e)
    output[base + lane * 4 + e] = rs_fp8(fminf(fmaxf(values[e] / scale, -448.f), 448.f));
}

extern "C" __global__ __launch_bounds__(256) void ripple_draft_swiglu_bf16(
    const __bf16 *__restrict__ gate_up, __bf16 *__restrict__ output,
    int rows, int width) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * width) return;
  const int row = i / width;
  const float gate = (float)gate_up[(long)row * 2 * width + i - row * width];
  const float up = (float)gate_up[(long)row * 2 * width + width + i - row * width];
  const float silu = (float)(__bf16)(gate * rs_sigmoid(gate));
  output[i] = (__bf16)(silu * up);
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_add_rms_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ residual,
    const __bf16 *__restrict__ weight, __bf16 *__restrict__ output,
    __bf16 *__restrict__ residual_out, int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float red[RS_NT];
  const int tid = threadIdx.x;
  const long offset = (long)row * hidden;
  float sum = 0.f;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)(__bf16)((float)input[offset + i] + (float)residual[offset + i]);
    residual_out[offset + i] = (__bf16)x;
    sum = __builtin_fmaf(x, x, sum);
  }
  red[tid] = sum;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / hidden + eps);
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)(__bf16)((float)input[offset + i] + (float)residual[offset + i]);
    const float normalized = (float)(__bf16)(x * inv);
    output[offset + i] = (__bf16)(normalized * (float)weight[i]);
  }
}

__device__ __forceinline__ void ripple_conv4(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, int row, int column, int hidden,
    int groups, int block_size, int side, float *out) {
  const int group = column / 16;
  const int position = row & (block_size - 1);
  for (int e = 0; e < 4; ++e) out[e] = 0.f;
  for (int tap = 0; tap < 4; ++tap) {
    if (tap > 0 && position < tap) continue;
    for (int e = 0; e < 4; ++e) {
      const float xv = (float)x[(long)(row - tap) * hidden + column + e];
      const float bv = (float)base[((long)side * 4 + tap) * hidden + column + e];
      const float delta = (float)coeff[(((long)row * 2 + side) * 4 + tap) * groups + group];
      out[e] += (bv + delta) * xv;
    }
  }
}

extern "C" __global__ __launch_bounds__(128) void ripple_draft_conv4_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, __bf16 *__restrict__ output,
    int rows, int hidden, int groups, int block_size, int side) {
  const int index = (int)(blockIdx.x * 128 + threadIdx.x) * 4;
  const int row = index / hidden;
  const int column = index - row * hidden;
  if (row >= rows) return;
  float value[4];
  ripple_conv4(input, coeff, base, row, column, hidden, groups, block_size, side, value);
  for (int e = 0; e < 4; ++e) output[index + e] = (__bf16)value[e];
}

extern "C" __global__ __launch_bounds__(128) void ripple_draft_conv4_quant_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, unsigned char *__restrict__ output,
    float *__restrict__ scales, int rows, int hidden, int groups, int block_size, int side) {
  const int wave = (int)(blockIdx.x * 4 + (threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int quant_groups = hidden / 128;
  const int row = wave / quant_groups;
  const int group = wave - row * quant_groups;
  if (row >= rows) return;
  const int column = group * 128 + lane * 4;
  float value[4];
  ripple_conv4(input, coeff, base, row, column, hidden, groups, block_size, side, value);
  __bf16 rounded[4];
  float amax = 0.f;
  for (int e = 0; e < 4; ++e) {
    rounded[e] = (__bf16)value[e];
    amax = fmaxf(amax, fabsf((float)rounded[e]));
  }
  for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
  const float scale = fmaxf(amax * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) scales[row * quant_groups + group] = scale;
  for (int e = 0; e < 4; ++e)
    output[(long)row * hidden + column + e] =
        rs_fp8(fminf(fmaxf((float)rounded[e] / scale, -448.f), 448.f));
}

extern "C" __global__ __launch_bounds__(32) void ripple_draft_qk_rope_h128(
    const __bf16 *__restrict__ qkv, const __bf16 *__restrict__ qw,
    const __bf16 *__restrict__ kw, const __bf16 *__restrict__ cos_sin,
    const long *__restrict__ positions, __bf16 *__restrict__ q_out,
    __bf16 *__restrict__ k_out, __bf16 *__restrict__ v_out,
    int rows, int nq, int nkv, float eps, int apply_rope) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int q_width = nq * RS_HD;
  const int kv_width = nkv * RS_HD;
  const int qkv_width = q_width + 2 * kv_width;
  if (row >= rows || head >= nq + 2 * nkv) return;
  if (head >= nq + nkv) {
    const int local = head - nq - nkv;
    for (int e = 0; e < 4; ++e) {
      const int i = lane * 4 + e;
      v_out[((long)row * nkv + local) * RS_HD + i] =
          qkv[(long)row * qkv_width + q_width + kv_width + local * RS_HD + i];
    }
    return;
  }
  const bool is_k = head >= nq;
  const int local = is_k ? head - nq : head;
  const int source_offset = is_k ? q_width + local * RS_HD : local * RS_HD;
  const __bf16 *input = qkv + (long)row * qkv_width + source_offset;
  const __bf16 *weight = is_k ? kw : qw;
  const int heads = is_k ? nkv : nq;
  __bf16 *output = (is_k ? k_out : q_out) + ((long)row * heads + local) * RS_HD;
  float sum = 0.f;
  for (int e = 0; e < 4; ++e) {
    const float x = (float)input[lane * 4 + e];
    sum += x * x;
  }
  const float inv = rsqrtf(rs_wave_sum(sum) / RS_HD + eps);
  const __bf16 *table = cos_sin + positions[row] * (long)RS_HD;
  for (int e = 0; e < 4; ++e) {
    const int i = lane * 4 + e;
    const int partner = i < 64 ? i + 64 : i - 64;
    const int rotary = i < 64 ? i : i - 64;
    const float normalized = (float)(__bf16)((float)input[i] * inv);
    const float normalized_partner = (float)(__bf16)((float)input[partner] * inv);
    const float value = normalized * (float)weight[i];
    const float value_partner = normalized_partner * (float)weight[partner];
    const float c = (float)table[rotary];
    const float s = (float)table[64 + rotary];
    output[i] = apply_rope
        ? (__bf16)(i < 64 ? value * c - value_partner * s
                           : value * c + value_partner * s)
        : (__bf16)value;
  }
}

extern "C" __global__ __launch_bounds__(32) void ripple_draft_ctx_append_fp8(
    const __bf16 *__restrict__ context_kv, const __bf16 *__restrict__ kw,
    const __bf16 *__restrict__ cos_sin, const long *__restrict__ positions,
    unsigned char *__restrict__ kc, unsigned char *__restrict__ vc,
    float *__restrict__ kscales, float *__restrict__ vscales,
    int rows, int nkv, int context_width, int layer_kv_offset,
    float eps, int apply_rope, int window, int cache_capacity) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || head >= nkv) return;
  const long position = positions[row];
  const long slot = window > 0 ? position % cache_capacity : position;
  const __bf16 *key = context_kv + (long)row * context_width + layer_kv_offset + head * RS_HD;
  const __bf16 *value = key + nkv * RS_HD;
  const __bf16 *table = cos_sin + position * (long)RS_HD;
  const long target = (slot * nkv + head) * RS_HD;
  float sum = 0.f;
  for (int e = 0; e < 4; ++e) {
    const float x = (float)key[lane * 4 + e];
    sum += x * x;
  }
  const float inv = rsqrtf(rs_wave_sum(sum) / RS_HD + eps);
  float ka = 0.f, va = 0.f;
  float keys[4], values[4];
  for (int e = 0; e < 4; ++e) {
    const int i = lane * 4 + e;
    const int partner = i < 64 ? i + 64 : i - 64;
    const int rotary = i < 64 ? i : i - 64;
    const float normalized = (float)(__bf16)((float)key[i] * inv);
    const float normalized_partner = (float)(__bf16)((float)key[partner] * inv);
    const float x = normalized * (float)kw[i];
    const float xp = normalized_partner * (float)kw[partner];
    const float c = (float)table[rotary];
    const float s = (float)table[64 + rotary];
    keys[e] = apply_rope ? (i < 64 ? x * c - xp * s : x * c + xp * s) : x;
    values[e] = (float)value[i];
    ka = fmaxf(ka, fabsf((float)(__bf16)keys[e]));
    va = fmaxf(va, fabsf(values[e]));
  }
  for (int d = 16; d; d >>= 1) {
    ka = fmaxf(ka, __shfl_xor(ka, d, 32));
    va = fmaxf(va, __shfl_xor(va, d, 32));
  }
  const float ks = fmaxf(ka * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float vs = fmaxf(va * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) {
    kscales[slot * nkv + head] = ks;
    vscales[slot * nkv + head] = vs;
  }
  for (int e = 0; e < 4; ++e) {
    kc[target + lane * 4 + e] = rs_fp8((float)(__bf16)keys[e] / ks);
    vc[target + lane * 4 + e] = rs_fp8(values[e] / vs);
  }
}

extern "C" __global__ __launch_bounds__(32) void ripple_draft_attn_h128_fp8(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const float *__restrict__ kscales,
    const float *__restrict__ vscales, __bf16 *__restrict__ output,
    int rows, int nq, int nkv, int position, int window, int cache_capacity) {
  const int row = blockIdx.x;
  const int qh = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || qh >= nq) return;
  const int kvh = qh / (nq / nkv);
  const int query_position = position + row;
  const int first = window > 0 ? max(0, query_position - window + 1) : 0;
  const int last = window > 0 ? min(position + rows - 1, query_position + window - 1)
                              : position + rows - 1;
  const __bf16 *query = q + ((long)row * nq + qh) * RS_HD;
  float maximum = -INFINITY, denominator = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int key = first; key <= last; ++key) {
    const long cache = window > 0 ? key % cache_capacity : key;
    const long offset = (cache * nkv + kvh) * RS_HD;
    const float ks = kscales[cache * nkv + kvh];
    const float vs = vscales[cache * nkv + kvh];
    float dot = 0.f;
    for (int e = 0; e < 4; ++e)
      dot += (float)query[lane * 4 + e] * rs_from_fp8(kc[offset + lane * 4 + e]) * ks;
    const float score = rs_wave_sum(dot) * rsqrtf((float)RS_HD);
    const float next_maximum = fmaxf(maximum, score);
    const float rescale = __builtin_expf(maximum - next_maximum);
    const float probability = __builtin_expf(score - next_maximum);
    denominator = denominator * rescale + probability;
    for (int e = 0; e < 4; ++e)
      acc[e] = acc[e] * rescale + probability * rs_from_fp8(vc[offset + lane * 4 + e]) * vs;
    maximum = next_maximum;
  }
  for (int e = 0; e < 4; ++e)
    output[((long)row * nq + qh) * RS_HD + lane * 4 + e] = (__bf16)(acc[e] / denominator);
}

__device__ __forceinline__ bool ripple_top_better(float av, int ai, float bv, int bi) {
  return av > bv || (av == bv && ai < bi);
}

extern "C" __global__ __launch_bounds__(32) void ripple_draft_top32_bf16(
    const __bf16 *__restrict__ logits, float *__restrict__ values,
    long *__restrict__ indices, int rows, int width, int k) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows || k > 32) return;
  float previous_value = 0.f;
  int previous_index = -1;
  for (int pass = 0; pass < k; ++pass) {
    float best_value = -INFINITY;
    int best_index = 0x7fffffff;
    for (int i = lane; i < width; i += 32) {
      const float value = (float)logits[(long)row * width + i];
      if (pass > 0 && !(value < previous_value ||
                        (value == previous_value && i > previous_index))) continue;
      if (ripple_top_better(value, i, best_value, best_index)) {
        best_value = value;
        best_index = i;
      }
    }
    for (int d = 16; d; d >>= 1) {
      const float other_value = __shfl_xor(best_value, d, 32);
      const int other_index = __shfl_xor(best_index, d, 32);
      if (ripple_top_better(other_value, other_index, best_value, best_index)) {
        best_value = other_value;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      values[row * k + pass] = best_value;
      indices[row * k + pass] = best_index;
    }
    previous_value = best_value;
    previous_index = best_index;
  }
}

extern "C" __global__ __launch_bounds__(32) void ripple_draft_score32_r256(
    const __bf16 *__restrict__ pred_cb, const __bf16 *__restrict__ succ_cb,
    const long *__restrict__ candidates, const int *__restrict__ anchor,
    const float *__restrict__ unary, const __bf16 *__restrict__ hidden,
    float *__restrict__ output, int positions, int l_len, int r, int k_top) {
  const int item = blockIdx.x;
  const int lane = threadIdx.x;
  if (item >= positions * k_top * k_top || r != 256 || k_top != 32 || l_len != positions) return;
  const int position = item / (k_top * k_top);
  const int remainder = item - position * (k_top * k_top);
  const int predecessor = remainder / k_top;
  const int candidate = remainder - predecessor * k_top;
  const long pid = position == 0 ? (long)anchor[0]
                                 : candidates[(long)(position - 1) * k_top + predecessor];
  const long cid = candidates[(long)position * k_top + candidate];
  float sum = 0.f;
  for (int e = 0; e < 8; ++e) {
    const int rank = lane * 8 + e;
    const float a = (float)(__bf16)((float)pred_cb[pid * r + rank] *
                                    (float)hidden[(long)position * r + rank]);
    sum += a * (float)succ_cb[cid * r + rank];
  }
  sum = rs_wave_sum(sum);
  if (lane == 0)
    output[item] = (float)(__bf16)sum + unary[(long)position * k_top + candidate];
}
