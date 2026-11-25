#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define RS_HD 128
#define RS_WAVE 32
#define RS_NT 256

__device__ __forceinline__ float rs_sigmoid(float x) {
  return 1.f / (1.f + __builtin_expf(-x));
}

__device__ __forceinline__ float rs_wave_sum(float value) {
  for (int d = 1; d < RS_WAVE; d <<= 1)
    value += __shfl_xor(value, d, RS_WAVE);
  return value;
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_rmsnorm_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ weight,
    __bf16 *__restrict__ output, int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float red[RS_NT];
  const int tid = threadIdx.x;
  float sum = 0.f;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)input[(long)row * hidden + i];
    sum += x * x;
  }
  red[tid] = sum;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const float inv = 1.f / sqrtf(red[0] / hidden + eps);
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)input[(long)row * hidden + i];
    const float normalized = (float)(__bf16)(x * inv);
    output[(long)row * hidden + i] = (__bf16)(normalized * (float)weight[i]);
  }
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_postnorm_residual_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ residual,
    const __bf16 *__restrict__ weight, __bf16 *__restrict__ output,
    int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float red[RS_NT];
  const int tid = threadIdx.x;
  float sum = 0.f;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)input[(long)row * hidden + i];
    sum += x * x;
  }
  red[tid] = sum;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const float inv = 1.f / sqrtf(red[0] / hidden + eps);
  for (int i = tid; i < hidden; i += RS_NT) {
    const long offset = (long)row * hidden + i;
    const float x = (float)input[offset];
    const float normalized = (float)(__bf16)(x * inv);
    const float normed = (float)(__bf16)(normalized * (float)weight[i]);
    output[offset] = (__bf16)((float)residual[offset] + normed);
  }
}

extern "C" __global__ __launch_bounds__(256) void ripple_value_lerp_bf16(
    const __bf16 *__restrict__ initial, const __bf16 *__restrict__ current,
    const __bf16 *__restrict__ lambda, __bf16 *__restrict__ output, int elements) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= elements) return;
  const float blend = (float)(__bf16)rs_sigmoid((float)lambda[0]);
  output[i] = (__bf16)((float)initial[i] + blend * ((float)current[i] - (float)initial[i]));
}

extern "C" __global__ __launch_bounds__(RS_WAVE) void ripple_qk_norm_rope_h128(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ k,
    const __bf16 *__restrict__ qw, const __bf16 *__restrict__ kw,
    const __bf16 *__restrict__ cos_sin, const long *__restrict__ positions,
    __bf16 *__restrict__ q_out, __bf16 *__restrict__ k_out,
    int rows, int nq, int nkv, float eps, int apply_rope) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || head >= nq + nkv) return;
  const bool is_k = head >= nq;
  const int local = is_k ? head - nq : head;
  const int heads = is_k ? nkv : nq;
  const __bf16 *input = (is_k ? k : q) + ((long)row * heads + local) * RS_HD;
  const __bf16 *weight = is_k ? kw : qw;
  __bf16 *output = (is_k ? k_out : q_out) + ((long)row * heads + local) * RS_HD;
  float sum = 0.f;
  for (int e = 0; e < 4; ++e) {
    const float x = (float)input[lane * 4 + e];
    sum += x * x;
  }
  const float inv = 1.f / sqrtf(rs_wave_sum(sum) / RS_HD + eps);
  const __bf16 *table = cos_sin + positions[row] * (long)RS_HD;
  for (int e = 0; e < 4; ++e) {
    const int i = lane * 4 + e;
    const int partner = i < 64 ? i + 64 : i - 64;
    const int rotary = i < 64 ? i : i - 64;
    const float x = (float)input[i];
    const float xp = (float)input[partner];
    const float normalized = (float)(__bf16)(x * inv);
    const float normalized_partner = (float)(__bf16)(xp * inv);
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

extern "C" __global__ __launch_bounds__(256) void ripple_kv_append_h128_bf16(
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    __bf16 *__restrict__ kc, __bf16 *__restrict__ vc,
    const long *__restrict__ slots, int rows, int nkv) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long slot = slots[row];
  if (slot < 0) return;
  for (int i = threadIdx.x; i < nkv * RS_HD; i += 256) {
    const long source = (long)row * nkv * RS_HD + i;
    const long target = slot * nkv * RS_HD + i;
    kc[target] = key[source];
    vc[target] = value[source];
  }
}

extern "C" __global__ __launch_bounds__(RS_WAVE) void ripple_attn_h128_bf16(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ kc,
    const __bf16 *__restrict__ vc, __bf16 *__restrict__ output,
    int rows, int nq, int nkv, int start_pos, int window) {
  const int row = blockIdx.x;
  const int qh = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || qh >= nq) return;
  const int gqa = nq / nkv;
  const int kvh = qh / gqa;
  const int position = start_pos + row;
  const int first = window > 0 ? max(0, position - window + 1) : 0;
  const __bf16 *query = q + ((long)row * nq + qh) * RS_HD;
  float maximum = -INFINITY;
  float denominator = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int key = first; key <= position; ++key) {
    const long cache = (window > 0 ? key % window : key);
    const __bf16 *k = kc + (cache * nkv + kvh) * RS_HD;
    float dot = 0.f;
    for (int e = 0; e < 4; ++e)
      dot += (float)query[lane * 4 + e] * (float)k[lane * 4 + e];
    const float score = rs_wave_sum(dot) * rsqrtf((float)RS_HD);
    const float next_maximum = fmaxf(maximum, score);
    const float rescale = __builtin_expf(maximum - next_maximum);
    const float probability = __builtin_expf(score - next_maximum);
    denominator = denominator * rescale + probability;
    const __bf16 *v = vc + (cache * nkv + kvh) * RS_HD;
    for (int e = 0; e < 4; ++e)
      acc[e] = acc[e] * rescale + probability * (float)v[lane * 4 + e];
    maximum = next_maximum;
  }
  for (int e = 0; e < 4; ++e)
    output[((long)row * nq + qh) * RS_HD + lane * 4 + e] =
        (__bf16)(acc[e] / denominator);
}

extern "C" __global__ __launch_bounds__(256) void ripple_gate_mul_bf16(
    const __bf16 *__restrict__ attention, const __bf16 *__restrict__ gate,
    __bf16 *__restrict__ output, int elements) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= elements) return;
  const float g = (float)(__bf16)rs_sigmoid((float)gate[i]);
  output[i] = (__bf16)((float)attention[i] * g);
}

extern "C" __global__ __launch_bounds__(256) void ripple_scale_bf16(
    __bf16 *__restrict__ values, int elements, float scale) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < elements) values[i] = (__bf16)((float)values[i] * scale);
}

__device__ __forceinline__ unsigned char rs_fp8(float x) {
  return (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(x, x, 0u, false) & 0xffu);
}

__device__ __forceinline__ float rs_from_fp8(unsigned char x) {
  return __builtin_amdgcn_cvt_f32_fp8((int)x, 0);
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_rms_quant_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ weight,
    unsigned char *__restrict__ output, float *__restrict__ scales,
    int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  __shared__ float red[RS_NT];
  const __bf16 *src = input + (long)row * hidden;
  unsigned char *dst = output + (long)row * hidden;
  float sum = 0.f;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float x = (float)src[i];
    sum = __builtin_fmaf(x, x, sum);
  }
  red[tid] = sum;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / hidden + eps);
  __syncthreads();
  float amax = 0.f;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float normalized = (float)(__bf16)((float)src[i] * inv);
    const float value = (float)(__bf16)(normalized * (float)weight[i]);
    amax = fmaxf(amax, fabsf(value));
  }
  red[tid] = amax;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(red[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float reciprocal = 1.f / scale;
  if (tid == 0) scales[row] = scale;
  for (int i = tid; i < hidden; i += RS_NT) {
    const float normalized = (float)(__bf16)((float)src[i] * inv);
    const float value = (float)(__bf16)(normalized * (float)weight[i]);
    dst[i] = rs_fp8(fminf(fmaxf(value * reciprocal, -448.f), 448.f));
  }
}

extern "C" __global__ __launch_bounds__(RS_NT) void ripple_swiglu_quant_bf16(
    const __bf16 *__restrict__ gate_up, unsigned char *__restrict__ output,
    float *__restrict__ scales, int rows, int width, float clamp_threshold) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  __shared__ float red[RS_NT];
  const __bf16 *src = gate_up + (long)row * 2 * width;
  unsigned char *dst = output + (long)row * width;
  float amax = 0.f;
  for (int i = tid; i < width; i += RS_NT) {
    const float gate = fminf((float)src[i], clamp_threshold);
    const float up = fminf(fmaxf((float)src[width + i], -clamp_threshold), clamp_threshold);
    const float silu = (float)(__bf16)(gate * rs_sigmoid(gate));
    const float value = (float)(__bf16)(silu * up);
    amax = fmaxf(amax, fabsf(value));
  }
  red[tid] = amax;
  __syncthreads();
  for (int stride = RS_NT / 2; stride; stride >>= 1) {
    if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(red[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float reciprocal = 1.f / scale;
  if (tid == 0) scales[row] = scale;
  for (int i = tid; i < width; i += RS_NT) {
    const float gate = fminf((float)src[i], clamp_threshold);
    const float up = fminf(fmaxf((float)src[width + i], -clamp_threshold), clamp_threshold);
    const float silu = (float)(__bf16)(gate * rs_sigmoid(gate));
    const float value = (float)(__bf16)(silu * up);
    dst[i] = rs_fp8(fminf(fmaxf(value * reciprocal, -448.f), 448.f));
  }
}

extern "C" __global__ __launch_bounds__(RS_WAVE) void ripple_kv_append_h128_fp8(
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    unsigned char *__restrict__ kc, unsigned char *__restrict__ vc,
    float *__restrict__ kscales, float *__restrict__ vscales,
    const long *__restrict__ slots, int rows, int nkv, int window, int cache_capacity) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || head >= nkv) return;
  const long position = slots[row];
  if (position < 0) return;
  const long slot = window > 0 ? position % cache_capacity : position;
  const long source = ((long)row * nkv + head) * RS_HD;
  const long target = (slot * nkv + head) * RS_HD;
  float ka = 0.f, va = 0.f;
  for (int e = 0; e < 4; ++e) {
    ka = fmaxf(ka, fabsf((float)key[source + lane * 4 + e]));
    va = fmaxf(va, fabsf((float)value[source + lane * 4 + e]));
  }
  for (int d = 16; d; d >>= 1) {
    ka = fmaxf(ka, __shfl_xor(ka, d, RS_WAVE));
    va = fmaxf(va, __shfl_xor(va, d, RS_WAVE));
  }
  const float ks = fmaxf(ka * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float vs = fmaxf(va * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) {
    kscales[slot * nkv + head] = ks;
    vscales[slot * nkv + head] = vs;
  }
  for (int e = 0; e < 4; ++e) {
    kc[target + lane * 4 + e] = rs_fp8((float)key[source + lane * 4 + e] / ks);
    vc[target + lane * 4 + e] = rs_fp8((float)value[source + lane * 4 + e] / vs);
  }
}

extern "C" __global__ __launch_bounds__(RS_WAVE) void ripple_attn_h128_fp8(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const float *__restrict__ kscales,
    const float *__restrict__ vscales, __bf16 *__restrict__ output,
    int rows, int nq, int nkv, int start_pos, int window, int cache_capacity) {
  const int row = blockIdx.x;
  const int qh = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || qh >= nq) return;
  const int gqa = nq / nkv;
  const int kvh = qh / gqa;
  const int position = start_pos + row;
  const int first = window > 0 ? max(0, position - window + 1) : 0;
  const __bf16 *query = q + ((long)row * nq + qh) * RS_HD;
  float maximum = -INFINITY;
  float denominator = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int key = first; key <= position; ++key) {
    const long cache = (window > 0 ? key % cache_capacity : key);
    const long head_offset = (cache * nkv + kvh) * RS_HD;
    const float ks = kscales[cache * nkv + kvh];
    const float vs = vscales[cache * nkv + kvh];
    float dot = 0.f;
    for (int e = 0; e < 4; ++e)
      dot += (float)query[lane * 4 + e] * rs_from_fp8(kc[head_offset + lane * 4 + e]) * ks;
    const float score = rs_wave_sum(dot) * rsqrtf((float)RS_HD);
    const float next_maximum = fmaxf(maximum, score);
    const float rescale = __builtin_expf(maximum - next_maximum);
    const float probability = __builtin_expf(score - next_maximum);
    denominator = denominator * rescale + probability;
    for (int e = 0; e < 4; ++e)
      acc[e] = acc[e] * rescale + probability * rs_from_fp8(vc[head_offset + lane * 4 + e]) * vs;
    maximum = next_maximum;
  }
  for (int e = 0; e < 4; ++e)
    output[((long)row * nq + qh) * RS_HD + lane * 4 + e] =
        (__bf16)(acc[e] / denominator);
}

extern "C" __global__ __launch_bounds__(256) void ripple_aux_capture_bf16(
    const __bf16 *__restrict__ input, __bf16 *__restrict__ output,
    int rows, int hidden, int tap) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * hidden) return;
  const int row = i / hidden;
  const int column = i - row * hidden;
  output[((long)row * 5 + tap) * hidden + column] = input[i];
}
