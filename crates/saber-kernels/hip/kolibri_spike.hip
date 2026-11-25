#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

extern "C" __global__ __launch_bounds__(256) void kolibri_router_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ weight,
    const float *__restrict__ bias, int *__restrict__ ids,
    float *__restrict__ weights, int rows, int hidden, int experts, int top_k) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows || experts > 384) return;
  __shared__ float logits[384];
  __shared__ unsigned char selected[384];
  for (int expert = tid; expert < experts; expert += 256) {
    float sum = 0.f;
    for (int k = 0; k < hidden; ++k)
      sum += (float)input[(long)row * hidden + k] *
             (float)weight[(long)expert * hidden + k];
    logits[expert] = sum;
    selected[expert] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    for (int slot = 0; slot < top_k; ++slot) {
      int best = -1;
      float best_score = -INFINITY;
      for (int expert = 0; expert < experts; ++expert) {
        const float score = logits[expert] + bias[expert];
        if (!selected[expert] && (best < 0 || score > best_score)) {
          best = expert;
          best_score = score;
        }
      }
      selected[best] = 1;
      ids[row * top_k + slot] = best;
      weights[row * top_k + slot] = 1.f / (1.f + __builtin_expf(-logits[best]));
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void kolibri_swiglu_bf16(
    const __bf16 *__restrict__ gate, const __bf16 *__restrict__ up,
    __bf16 *__restrict__ output, int elements) {
  const int index = blockIdx.x * 256 + threadIdx.x;
  if (index >= elements) return;
  const float x = (float)gate[index];
  const float activated = (float)(__bf16)(x / (1.f + __builtin_expf(-x)));
  output[index] = (__bf16)(activated * (float)up[index]);
}

extern "C" __global__ __launch_bounds__(256) void kolibri_accumulate_bf16(
    __bf16 *__restrict__ accumulated, const __bf16 *__restrict__ input,
    float scale, int elements) {
  const int index = blockIdx.x * 256 + threadIdx.x;
  if (index >= elements) return;
  const float scaled = (float)(__bf16)((float)input[index] * scale);
  accumulated[index] = (__bf16)((float)accumulated[index] + scaled);
}

extern "C" __global__ __launch_bounds__(256) void kolibri_quant_bf16(
    const __bf16 *__restrict__ input, unsigned char *__restrict__ output,
    float *__restrict__ scales, int rows, int hidden) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  __shared__ float red[256];
  float amax = 0.f;
  for (int i = tid; i < hidden; i += 256)
    amax = fmaxf(amax, fabsf((float)input[(long)row * hidden + i]));
  red[tid] = amax;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(red[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float reciprocal = 1.f / scale;
  if (tid == 0) scales[row] = scale;
  for (int i = tid; i < hidden; i += 256) {
    const float value = (float)input[(long)row * hidden + i] * reciprocal;
    output[(long)row * hidden + i] =
        (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(
                            fminf(fmaxf(value, -448.f), 448.f),
                            fminf(fmaxf(value, -448.f), 448.f), 0u, false) & 0xffu);
  }
}

extern "C" __global__ __launch_bounds__(256) void kolibri_route_bf16(
    const __bf16 *__restrict__ input, const __bf16 *__restrict__ weight,
    const float *__restrict__ bias, int *__restrict__ ids,
    float *__restrict__ weights, int *__restrict__ route_pos,
    int *__restrict__ row_tok, int *__restrict__ tile_expert,
    int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta,
    int rows, int hidden, int experts, int top_k) {
  if (rows > 16 || experts > 384 || top_k > 6) return;
  __shared__ float logits[16 * 384];
  __shared__ int counts[384], fill[384], offsets[385], tile_offsets[384];
  const int tid = threadIdx.x;
  for (int index = tid; index < rows * experts; index += 256) {
    const int row = index / experts;
    const int expert = index - row * experts;
    float sum = 0.f;
    for (int k = 0; k < hidden; ++k)
      sum += (float)input[(long)row * hidden + k] *
             (float)weight[(long)expert * hidden + k];
    logits[index] = sum;
  }
  for (int expert = tid; expert < experts; expert += 256) {
    counts[expert] = 0;
    fill[expert] = 0;
  }
  __syncthreads();
  if (tid < rows) {
    const int row = tid;
    for (int slot = 0; slot < top_k; ++slot) {
      int best = -1;
      float best_score = -INFINITY;
      for (int expert = 0; expert < experts; ++expert) {
        const float score = logits[row * experts + expert] + bias[expert];
        if (best < 0 || score > best_score) {
          best = expert;
          best_score = score;
        }
      }
      const float selected = logits[row * experts + best];
      ids[row * top_k + slot] = best;
      weights[row * top_k + slot] = 1.f / (1.f + __builtin_expf(-selected));
      logits[row * experts + best] = -INFINITY;
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int route = 0; route < rows * top_k; ++route) ++counts[ids[route]];
    int row_offset = 0, tile_offset = 0, touched = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int blocks = (counts[expert] + 15) / 16;
      offsets[expert] = row_offset;
      tile_offsets[expert] = tile_offset;
      row_offset += blocks * 16;
      tile_offset += blocks;
      touched += counts[expert] > 0;
      expert_cnt[expert] = counts[expert];
    }
    offsets[experts] = row_offset;
    meta[0] = tile_offset + (rows + 15) / 16;
    meta[1] = row_offset + (rows + 15) / 16 * 16;
    meta[2] = row_offset;
    meta[3] = rows * top_k;
    meta[4] = touched;
    meta[5] = row_offset;
    for (int route = 0; route < rows * top_k; ++route) {
      const int expert = ids[route];
      const int position = offsets[expert] + fill[expert]++;
      route_pos[route] = position;
      row_tok[position] = route / top_k;
    }
    for (int row = 0; row < rows; ++row) row_tok[row_offset + row] = row;
    for (int row = rows; row < (rows + 15) / 16 * 16; ++row)
      row_tok[row_offset + row] = -1;
  }
  __syncthreads();
  for (int expert = tid; expert < experts; expert += 256) {
    const int count = counts[expert];
    const int blocks = (count + 15) / 16;
    for (int block = 0; block < blocks; ++block) {
      const int tile = tile_offsets[expert] + block;
      tile_expert[tile] = expert;
      tile_row0[tile] = offsets[expert] + block * 16;
      tile_rows[tile] = min(16, count - block * 16);
    }
    for (int row = offsets[expert] + count; row < offsets[expert] + blocks * 16; ++row)
      row_tok[row] = -1;
  }
  for (int tile = meta[0] - (rows + 15) / 16 + tid; tile < meta[0]; tile += 256) {
    const int local = tile - (meta[0] - (rows + 15) / 16);
    tile_expert[tile] = experts;
    tile_row0[tile] = meta[2] + local * 16;
    tile_rows[tile] = min(16, rows - local * 16);
  }
}

extern "C" __global__ __launch_bounds__(256) void kolibri_sort_top6(
    const int *__restrict__ ids, int *__restrict__ route_pos,
    int *__restrict__ row_tok, int *__restrict__ tile_expert,
    int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta,
    int rows, int experts, int top_k) {
  if (rows > 256 || experts > 384 || top_k > 6) return;
  __shared__ int counts[384], fill[384], offsets[385], tile_offsets[384];
  const int tid = threadIdx.x;
  for (int expert = tid; expert < experts; expert += 256) {
    counts[expert] = 0;
    fill[expert] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    for (int route = 0; route < rows * top_k; ++route) ++counts[ids[route]];
    int row_offset = 0, tile_offset = 0, touched = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int blocks = (counts[expert] + 15) / 16;
      offsets[expert] = row_offset;
      tile_offsets[expert] = tile_offset;
      row_offset += blocks * 16;
      tile_offset += blocks;
      touched += counts[expert] > 0;
      expert_cnt[expert] = counts[expert];
    }
    offsets[experts] = row_offset;
    meta[0] = tile_offset + (rows + 15) / 16;
    meta[1] = row_offset + (rows + 15) / 16 * 16;
    meta[2] = row_offset;
    meta[3] = rows * top_k;
    meta[4] = touched;
    meta[5] = row_offset;
    for (int route = 0; route < rows * top_k; ++route) {
      const int expert = ids[route];
      const int position = offsets[expert] + fill[expert]++;
      route_pos[route] = position;
      row_tok[position] = route / top_k;
    }
    for (int row = 0; row < rows; ++row) row_tok[row_offset + row] = row;
    for (int row = rows; row < (rows + 15) / 16 * 16; ++row)
      row_tok[row_offset + row] = -1;
  }
  __syncthreads();
  for (int expert = tid; expert < experts; expert += 256) {
    const int count = counts[expert];
    const int blocks = (count + 15) / 16;
    for (int block = 0; block < blocks; ++block) {
      const int tile = tile_offsets[expert] + block;
      tile_expert[tile] = expert;
      tile_row0[tile] = offsets[expert] + block * 16;
      tile_rows[tile] = min(16, count - block * 16);
    }
    for (int row = offsets[expert] + count; row < offsets[expert] + blocks * 16; ++row)
      row_tok[row] = -1;
  }
  for (int tile = meta[0] - (rows + 15) / 16 + tid; tile < meta[0]; tile += 256) {
    const int local = tile - (meta[0] - (rows + 15) / 16);
    tile_expert[tile] = experts;
    tile_row0[tile] = meta[2] + local * 16;
    tile_rows[tile] = min(16, rows - local * 16);
  }
}

extern "C" __global__ __launch_bounds__(256) void kolibri_combine_bf16(
    const __bf16 *__restrict__ y, const float *__restrict__ weights,
    const int *__restrict__ route_pos, const int *__restrict__ meta,
    __bf16 *__restrict__ output, int rows, int hidden, int top_k) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int column = threadIdx.x; column < hidden; column += 256) {
    __bf16 accumulated = (__bf16)0.f;
    for (int slot = 0; slot < top_k; ++slot) {
      const int route = row * top_k + slot;
      const __bf16 value = y[(long)route_pos[route] * hidden + column];
      const __bf16 weighted = (__bf16)(weights[route] * (float)value);
      accumulated = (__bf16)((float)accumulated + (float)weighted);
    }
    const __bf16 shared = y[(long)(meta[2] + row) * hidden + column];
    output[(long)row * hidden + column] =
        (__bf16)((float)accumulated + (float)shared);
  }
}

extern "C" __global__ __launch_bounds__(32) void kolibri_attn_decode_partial_h128_fp8(
    const __bf16 *__restrict__ q, const unsigned char *__restrict__ kc,
    const unsigned char *__restrict__ vc, const float *__restrict__ kscales,
    const float *__restrict__ vscales, float *__restrict__ partial,
    float *__restrict__ partial_ml, int rows, int nq, int nkv, int start_pos,
    int window, int cache_capacity, int splits) {
  const int row = blockIdx.x;
  const int qh = blockIdx.y;
  const int split = blockIdx.z;
  const int lane = threadIdx.x;
  if (row >= rows || qh >= nq || split >= splits) return;
  const int position = start_pos + row;
  const int first = window > 0 ? max(0, position - window + 1) : 0;
  const int count = position - first + 1;
  const int per_split = (count + splits - 1) / splits;
  const int begin = first + split * per_split;
  const int end = min(position + 1, begin + per_split);
  const int kvh = qh / (nq / nkv);
  const __bf16 *query = q + ((long)row * nq + qh) * 128;
  float maximum = -INFINITY;
  float denominator = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int key = begin; key < end; ++key) {
    const long cache = window > 0 ? key % cache_capacity : key;
    const long head_offset = (cache * nkv + kvh) * 128;
    const float ks = kscales[cache * nkv + kvh];
    const float vs = vscales[cache * nkv + kvh];
    float dot = 0.f;
    for (int e = 0; e < 4; ++e)
      dot += (float)query[lane * 4 + e] *
             __builtin_amdgcn_cvt_f32_fp8((int)kc[head_offset + lane * 4 + e], 0) * ks;
    for (int d = 1; d < 32; d <<= 1) dot += __shfl_xor(dot, d, 32);
    const float score = dot * rsqrtf(128.f);
    const float next_maximum = fmaxf(maximum, score);
    const float rescale = __builtin_expf(maximum - next_maximum);
    const float probability = __builtin_expf(score - next_maximum);
    denominator = denominator * rescale + probability;
    for (int e = 0; e < 4; ++e)
      acc[e] = acc[e] * rescale + probability *
               __builtin_amdgcn_cvt_f32_fp8((int)vc[head_offset + lane * 4 + e], 0) * vs;
    maximum = next_maximum;
  }
  const long base = (((long)row * nq + qh) * splits + split) * 128 + lane * 4;
  for (int e = 0; e < 4; ++e) partial[base + e] = acc[e];
  if (lane == 0) {
    const long ml = ((long)row * nq + qh) * splits * 2 + split * 2;
    partial_ml[ml] = maximum;
    partial_ml[ml + 1] = denominator;
  }
}

extern "C" __global__ __launch_bounds__(32) void kolibri_attn_decode_reduce_h128(
    const float *__restrict__ partial, const float *__restrict__ partial_ml,
    __bf16 *__restrict__ output, int rows, int nq, int splits) {
  const int row = blockIdx.x;
  const int qh = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || qh >= nq) return;
  float maximum = -INFINITY;
  for (int split = 0; split < splits; ++split) {
    const long ml = ((long)row * nq + qh) * splits * 2 + split * 2;
    maximum = fmaxf(maximum, partial_ml[ml]);
  }
  float denominator = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int split = 0; split < splits; ++split) {
    const long ml = ((long)row * nq + qh) * splits * 2 + split * 2;
    const float rescale = __builtin_expf(partial_ml[ml] - maximum);
    denominator += partial_ml[ml + 1] * rescale;
    const long base = (((long)row * nq + qh) * splits + split) * 128 + lane * 4;
    for (int e = 0; e < 4; ++e) acc[e] += partial[base + e] * rescale;
  }
  for (int e = 0; e < 4; ++e)
    output[((long)row * nq + qh) * 128 + lane * 4 + e] =
        (__bf16)(acc[e] / denominator);
}
