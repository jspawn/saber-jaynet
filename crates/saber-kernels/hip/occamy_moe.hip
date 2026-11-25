#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define MOE_EXPERTS 256
#define MOE_TOPK 8
#define MOE_TPB 32
#define MOE_THREADS 256

static __device__ __forceinline__ float f32_bits(unsigned int u) { return __uint_as_float(u); }

static __device__ __forceinline__ float saber_exp_nonpos(float x) {
  if (!(x > -80.f)) return 0.f;
  const float n = __builtin_rintf(x * f32_bits(0x3FB8AA3Bu));
  float r = x - n * f32_bits(0x3F317200u);
  r = r - n * f32_bits(0x35BFBE8Eu);
  float p = f32_bits(0x39500D01u);
  p = p * r + f32_bits(0x3AB60B61u);
  p = p * r + f32_bits(0x3C088889u);
  p = p * r + f32_bits(0x3D2AAAABu);
  p = p * r + f32_bits(0x3E2AAAABu);
  p = p * r + 0.5f;
  p = p * r + 1.f;
  p = p * r + 1.f;
  return __builtin_ldexpf(p, (int)n);
}

static __device__ __forceinline__ float saber_sigmoid(float g) {
  const float e = saber_exp_nonpos(-fabsf(g));
  const float d = 1.f + e;
  return g >= 0.f ? 1.f / d : e / d;
}

static __device__ __forceinline__ int block_exscan(int v, int *s_wave, int *total) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up(x, o, 32);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_wave[wave] = x;
  __syncthreads();
  int base = 0, all = 0;
#pragma unroll
  for (int w = 0; w < MOE_THREADS / 32; ++w) {
    const int s = s_wave[w];
    if (w < wave) base += s;
    all += s;
  }
  __syncthreads();
  *total = all;
  return base + x - v;
}

extern "C" __global__ __launch_bounds__(MOE_THREADS) void occamy_moe_route_topk8_sort_e256(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  __shared__ int s_ids[MOE_TPB * MOE_TOPK];
  __shared__ int s_cnt[MOE_EXPERTS];
  __shared__ int s_off[MOE_EXPERTS + 1], s_toff[MOE_EXPERTS + 1];
  __shared__ int s_wave[MOE_THREADS / 32];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int nblk = gridDim.x, blk = blockIdx.x, t0 = blk * MOE_TPB;

  for (int e = tid; e < MOE_EXPERTS; e += MOE_THREADS) s_cnt[e] = 0;
  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) s_ids[i] = -1;
  __syncthreads();

  for (int i = 0; i < MOE_TPB / (MOE_THREADS / 32); ++i) {
    const int tl = wave * (MOE_TPB / (MOE_THREADS / 32)) + i, t = t0 + tl;
    if (t >= T) break;
    float p[16];
    if (lane < 16) {
      const uint4 *src = (const uint4 *)(logits + (size_t)t * MOE_EXPERTS + lane * 16);
      union { uint4 u; __bf16 h[8]; } v0, v1;
      v0.u = src[0];
      v1.u = src[1];
#pragma unroll
      for (int j = 0; j < 8; ++j) { p[j] = (float)v0.h[j]; p[8 + j] = (float)v1.h[j]; }
    } else {
#pragma unroll
      for (int j = 0; j < 16; ++j) p[j] = -__builtin_inff();
    }
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + __shfl_xor(s, o, 32);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;
      p[j] = q;
    }
    float sel = 0.f, wk[MOE_TOPK];
    int ik[MOE_TOPK];
#pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
      float bv = p[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (p[j] > bv) { bv = p[j]; bj = j; }
      int bi = lane * 16 + bj;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi >> 4) == lane) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j == (bi & 15)) p[j] = -1.f;
      }
      wk[k] = bv;
      ik[k] = bi;
      sel = sel + bv;
    }
    const float scale = 1.f / (sel > 0.f ? sel : 1.f);
    if (lane < MOE_TOPK) {
      float wv = 0.f;
      int iv = 0;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k)
        if (k == lane) { wv = wk[k]; iv = ik[k]; }
      topk_ids[(size_t)t * MOE_TOPK + lane] = iv;
      topk_w[(size_t)t * MOE_TOPK + lane] = wv * scale;
      s_ids[tl * MOE_TOPK + lane] = iv;
    }
  }
  __syncthreads();

  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) {
    const int e = s_ids[i];
    const int le = e - e_lo;
    if (e < 0 || le < 0 || le >= e_loc) continue;
    atomicAdd(&s_cnt[le], 1);
    const int tl = i / MOE_TOPK;
    int rk = 0;
    for (int u = 0; u < tl * MOE_TOPK; ++u) rk += s_ids[u] == e;
    rank[(size_t)t0 * MOE_TOPK + i] = rk;
  }
  __syncthreads();
  for (int e = tid; e < e_loc; e += MOE_THREADS) blk_cnt[(size_t)blk * e_loc + e] = s_cnt[e];
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(arrive, 1) == nblk - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) *arrive = 0;

  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    int run = 0;
    for (int b = 0; b < nblk; ++b) {
      const int c = blk_cnt[(size_t)b * e_loc + e];
      blk_cnt[(size_t)b * e_loc + e] = run;
      run += c;
    }
    s_cnt[e] = run;
    expert_cnt[e] = run;
  }
  __syncthreads();
  const int G = tile_blocks, TR = 16 * tile_blocks;
  int r2 = 0, t2 = 0, touched = 0, routes = 0;
  for (int h = 0; h < 2; ++h) {
    const int e = 2 * tid + h;
    if (e < e_loc) {
      const int c = s_cnt[e], nb = (c + 15) / 16;
      r2 += nb * 16;
      t2 += (nb + G - 1) / G;
      touched += c > 0;
      routes += c;
    }
  }
  int rows_total, tiles_total, touched_total, routes_total;
  const int rbase = block_exscan(r2, s_wave, &rows_total);
  const int tbase = block_exscan(t2, s_wave, &tiles_total);
  block_exscan(touched, s_wave, &touched_total);
  block_exscan(routes, s_wave, &routes_total);
  {
    int ro = rbase, to = tbase;
    for (int h = 0; h < 2; ++h) {
      const int e = 2 * tid + h;
      if (e < e_loc) {
        const int c = s_cnt[e], nb = (c + 15) / 16;
        s_off[e] = ro;
        s_toff[e] = to;
        ro += nb * 16;
        to += (nb + G - 1) / G;
      }
    }
  }
  __syncthreads();
  const int sh_row0 = rows_total;
  const int sh_blocks = with_shared ? (T + 15) / 16 : 0;
  const int sh_tiles = (sh_blocks + G - 1) / G;
  if (tid == 0) {
    meta[0] = tiles_total + sh_tiles;
    meta[1] = rows_total + sh_blocks * 16;
    meta[2] = with_shared ? sh_row0 : -1;
    meta[3] = routes_total;
    meta[4] = touched_total;
    meta[5] = rows_total;
  }
  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    const int c = s_cnt[e], nb = (c + 15) / 16, nt = (nb + G - 1) / G;
    for (int j = 0; j < nt; ++j) {
      tile_expert[s_toff[e] + j] = e;
      tile_row0[s_toff[e] + j] = s_off[e] + j * TR;
      tile_rows[s_toff[e] + j] = min(TR, c - j * TR);
    }
    for (int r = s_off[e] + c; r < s_off[e] + nb * 16; ++r) { row_tok[r] = -1; row_route[r] = -1; }
  }
  for (int j = tid; j < sh_tiles; j += MOE_THREADS) {
    tile_expert[tiles_total + j] = e_loc;
    tile_row0[tiles_total + j] = sh_row0 + j * TR;
    tile_rows[tiles_total + j] = min(TR, T - j * TR);
  }
  for (int r = tid; r < sh_blocks * 16; r += MOE_THREADS) {
    row_tok[sh_row0 + r] = r < T ? r : -1;
    row_route[sh_row0 + r] = -1;
  }
  for (int r = tid; r < T * MOE_TOPK; r += MOE_THREADS) {
    const int e = topk_ids[r], le = e - e_lo;
    if (le < 0 || le >= e_loc) { route_pos[r] = -1; continue; }
    const int t = r / MOE_TOPK, b = t / MOE_TPB;
    const int p = s_off[le] + blk_cnt[(size_t)b * e_loc + le] + rank[r];
    row_tok[p] = t;
    row_route[p] = r;
    route_pos[r] = p;
  }
}

extern "C" __global__ __launch_bounds__(256) void occamy_moe_combine_topk8_bf16(
    const __bf16 *__restrict__ Y, const float *__restrict__ topk_w, const int *__restrict__ route_pos,
    const __bf16 *__restrict__ sgate, const int *__restrict__ meta, __bf16 *__restrict__ out,
    int H) {
  const int t = blockIdx.x;
  __shared__ int s_pos[MOE_TOPK];
  __shared__ float s_w[MOE_TOPK];
  if (threadIdx.x < MOE_TOPK) {
    s_pos[threadIdx.x] = route_pos[(size_t)t * MOE_TOPK + threadIdx.x];
    s_w[threadIdx.x] = topk_w[(size_t)t * MOE_TOPK + threadIdx.x];
  }
  __syncthreads();
  const int sh0 = meta[2];
  const bool shared = sh0 >= 0 && sgate != nullptr;
  const float sg = shared ? saber_sigmoid((float)sgate[t]) : 0.f;
  union V { uint4 u; __bf16 h[8]; };
  for (int g = threadIdx.x; g < H / 8; g += blockDim.x) {
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    for (int k = 0; k < MOE_TOPK; ++k) {
      const int p = s_pos[k];
      if (p < 0) continue;
      const float w = s_w[k];
      V y;
      y.u = ((const uint4 *)(Y + (size_t)p * H))[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = acc[j] + w * (float)y.h[j];
    }
    if (shared) {
      V y;
      y.u = ((const uint4 *)(Y + (size_t)(sh0 + t) * H))[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = acc[j] + sg * (float)y.h[j];
    }
    V o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)acc[j];
    ((uint4 *)(out + (size_t)t * H))[g] = o.u;
  }
}
