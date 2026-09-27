#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "random.cuh"

// Fixed-shape DSpark T=1, top-p=.95 sampler. No allocation or host decisions
// during replay. Kernel boundaries provide all inter-CTA synchronization.
namespace naivert_dspark_sampling {
constexpr int vocabulary = 152576;
constexpr int threads = 256;
constexpr int items = 8;
constexpr int tile = threads * items;
constexpr int tiles = (vocabulary + tile - 1) / tile;
constexpr int bins = 256;

struct Args {
  float* p;
  float* q;
  int32_t* tokens;
  const int64_t* proposals;
  const int64_t* rng;
  unsigned* watchdog;
  bool anchor;
};

struct Scratch {
  float maxima[15 * tiles];
  double sums[15 * tiles];
  float peak[15], inverse[15];
  double histogram[8 * tiles * bins];
  unsigned prefix[8];
  double remaining[8], above[8], kept[8], kept_inverse[8], mass[8], chunk_prefix[8 * tiles];
  int reject[8], selected_tile[8];
};

struct Maximum {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Minimum {
  __device__ int operator()(int a, int b) const { return min(a, b); }
};
__device__ inline float* distribution(Args a, int row) {
  return row < 8 ? a.p + row * vocabulary : a.q + (row - 8) * vocabulary;
}

__global__ void partial_maximum(Args a, Scratch* s) {
  using R = cub::BlockReduce<float, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  const float* x = distribution(a, row);
  float v = -INFINITY;
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) v = fmaxf(v, x[i]);
  }
  v = R(temp).Reduce(v, Maximum());
  if (threadIdx.x == 0) s->maxima[row * tiles + part] = v;
}

__global__ void maximum(Scratch* s) {
  using R = cub::BlockReduce<float, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.x, t = threadIdx.x;
  float v = t < tiles ? s->maxima[row * tiles + t] : -INFINITY;
  v = R(temp).Reduce(v, Maximum());
  if (t == 0) s->peak[row] = v;
}

__global__ void exponentiate(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  float* x = distribution(a, row);
  const float peak = s->peak[row];
  double sum = 0;
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) { float v = expf(x[i] - peak); x[i] = v; sum += v; }
  }
  sum = R(temp).Sum(sum);
  if (threadIdx.x == 0) s->sums[row * tiles + part] = sum;
}

__global__ void inverse_sum(Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.x, t = threadIdx.x;
  double sum = R(temp).Sum(t < tiles ? s->sums[row * tiles + t] : 0.0);
  if (t == 0) {
    s->inverse[row] = float(1.0 / sum);
    if (row < 8) { s->prefix[row] = 0; s->remaining[row] = 0.95; s->above[row] = 0; }
  }
}

__global__ void normalize(Args a, const Scratch* s) {
  const int row = blockIdx.y, part = blockIdx.x;
  float* x = distribution(a, row);
  const float inv = s->inverse[row];
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) x[i] *= inv;
  }
}

// Weighted radix histograms. Labeled warp groups combine identical digits
// before the shared-memory atomic. Later passes skip warps with no survivors.
template<int shift>
__global__ void radix_histogram(Args a, Scratch* s) {
  namespace cg = cooperative_groups;
  __shared__ double hist[bins];
  const int t = threadIdx.x, row = blockIdx.y, part = blockIdx.x;
  hist[t] = 0;
  __syncthreads();
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const unsigned prefix = s->prefix[row];
  constexpr unsigned mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
  for (int j = t; j < tile; j += threads) {
    const int i = part * tile + j;
    const float v = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    const unsigned bits = __float_as_uint(v);
    const bool active = i < vocabulary && (bits & mask) == prefix;
    if (!__any_sync(0xffffffffu, active && v > 0)) continue;
    const unsigned digit = (bits >> shift) & 255;
    const auto group = cg::labeled_partition(warp, digit);
    const double mass = cg::reduce(group, active ? double(v) : 0.0, cg::plus<double>());
    if (group.thread_rank() == 0 && mass != 0) atomicAdd(&hist[digit], mass);
  }
  __syncthreads();
  s->histogram[(row * tiles + part) * bins + t] = hist[t];
}

template<int shift>
__global__ void select_digit(Args a, Scratch* s) {
  using Scan = cub::BlockScan<double, threads>;
  __shared__ Scan::TempStorage temp;
  const int t = threadIdx.x, row = blockIdx.x, digit = 255 - t;
  double mass = 0;
  const double remaining = s->remaining[row];
  const double previously_above = s->above[row];
  for (int part = 0; part < tiles; ++part)
    mass += s->histogram[(row * tiles + part) * bins + digit];
  double above;
  Scan(temp).ExclusiveSum(mass, above);
  __syncthreads();
  if (above < remaining && above + mass >= remaining) {
    s->prefix[row] |= unsigned(digit) << shift;
    s->remaining[row] = remaining - above;
    s->above[row] = previously_above + above;
    if constexpr (shift == 0) {
      // The retained set is all higher buckets plus the entire boundary bucket.
      // Its FP64 mass is already available; no extra vocabulary reduction.
      const double kept = previously_above + above + mass;
      s->kept[row] = kept;
      s->kept_inverse[row] = 1.0 / kept;
      const int candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
      const float raw = candidate >= 0 ? a.p[row * vocabulary + candidate] : 0;
      const float p = __float_as_uint(raw) >= s->prefix[row] ? float(raw / kept) : 0;
      s->reject[row] = candidate >= 0 &&
        !(naivert_dspark_ops::sampling_uniform(a.rng, 1, row) * a.q[row * vocabulary + candidate] < p);
    }
  }
}

__global__ void normalize_nucleus(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  const double inverse = s->kept_inverse[row];
  const unsigned threshold = s->prefix[row];
  const bool reject = s->reject[row];
  double sum = 0;
  for (int j = threadIdx.x; j < tile; j += threads) {
    const int i = part * tile + j;
    if (i < vocabulary) {
      float& v = a.p[row * vocabulary + i];
      if (__float_as_uint(v) < threshold) v = 0;
      v = float(double(v) * inverse);
      sum += reject ? fmaxf(v - a.q[row * vocabulary + i], 0.0f) : v;
    }
  }
  sum = R(temp).Sum(sum);
  if (threadIdx.x == 0) s->sums[row * tiles + part] = sum;
}

__global__ void choose_tile(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  using Scan = cub::BlockScan<double, threads>;
  __shared__ union { R::TempStorage r; Scan::TempStorage scan; } temp;
  __shared__ double mass;
  const int row = blockIdx.x, t = threadIdx.x;
  double total = t < tiles ? s->sums[row * tiles + t] : 0.0;
  double sum = R(temp.r).Sum(total);
  if (t == 0) { mass = sum; s->mass[row] = sum; s->selected_tile[row] = -1; }
  __syncthreads();
  double prefix;
  Scan(temp.scan).ExclusiveSum(total, prefix);
  const double cut = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row) * mass;
  if (t < tiles) {
    s->chunk_prefix[row * tiles + t] = prefix;
    if (prefix <= cut && prefix + total > cut) s->selected_tile[row] = t;
  }
}

__global__ void choose_token(Args a, const Scratch* s) {
  using Scan = cub::BlockScan<double, threads>;
  using R = cub::BlockReduce<int, threads>;
  __shared__ union { Scan::TempStorage scan; R::TempStorage r; } temp;
  const int row = blockIdx.x, t = threadIdx.x, part = s->selected_tile[row];
  const bool reject = s->reject[row];
  double values[items], total = 0;
  // Contiguous intervals preserve vocabulary order, as in the old sampler.
  for (int j = 0; j < items; ++j) {
    int i = part * tile + t * items + j;
    float v = part >= 0 && i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    if (reject && part >= 0 && i < vocabulary) v = fmaxf(v - a.q[row * vocabulary + i], 0.0f);
    values[j] = v; total += v;
  }
  double prefix;
  Scan(temp.scan).ExclusiveSum(total, prefix);
  __syncthreads();
  prefix += part >= 0 ? s->chunk_prefix[row * tiles + part] : 0.0;
  const double cut = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row) * s->mass[row];
  int winner = vocabulary;
  for (int j = 0; j < items; ++j) {
    prefix += values[j];
    if (values[j] > 0 && prefix > cut) { winner = part * tile + t * items + j; break; }
  }
  winner = R(temp.r).Reduce(winner, Minimum());
  if (t == 0) {
    if (!a.anchor && row < 7 && !reject) winner = int(a.proposals[row]);
    if (winner >= vocabulary || !isfinite(s->kept[row]) || s->kept[row] <= 0 ||
        !isfinite(s->mass[row]) || s->mass[row] <= 0) {
      atomicCAS(a.watchdog, 0u, 0xD6FBu); winner = 0;
    }
    a.tokens[row] = winner;
  }
}

inline void run(Args a, Scratch* s, cudaStream_t stream = nullptr) {
  int rows = a.anchor ? 1 : 8, distributions = a.anchor ? 1 : 15;
  partial_maximum<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  maximum<<<distributions, threads, 0, stream>>>(s);
  exponentiate<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  inverse_sum<<<distributions, threads, 0, stream>>>(s);
  normalize<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  radix_histogram<24><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  select_digit<24><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<16><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  select_digit<16><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<8><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  select_digit<8><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<0><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  select_digit<0><<<rows, threads, 0, stream>>>(a, s);
  normalize_nucleus<<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  choose_tile<<<rows, threads, 0, stream>>>(a, s);
  choose_token<<<rows, threads, 0, stream>>>(a, s);
}
} // namespace naivert_dspark_sampling
