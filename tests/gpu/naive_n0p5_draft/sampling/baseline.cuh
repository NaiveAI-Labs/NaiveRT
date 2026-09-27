// Frozen original sample_target, production source SHA256
// c590aaf913b059d1d6cca08802c85d963fa5a287c2bbc4b3c09822a9bf32e97f.
// Only unused transport helpers and their include were removed.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "naivert/body/dspark_sampling/random.cuh"


namespace {
constexpr int N = 256, V = 152576, SHARD = V / 8;
struct Maximum { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct Minimum { __device__ int operator()(int a, int b) const { return min(a, b); } };
struct Args {
  const float* logits[8];
  const float* draft_logits[8];
  unsigned long long* flags[8];  // ready, consumed, local invocation counter
  float* full;
  float* draft_full;
  std::int32_t* tokens;
  const std::int64_t* proposals;
  const std::int64_t* rng;
  unsigned* watchdog;
  int rank, rows;
  bool anchor;
};
// One CTA owns a row. Keep normalized probabilities in persistent scratch.
// Radix bisection over positive FP32 probability bits finds the exact nucleus
// boundary, including ties. This deliberately favors a small, testable
// implementation over an approximate top-p threshold.
__global__ void sample_target(const __grid_constant__ Args a) {
  using RF = cub::BlockReduce<float, N>;
  using RD = cub::BlockReduce<double, N>;
  using RI = cub::BlockReduce<int, N>;
  using SD = cub::BlockScan<double, N>;
  __shared__ union {
    RF::TempStorage f;
    RD::TempStorage d;
    RI::TempStorage i;
    SD::TempStorage scan;
  } storage;
  __shared__ float peak, inv_sum;
  __shared__ double mass, kept, cut;
  __shared__ unsigned lo, hi, mid;
  __shared__ int reject, candidate;
  int row = blockIdx.x, tid = threadIdx.x;
  float* p = a.full + row * V;
  float* q = !a.anchor && row < 7 ? a.draft_full + row * V : nullptr;
  float local_max = -INFINITY;
  for (int i = tid; i < V; i += N) local_max = fmaxf(local_max, p[i]);
  float maximum = RF(storage.f).Reduce(local_max, Maximum());
  if (tid == 0) peak = maximum;
  __syncthreads();
  double total = 0;
  for (int i = tid; i < V; i += N) {
    p[i] = expf(p[i] - peak);
    total += p[i];
  }
  double sum = RD(storage.d).Sum(total);
  if (tid == 0) inv_sum = float(1.0 / sum);
  __syncthreads();
  local_max = 0;
  for (int i = tid; i < V; i += N) {
    p[i] *= inv_sum;
    local_max = fmaxf(local_max, p[i]);
  }
  maximum = RF(storage.f).Reduce(local_max, Maximum());
  if (tid == 0) { lo = 0; hi = __float_as_uint(maximum) + 1; }
  __syncthreads();
  while (lo + 1 < hi) {
    if (tid == 0) mid = lo + (hi - lo) / 2;
    __syncthreads();
    float threshold = __uint_as_float(mid);
    total = 0;
    for (int i = tid; i < V; i += N) if (p[i] >= threshold) total += p[i];
    sum = RD(storage.d).Sum(total);
    if (tid == 0) {
      if (sum >= 0.95) lo = mid;
      else hi = mid;
    }
    __syncthreads();
  }
  float threshold = __uint_as_float(lo);
  total = 0;
  for (int i = tid; i < V; i += N) {
    if (p[i] < threshold) p[i] = 0;
    total += p[i];
  }
  sum = RD(storage.d).Sum(total);
  if (tid == 0) kept = sum;
  __syncthreads();
  for (int i = tid; i < V; i += N) p[i] = float(p[i] / kept);
  __syncthreads();
  if (q) {
    local_max = -INFINITY;
    for (int i = tid; i < V; i += N) local_max = fmaxf(local_max, q[i]);
    maximum = RF(storage.f).Reduce(local_max, Maximum());
    if (tid == 0) peak = maximum;
    __syncthreads();
    total = 0;
    for (int i = tid; i < V; i += N) {
      q[i] = expf(q[i] - peak);
      total += q[i];
    }
    sum = RD(storage.d).Sum(total);
    if (tid == 0) inv_sum = float(1.0 / sum);
    __syncthreads();
    for (int i = tid; i < V; i += N) q[i] *= inv_sum;
    __syncthreads();
  }
  if (tid == 0) {
    candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
    reject = candidate >= 0 &&
        !(naivert_dspark_ops::sampling_uniform(a.rng, 1, row) * q[candidate] < p[candidate]);
  }
  __syncthreads();
  // CDF in token order. Contiguous per-thread intervals + prefix scan retain
  // that order even when the vocabulary does not divide the block size.
  constexpr int chunk = (V + N - 1) / N;
  int begin = tid * chunk, end = min(begin + chunk, V);
  total = 0;
  for (int i = begin; i < end; ++i)
    total += reject ? fmaxf(p[i] - q[i], 0.0f) : p[i];
  sum = RD(storage.d).Sum(total);
  if (tid == 0) {
    mass = sum;
    cut = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row) * mass;
  }
  __syncthreads();
  double prefix;
  SD(storage.scan).ExclusiveSum(total, prefix);
  __syncthreads();
  int winner = V;
  if (prefix <= cut && prefix + total > cut) {
    for (int i = begin; i < end; ++i) {
      double w = reject ? fmaxf(p[i] - q[i], 0.0f) : p[i];
      prefix += w;
      if (w > 0 && prefix > cut) { winner = i; break; }
    }
  }
  int token = RI(storage.i).Reduce(winner, Minimum());
  if (tid == 0) {
    if (candidate >= 0 && !reject) token = candidate;
    if (token == V || !isfinite(kept) || kept <= 0 || !isfinite(mass) || mass <= 0) {
      atomicCAS(a.watchdog, 0u, 0xD6FBu);
      token = 0;
    }
    a.tokens[row] = token;
  }
  // p and q remain available for independent tests.
}

}
