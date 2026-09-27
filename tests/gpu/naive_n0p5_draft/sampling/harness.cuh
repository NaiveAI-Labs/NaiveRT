#pragma once
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "baseline.cuh"
#include "naivert/body/dspark_sampling/parallel.cuh"
namespace fast = naivert_dspark_sampling;
#define CK(expr)                                                   \
  do {                                                             \
    auto err = (expr);                                             \
    if (err != cudaSuccess) {                                      \
      fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(err)); \
      exit(1);                                                     \
    }                                                              \
  } while (0)
#define REQUIRE(expr)                                           \
  do {                                                          \
    if (!(expr)) {                                              \
      fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #expr); \
      exit(2);                                                  \
    }                                                           \
  } while (0)
template <class T>
std::vector<T> read(const std::string& file, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(file, std::ios::binary);
  f.read((char*)v.data(), n * sizeof(T));
  REQUIRE(f && f.peek() == EOF);
  return v;
}
template <class T>
void alloc(T*& p, size_t n) {
  CK(cudaMalloc(&p, n * sizeof(T)));
}
template <class T>
std::vector<T> download(T* p, size_t n) {
  std::vector<T> v(n);
  CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
struct Buffers {
  Args a{};
  fast::Scratch* scratch;
  int64_t *rng, *proposals;
  Buffers() {
    alloc(a.full, 8 * V);
    alloc(a.draft_full, 7 * V);
    alloc(a.tokens, 8);
    alloc(a.watchdog, 1);
    alloc(rng, 2);
    alloc(proposals, 7);
    alloc(scratch, 1);
    a.rng = rng;
    a.proposals = proposals;
  }
  void input(const std::vector<float>& p, const std::vector<float>& q,
             const std::vector<int64_t>& ids, bool anchor, int seed,
             int counter) {
    a.anchor = anchor;
    a.rows = anchor ? 1 : 8;
    int64_t state[2] = {seed, counter};
    CK(cudaMemcpy(a.full, p.data(), p.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(a.draft_full, q.data(), q.size() * sizeof(float),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rng, state, sizeof(state), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(proposals, ids.data(), 7 * sizeof(int64_t),
                  cudaMemcpyHostToDevice));
    CK(cudaMemset(a.watchdog, 0, 4));
  }
  void launch(bool candidate, cudaStream_t stream = nullptr) {
    if (candidate)
      fast::run({a.full, a.draft_full, a.tokens, a.proposals, a.rng, a.watchdog,
                 a.anchor},
                scratch, stream);
    else
      sample_target<<<a.rows, 256, 0, stream>>>(a);
    CK(cudaGetLastError());
  }
};
// Independent CPU math oracle: full descending sort chooses the boundary.
// It uses FP32 exp/normalization with FP64 sums, includes boundary ties, and
// checks candidate p/q decisions and the positive residual distribution.
std::vector<float> cpu_probability(const float* input, bool nucleus) {
  std::vector<float> p(V);
  float peak = *std::max_element(input, input + V);
  double sum = 0;
  for (int i = 0; i < V; ++i) {
    p[i] = expf(input[i] - peak);
    sum += p[i];
  }
  float inv = float(1.0 / sum);
  for (float& v : p) v *= inv;
  if (nucleus) {
    auto sorted = p;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    double acc = 0;
    float threshold = 0;
    for (float v : sorted) {
      acc += v;
      if (acc >= 0.95) {
        threshold = v;
        break;
      }
    }
    sum = 0;
    for (float& v : p) {
      if (v < threshold) v = 0;
      sum += v;
    }
    for (float& v : p) v = float(v / sum);
  }
  return p;
}
__global__ void uniforms(const int64_t* rng, float* out, bool anchor) {
  int r = threadIdx.x;
  if (r < 8) {
    out[r] = naivert_dspark_ops::sampling_uniform(rng, 1, r);
    out[8 + r] = naivert_dspark_ops::sampling_uniform(rng, anchor ? 3 : 2, r);
  }
}
