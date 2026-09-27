#pragma once
#include "naivert/common/resident_grid_barrier.cuh"
namespace naivert_dspark_ops {
constexpr int kWindow = 8;
constexpr int kHidden = 4096;
constexpr int kThreads = 384;
__device__ inline void grid_barrier(unsigned* state) {
  naivert::resident_grid_barrier(state);
}
}
