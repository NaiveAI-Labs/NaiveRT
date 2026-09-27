#include "naivert/body/execute_top_p/sampler_mailbox.hpp"
#include "naivert/entry/tp8_argmax_rows_launch.hpp"
#include "naivert/common/tp8_mailbox.cuh"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace naivert_tp8_argmax_rows {
namespace {

using naivert_replica_execute_top_p::SamplerCandidate;
using naivert_replica_execute_top_p::SamplerPeerState;
constexpr int kMailboxStride =
    naivert_replica_execute_top_p::kSamplerMailboxBytes;

__device__ __forceinline__ unsigned ordered_float_key(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned flip =
      static_cast<unsigned>(static_cast<int>(bits) >> 31) |
      0x80000000u;
  return bits ^ flip;
}

__device__ __forceinline__ float ordered_key_float(unsigned key) {
  const unsigned flip =
      (key & 0x80000000u) != 0u ? 0x80000000u : 0xffffffffu;
  return __uint_as_float(key ^ flip);
}

__device__ __forceinline__ unsigned system_load(unsigned* pointer) {
  return naivert::load_acquire_sys(pointer);
}

__device__ __forceinline__ bool wait_generation(
    unsigned* pointer, unsigned generation) {
  constexpr unsigned kMaxPolls = 1u << 20;
  for (unsigned poll = 0; poll < kMaxPolls; ++poll) {
    const unsigned observed = system_load(pointer);
    if (observed == generation) {
      return true;
    }
    if (observed > generation) {
      return false;
    }
    asm volatile("nanosleep.u32 64;" ::: "memory");
  }
  return false;
}

__global__ void naivert_tp8_argmax_rows_kernel(
    const __grid_constant__ Tp8ArgmaxRowsArgs args) {
  __shared__ SamplerCandidate candidates[8];
  __shared__ int valid[8];
  __shared__ unsigned generation;
  const int peer = int(threadIdx.x);
  const int row = int(blockIdx.x);
  auto* local = reinterpret_cast<SamplerPeerState*>(
      const_cast<std::uint8_t*>(
          args.mailbox_peers[args.rank] +
          row * kMailboxStride));
  if (peer == 0) {
    generation = system_load(
        &local->top1_inbox_generation[args.rank]);
  }
  __syncwarp(0xffu);
  const bool ready =
      generation != 0u &&
      wait_generation(
          &local->top1_inbox_generation[peer], generation);
  const unsigned long long packed =
      ready ? local->top1_inbox_packed[peer] : 0ull;
  valid[peer] = packed != 0ull;
  if (packed != 0ull) {
    candidates[peer] = SamplerCandidate{
        ordered_key_float(static_cast<unsigned>(packed >> 32)),
        static_cast<int>(
            0xffffffffu - static_cast<unsigned>(packed))};
  }
  __syncwarp(0xffu);
  if (peer != 0) {
    return;
  }
  int token = -1;
  bool complete = true;
  for (int rank = 0; rank < 8; ++rank) {
    complete = complete && valid[rank] != 0;
  }
  if (complete) {
    SamplerCandidate best{
        __int_as_float(0xff800000), 0x7fffffff};
    unsigned best_key = ordered_float_key(best.score);
    for (int rank = 0; rank < 8; ++rank) {
      const SamplerCandidate candidate = candidates[rank];
      const int global_index =
          candidate.index + rank * args.vocab;
      const unsigned key = ordered_float_key(candidate.score);
      if (
          key > best_key ||
          (key == best_key && global_index < best.index)) {
        best = SamplerCandidate{candidate.score, global_index};
        best_key = key;
      }
    }
    token = best.index;
  }
  args.token_out[row] = token;
  if (args.top_p_scores != nullptr) {
    args.top_p_scores[row] = token >= 0 ? 1.0f : 0.0f;
  }
}

}  // namespace

void launch_tp8_argmax_rows(
    const Tp8ArgmaxRowsArgs& args, cudaStream_t stream) {
  if (
      args.logits == nullptr || args.token_out == nullptr ||
      args.vocab != naivert::geometry::kVocabShard ||
      args.rows < 1 || args.rows > 8 ||
      args.rank < 0 || args.rank >= 8) {
    throw std::invalid_argument(
        "ExecuteTop1 rows received invalid fixed arguments");
  }
  for (const std::uint8_t* pointer : args.mailbox_peers) {
    if (pointer == nullptr) {
      throw std::invalid_argument(
          "ExecuteTop1 rows requires all TP8 mailboxes");
    }
  }
  naivert_tp8_argmax_rows_kernel<<<args.rows, 8, 0, stream>>>(args);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("launch ExecuteTop1 rows: ") +
        cudaGetErrorString(status));
  }
}

}  // namespace naivert_tp8_argmax_rows
