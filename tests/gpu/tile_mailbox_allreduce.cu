// Synthetic TP8 all-reduce over naivert::TileMailbox: one warp per 64-output
// tile, eight ranks, rank-order accumulation from zero. Built per hidden width
// by tests/gpu/run_tile_mailbox_allreduce.py to show the mailbox is not tied to
// one model's geometry.
#include "naivert/common/tp8_mailbox.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace naivert_tile_mailbox_test {

constexpr int kOutputsPerTile = 64;
constexpr int kQwords = kOutputsPerTile / 2;
constexpr int kTiles = naivert::geometry::kHidden / kOutputsPerTile;

struct TestLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task,
                                                                int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};
static_assert(naivert::geometry::kHidden % kOutputsPerTile == 0);
static_assert(kQwords * sizeof(unsigned long long) <= naivert::kMailboxWarpStride);
static_assert(std::size_t(kTiles) * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride,
              "tiles must fit one phase of one lane");

__global__ void tile_allreduce_kernel(const __nv_bfloat16* input,
                                      __nv_bfloat16* output,
                                      const std::uint8_t* const* peers,
                                      int rank, unsigned generation) {
  const int tile = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  __shared__ std::uint32_t staged[kQwords];
  const auto mailbox =
      naivert::TileMailbox<TestLayout>::make(peers, rank, generation, 0);
  const auto* in = reinterpret_cast<const std::uint32_t*>(
      input + std::size_t(tile) * kOutputsPerTile);
  staged[lane] = in[lane];
  mailbox.publish(tile, lane, staged[lane]);
  __syncwarp();
  if (lane < kQwords / 2) {
    const int qword = lane * 2;
    naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::zero();
    naivert::accumulate_rank_order(
        acc, rank, staged[qword], staged[qword + 1], [&](int source_slot) {
          return mailbox.wait_pair<true>(source_slot, tile, qword);
        });
    *reinterpret_cast<unsigned long long*>(
        output + std::size_t(tile) * kOutputsPerTile + qword * 2) =
        acc.packed();
  }
}

void launch_tile_allreduce(const void* input, void* output,
                           const void* const* peers, int rank,
                           unsigned generation, int device,
                           cudaStream_t stream) {
  cudaSetDevice(device);
  tile_allreduce_kernel<<<kTiles, kQwords, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<__nv_bfloat16*>(output),
      reinterpret_cast<const std::uint8_t* const*>(peers), rank, generation);
}

int tile_count() { return kTiles; }
int mailbox_bytes_per_rank() {
  return int((naivert::geometry::kTpRanks - 1) * naivert::kMailboxPeerStride);
}

}  // namespace naivert_tile_mailbox_test
