// The accepted-row UnprojO variant selects one of four candidate rows on the
// device from the verify/commit accepted count.  It has entry logic of its
// own, so it stays hand-written next to the generated unproj_o_allreduce
// entry (naivert/kernels/ops/unproj_o_allreduce.py).
#include "naivert/entry/unproj_o_allreduce_accepted_row.hpp"
#include "naivert/body/unproj_o_allreduce/unproj_o_allreduce_body.cuh"
#include <cstddef>
#include <cstring>

namespace naivert_replica_unproj_o_allreduce {

__global__ __launch_bounds__(kBlockThreads, 1)
void naivert_unproj_allreduce_accepted_row_kernel(
    const __grid_constant__ UnprojOAllreduceArgs args,
    const std::uint32_t* accepted_count) {
  UnprojOAllreduceArgs row_args = args;
  const std::uint32_t accepted =
      accepted_count == nullptr ? 1u : *accepted_count;
  const std::uint32_t row =
      accepted >= 1u && accepted <= 4u ? accepted - 1u : 0u;
  row_args.proj_o += row * args.projection_k;
  row_args.residual += row * args.hidden;
  extern __shared__ __align__(128) unsigned char shared[];
  UnprojOAllreduceBody::run<false>(row_args, shared);
}

cudaError_t launch_unproj_o_allreduce_selected_rank(
    const void* proj_o_rows, const void* weight, const void* scales,
    const void* residual_rows, const std::uint32_t* accepted_count,
    void* output, const void* const mailbox_peers[8], int rank,
    std::uint32_t mailbox_tag, const std::uint64_t* replay_iteration,
    int device, cudaStream_t stream) {
  if (accepted_count == nullptr) {
    return cudaErrorInvalidValue;
  }
  UnprojOAllreduceArgs args;
  std::memset(&args, 0, sizeof(args));
  args.total_tiles = kHidden / kOutputsPerTile;
  args.active_ctas = args.total_tiles;
  args.workspace_base = nullptr;
  args.step_block_base =
      reinterpret_cast<const std::uint8_t*>(replay_iteration);
  args.proj_o = reinterpret_cast<const __nv_bfloat16*>(proj_o_rows);
  args.weight = reinterpret_cast<const __nv_fp8_e4m3*>(weight);
  args.scales = reinterpret_cast<const __nv_bfloat16*>(scales);
  args.residual = reinterpret_cast<const __nv_bfloat16*>(residual_rows);
  args.output = reinterpret_cast<__nv_bfloat16*>(output);
  args.hidden = kHidden;
  args.projection_k = kProjectionK;
  args.outputs_per_tile = kOutputsPerTile;
  args.rank = rank;
  args.mailbox_tag = mailbox_tag;
  for (int peer = 0; peer < 8; ++peer) {
    args.mailbox_peers[peer] =
        mailbox_peers == nullptr
            ? nullptr
            : reinterpret_cast<const std::uint8_t*>(mailbox_peers[peer]);
  }
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(
      naivert_unproj_allreduce_accepted_row_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaFuncSetAttribute(
      naivert_unproj_allreduce_accepted_row_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (err != cudaSuccess) {
    return err;
  }
  naivert_unproj_allreduce_accepted_row_kernel<<<
      naivert::geometry::kSmCount, kBlockThreads, kSharedBytes, stream>>>(args, accepted_count);
  return cudaGetLastError();
}

}  // namespace naivert_replica_unproj_o_allreduce
