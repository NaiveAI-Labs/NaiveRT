#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

// DownAllreduce entry args: the common entry header + dense down-proj body
// payload + 8-peer mailbox payload.
// Interface reference:
//   dev_redesign_abi/new_abi/kernel_interface.py::down_allreduce(layer)
namespace naivert_replica_down_allreduce {

struct DownAllreduceArgs {
  // --- common entry header ---
  int total_tiles;
  int active_ctas;
  std::uint8_t* workspace_base;
  const std::uint8_t* step_block_base;

  // --- body payload: dense FFN down projection + residual ---
  const __nv_bfloat16* up_gate;       // [9, 256] local TP shard
  const __nv_fp8_e4m3* w_down;        // [9, 7168, 256] native swizzle
  const __nv_bfloat16* w_down_scale;  // [9, 1024, 2] native scale layout
  const __nv_bfloat16* residual;
  __nv_bfloat16* exp_out;
  __nv_bfloat16* tp_result;

  // --- shape config ---
  int experts;
  int hidden;
  int inter;  // local TP inter (geometry::kDenseInter)
  int outputs_per_tile;

  // --- all-reduce payload ---
  int rank;
  const std::uint8_t* mailbox_peers[8];
  // Optional production replay source.  The setup graph updates this fixed
  // device address from CUR_POS before every body replay.  The kernel combines
  // it with generation_salt so different allreduce executors cannot accept
  // one another's tags at the same replay position.  Standalone/new-ABI tests
  // leave it null and keep using step_block_base + 8.
  const std::int32_t* replay_generation;
  std::uint32_t generation_salt;
};

}  // namespace naivert_replica_down_allreduce
