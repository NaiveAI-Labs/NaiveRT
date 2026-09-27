#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

// RMSNormUpGate entry args: the common entry header + dense MLP body payload.
// Interface reference:
//   dev_redesign_abi/new_abi/kernel_interface.py::rmsnorm_upgate(layer)
namespace naivert_replica_rmsnorm_up_gate {

struct RMSNormUpGateArgs {
  // --- common entry header ---
  int total_tiles;
  int active_ctas;
  std::uint8_t* workspace_base;
  const std::uint8_t* step_block_base;

  // --- body payload: post-attn RMSNorm + dense up/gate + SiLU ---
  const __nv_bfloat16* unproj_o;
  const float* post_norm;
  const __nv_fp8_e4m3* w_gate;
  const __nv_bfloat16* w_gate_scale;
  const __nv_fp8_e4m3* w_up;
  const __nv_bfloat16* w_up_scale;
  __nv_bfloat16* up_gate;

  // --- shape config ---
  int dim;
  int inter_dim;
};

}  // namespace naivert_replica_rmsnorm_up_gate
