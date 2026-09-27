#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace naivert_replica_unproj_o_allreduce {

struct UnprojOAllreduceArgs {
  // Fixed common entry header.
  int total_tiles;
  int active_ctas;
  std::uint8_t* workspace_base;
  // Optional MTP replay counter, encoded as a byte pointer to preserve the
  // frozen 152-byte Seq1 ABI. Seq1 keeps this null.
  const std::uint8_t* step_block_base;

  // Native descriptor words 0..4.
  const __nv_bfloat16* proj_o;       // [2048]
  const __nv_fp8_e4m3* weight;       // [112, 4, 32768], native QMMA swizzle
  const __nv_bfloat16* scales;       // [56, 16], 128x128 block scales
  const __nv_bfloat16* residual;     // [7168]
  __nv_bfloat16* output;             // [7168]

  int hidden;
  int projection_k;
  int outputs_per_tile;

  // Native GlbArgs words 4..11 are destination-rank mailbox bases.
  int rank;
  const std::uint8_t* mailbox_peers[8];

  // Native descriptor word 5.  Production uses 2 * layer + 2, and the low
  // bit selects the 0x8000-byte double-buffer window.
  std::uint32_t mailbox_tag;
};

}  // namespace naivert_replica_unproj_o_allreduce
