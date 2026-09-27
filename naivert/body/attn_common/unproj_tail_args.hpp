#pragma once
// Arguments of the o-projection + TP8 all-reduce tail fused into the Naive N attention kernels
//. Host-visible (no device code): appended to AttnSwaArgs / AttnDsaArgs.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace naivert::attn_unproj {

struct UnprojTailArgs {
  const __nv_bfloat16* o_weight_bf16;  // optional natural [hidden, projection_k] checkpoint matrix
  const __nv_fp8_e4m3* o_weight;      // pack_unproj pages [hidden/64 tiles, 4 slots, 64 x 256]; nullptr = no tail
  const __nv_bfloat16* o_scales;      // [hidden/128, 8] 128 x 128 block scales
  __nv_bfloat16* hidden_out;          // [rows, hidden] residual + all-reduced projection
  const std::uint8_t* mailbox_peers[8];   // every rank's mailbox base (row slabs of 0x70000)
  const std::uint64_t* replay_epoch;  // generation = (epoch << 8) | tag; nullptr = eager (epoch 0)
  int rank;
  std::uint32_t mailbox_tag;
};

}  // namespace naivert::attn_unproj
