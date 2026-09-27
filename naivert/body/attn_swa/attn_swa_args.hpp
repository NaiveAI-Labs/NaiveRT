#pragma once
// Naive N0.5 sliding-window attention layer step (one TP8 rank, one query row).
//
// Per rank: 8 query heads x 192 (64 rope dims, theta 1e4), one kv head (192 / 128),
// a 128-token window kept as a ring cache (row = position % 128), and one attention
// sink logit per head that joins the softmax denominator without a value row.
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "naivert/body/attn_common/unproj_tail_args.hpp"

namespace naivert::attn_swa {

struct AttnSwaArgs {
  float value_scale;  // zero means 1; applied after the V projection, never to its weights
  const __nv_bfloat16* residual;   // [rows, 4096] layer input before RMSNorm
  const float* input_norm;         // [4096] RMSNorm gamma (eps 1e-5)
  const __nv_bfloat16* wq;         // [8 * 192, 4096] local query heads
  const __nv_bfloat16* wk;         // [192, 4096] this rank's kv head
  const __nv_bfloat16* wv;         // [128, 4096] value rows (x 0.707 folded)
  const float* rope_freqs;         // [max_seq_len, 64] (cos, sin) per pair, theta 1e4
  const float* sink_bias;          // [8] attention sink logit per local head
  __nv_bfloat16* k_window;         // [kWindowRows, 192] ring: row = position % kWindowRows
  __nv_bfloat16* v_window;         // [kWindowRows, 128]
  __nv_bfloat16* proj_o;           // [rows, 8, 128] attention output for o_proj
  std::uint8_t* workspace;         // AttnSwaScratchLayout::kBytes
  const int* cur_pos;              // position of query row 0
  int max_seq_len;
  naivert::attn_unproj::UnprojTailArgs unproj;   // o_weight == nullptr: no fused o-projection
};

constexpr std::size_t attn_swa_align(std::size_t value) {
  return (value + 255u) & ~std::size_t(255u);
}

#ifndef NAIVERT_ATTN_QUERY_ROWS
#define NAIVERT_ATTN_QUERY_ROWS 1  // decode; the verify build compiles 8 query rows
#endif

struct AttnSwaScratchLayout {
  static constexpr int kQueryRows = NAIVERT_ATTN_QUERY_ROWS;
  static constexpr int kDim = 4096;
  static constexpr int kLocalHeads = 8;
  static constexpr int kHeadDim = 192;
  static constexpr int kRope = 64;
  static constexpr int kVDim = 128;
  static constexpr int kWindow = 128;        // tokens attended, including the query
  // Ring rows per layer (tensor_abi: window + 8). The ring is indexed by kWindowRows,
  // not kWindow: up to eight query rows publish their keys before any of them attends,
  // and those pending rows must not overwrite a key still inside an earlier row's window.
  static constexpr int kWindowRows = 136;
  static_assert(kWindowRows >= kWindow + kQueryRows - 1, "pending query rows must fit beside the window");
  static constexpr int kMaxSeqLen = 1048576;
  static constexpr int kQ = kLocalHeads * kHeadDim;
  static constexpr int kQBase = 0;
  static constexpr int kKBase = kQBase + kQ;
  static constexpr int kVBase = kKBase + kHeadDim;
  static constexpr int kFused = kVBase + kVDim;  // 1856
  // The shared CTA helpers expect these names; no phase timers are recorded.
  static constexpr int kPhaseTimerFloatOffset = 0;
  static constexpr int kPhaseTimerCount = 0;

  static constexpr std::size_t kFusedOffset = 0;
  static constexpr std::size_t kFusedBytes =
      kQueryRows * kFused * sizeof(__nv_bfloat16);
  static constexpr std::size_t kQRotOffset =
      attn_swa_align(kFusedOffset + kFusedBytes);
  static constexpr std::size_t kQRotBytes =
      kQueryRows * kQ * sizeof(__nv_bfloat16);
  static constexpr bool kUseProjectionTma = kQueryRows == 8;
  static constexpr int kProjectionMapCount = 3;
  static constexpr std::size_t kProjectionMapBytes = 128;
  // Prepared once for this workspace's Q/K/V weight pointers, before capture.
  static constexpr std::size_t kProjectionMapsOffset = attn_swa_align(kQRotOffset + kQRotBytes);
  static constexpr std::size_t kBytes = kProjectionMapsOffset +
      (kUseProjectionTma ? kProjectionMapCount * kProjectionMapBytes : 0);
};

}  // namespace naivert::attn_swa
