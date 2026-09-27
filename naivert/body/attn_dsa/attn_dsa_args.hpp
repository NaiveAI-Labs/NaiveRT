#pragma once
// Naive N0.5 full-attention layer step on one TP8 rank: arguments and
// workspace layout for naivert/body/attn_dsa/attn_dsa_body.cuh.
//
// Per rank: 8 query heads (64 / 8) against this rank's single kv head
// (4 kv heads / 8 ranks -> the head is shared by two ranks), head 192 =
// 64 rotary + 128 non-rotary dims, value 128; the sparse-attention
// indexer (16 or 64 index heads x 128, one key head, top-k 2048) is replicated on
// every rank so the selection needs no cross-rank exchange.
//
// Weight rows are bf16 checkpoint rows. The 64 rotary dims of every q / k /
// index-q / index-k head are row-permuted at conversion time from HF's
// rotate_half order (i, i + 32) to interleaved pairs (2i, 2i + 1), so the
// kernel rotates adjacent pairs with one (cos, sin) table entry per pair.
// The 0.707 value scale is folded into wv.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

#include "naivert/body/attn_common/unproj_tail_args.hpp"

namespace naivert::attn_dsa {

struct AttnDsaArgs {
  const __nv_bfloat16* wiq_bf16;  // natural BF16 index-query weights; nullptr uses legacy FP8
  float value_scale;  // zero means 1; applied after the V projection, never to its weights
  const __nv_bfloat16* residual;   // [rows, 4096] layer input before RMSNorm
  const float* input_norm;         // [4096] RMSNorm gamma (eps 1e-5)
  const __nv_bfloat16* wq;         // [8 * 192, 4096] local query heads
  const __nv_bfloat16* wk;         // [192, 4096] this rank's kv head
  const __nv_bfloat16* wv;         // [128, 4096] value rows (x 0.707 folded)
  const __nv_fp8_e4m3* wiq;        // [index_heads * 128, 4096] replicated index queries, e4m3
  const float* wiq_scale;          // [index_heads, 32] block-128 scales of wiq
  const __nv_bfloat16* wik;        // [128, 4096] index key
  const float* ik_norm_weight;     // [128] LayerNorm on the index key
  const float* ik_norm_bias;       // [128]
  const __nv_bfloat16* wis;        // [index_heads, 4096] indexer weights_proj
  const float* rope_freqs;         // [max_seq_len, 64] (cos, sin) per pair
  __nv_bfloat16* k;                // [max_seq_len, 192] key cache
  __nv_bfloat16* v;                // [max_seq_len, 128] value cache
  __nv_fp8_e4m3* ki;               // [max_seq_len, 128] index-key cache, e4m3
  float* ki_scale;                 // [max_seq_len] per-position dequantization scale (row amax / 448)
  __nv_bfloat16* proj_o;           // [rows, 8, 128] attention output for o_proj
  std::uint8_t* workspace;         // AttnDsaScratchLayout::kBytes
  const int* cur_pos;              // position of query row 0
  int max_seq_len;
  float attention_scale_multiplier;  // 0 = none
  naivert::attn_unproj::UnprojTailArgs unproj;   // o_weight == nullptr: no fused o-projection
};

constexpr std::size_t attn_dsa_align(std::size_t value) {
  return (value + 255u) & ~std::size_t(255u);
}

#ifndef NAIVERT_ATTN_QUERY_ROWS
#define NAIVERT_ATTN_QUERY_ROWS 1  // decode; the verify build compiles 8 query rows
#endif
#ifndef NAIVERT_DSA_INDEX_HEADS
#define NAIVERT_DSA_INDEX_HEADS 64  // 64 in the 64-head layout; 16 in the FP8 checkpoint (replicated per TP rank)
#endif

struct AttnDsaScratchLayout {
  static constexpr int kQueryRows = NAIVERT_ATTN_QUERY_ROWS;
  static constexpr int kDim = 4096;
  static constexpr int kLocalHeads = 8;
  static constexpr int kPaddedHeads = 16;  // wmma N tile; rows 8..15 stay zero
  static constexpr int kHeadDim = 192;
  static constexpr int kRope = 64;
  static constexpr int kVDim = 128;
  static constexpr int kIndexHeads = NAIVERT_DSA_INDEX_HEADS;
  static_assert(kIndexHeads == 16 || kIndexHeads == 64, "supported DSA index heads: 16 or 64");
  static_assert(kQueryRows == 1 || kQueryRows == 8, "supported DSA query rows: 1 or 8");
  static constexpr int kIndexDim = 128;
  static constexpr int kIndexTopK = 2048;
  static constexpr int kIqRows = kIndexHeads * kIndexDim;
  static constexpr int kMaxSeqLen = 1048576;

  // Fused projection row table: [q | k | v | ik | iq]; weights_proj rows
  // follow in the same pass but land in the float index_weights buffer.
  static constexpr int kQ = kLocalHeads * kHeadDim;
  static constexpr int kIq = kIndexHeads * kIndexDim;
  static constexpr int kQBase = 0;
  static constexpr int kKBase = kQBase + kQ;
  static constexpr int kVBase = kKBase + kHeadDim;
  static constexpr int kIkBase = kVBase + kVDim;
  static constexpr int kIqBase = kIkBase + kIndexDim;
  static constexpr int kFused = kIqBase + kIq;  // 4032 (16 heads) or 10176 (64 heads)

  // Attention scratch per query row (floats): scores, bf16 probabilities,
  // split outputs, split stats; then phase timers and ready flags in the
  // tail. Same stride as MLA-v28 so the shared top-k stage keeps its
  // assumptions.
  static constexpr int kAttentionTempStride = 312576;
  static constexpr int kPhaseTimerFloatOffset = 312320;
  static constexpr int kPhaseTimerCount = 17;

  static constexpr std::size_t kFusedOffset = 0;
  static constexpr std::size_t kFusedBytes =
      kQueryRows * kFused * sizeof(__nv_bfloat16);
  static constexpr std::size_t kQRotOffset =
      attn_dsa_align(kFusedOffset + kFusedBytes);
  static constexpr std::size_t kQRotBytes =
      kQueryRows * kPaddedHeads * kHeadDim * sizeof(__nv_bfloat16);
  static constexpr std::size_t kIqPackedOffset =
      attn_dsa_align(kQRotOffset + kQRotBytes);
  static constexpr std::size_t kIqPackedBytes =
      kQueryRows * kIq * sizeof(__nv_bfloat16);
  static constexpr std::size_t kIndexWeightsOffset =
      attn_dsa_align(kIqPackedOffset + kIqPackedBytes);
  static constexpr std::size_t kIndexWeightsBytes =
      kQueryRows * kIndexHeads * sizeof(float);
  // The index weights times the (row, head) query scales, read by the e4m3 sweep.
  static constexpr std::size_t kIndexWeightsScaledOffset =
      attn_dsa_align(kIndexWeightsOffset + kIndexWeightsBytes);
  static constexpr std::size_t kIndexWeightsScaledBytes = kIndexWeightsBytes;
  static constexpr std::size_t kIndexLogitsOffset =
      attn_dsa_align(kIndexWeightsScaledOffset + kIndexWeightsScaledBytes);
  static constexpr std::size_t kIndexLogitsBytes =
      kQueryRows * kMaxSeqLen * sizeof(float);
  static constexpr std::size_t kIndexSelectsOffset =
      attn_dsa_align(kIndexLogitsOffset + kIndexLogitsBytes);
  static constexpr std::size_t kIndexSelectsBytes =
      kQueryRows * kIndexTopK * sizeof(int);
  static constexpr std::size_t kRadixControlOffset =
      attn_dsa_align(kIndexSelectsOffset + kIndexSelectsBytes);
  static constexpr std::size_t kRadixControlBytes =
      (3u * kQueryRows + 10u + 1u) * sizeof(unsigned);
  static constexpr std::size_t kAttentionTempOffset =
      attn_dsa_align(kRadixControlOffset + kRadixControlBytes);
  static constexpr std::size_t kAttentionTempBytes =
      kQueryRows * kAttentionTempStride * sizeof(float);
  // Grid-parallel top-k selection: per query four 256-bin digit histograms and
  // four 4096-entry chunk count/prefix arrays, two list counters, the above list and
  // the 8192-entry candidate token/key lists (topk_grid_select.cuh).
  static constexpr std::size_t kTopKGridOffset =
      attn_dsa_align(kAttentionTempOffset + kAttentionTempBytes);
  static constexpr std::size_t kTopKGridBytes =
      std::size_t(kQueryRows) *
      (4u * 256u + 4u * 4096u + 2u + std::size_t(kIndexTopK) + 2u * 8192u) * sizeof(unsigned);
  // Four descriptors for natural BF16 Q/K/V/index-key matrices. Prepared once
  // before graph capture; replacing weights or workspace requires preparation.
  static constexpr bool kUseProjectionTma = kQueryRows == 8 && kIndexHeads == 16;
  static constexpr int kProjectionMapCount = 4;
  static constexpr std::size_t kProjectionMapBytes = 128;
  static constexpr std::size_t kProjectionMapsOffset =
      attn_dsa_align(kTopKGridOffset + kTopKGridBytes);
  static constexpr std::size_t kBytes = kProjectionMapsOffset +
      (kUseProjectionTma ? kProjectionMapCount * kProjectionMapBytes : 0);

};

}  // namespace naivert::attn_dsa
