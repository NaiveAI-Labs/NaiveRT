#pragma once
// The o-projection and its TP8 all-reduce as the last phase of the Naive N attention kernels
//. Same math, weight layout, scales, mailbox layout, tags and summation recipes as the
// standalone unproj_o_allreduce kernel (one row: the eight partials in rank order from zero;
// several rows: local first, then the peer lanes), so the fused output
// is bit-identical to attention followed by unproj, and the kernel boundary between them is
// gone. One CTA per 64-output tile (64 tiles at hidden 4096); the weights are read from global
// memory in the pack_unproj swizzle, the query rows sit in the MMA's N.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

#include "naivert/body/attn_common/unproj_tail_args.hpp"
#include "naivert/body/attn_common/bf16_linear_allreduce.cuh"
#include "naivert/common/tp8_mailbox.cuh"

namespace naivert::attn_unproj {

// A nonzero kKnownThreads must match the caller's block size.
template <int kRows, int kHidden, int kProjectionK, bool kPackedFinite = false,
          int kKnownThreads = 0>
struct UnprojTail {
  static constexpr int kInputSlots = 4;
  static constexpr int kInter = kProjectionK / kInputSlots;       // 256
  static constexpr int kInterChunks = kInter / 32;                 // 8
  static constexpr int kWarpK = kInter / 8;                        // 32: eight warps split a slot
  static constexpr int kOutputsPerTile = 64;
  static constexpr int kTiles = kHidden / kOutputsPerTile;
  static constexpr int kScaleColumns = kProjectionK / 128;         // 8
  static constexpr int kBlocksPerSlot = kInter / 128;              // 2
  static constexpr int kWeightPaneBytes = kOutputsPerTile * kInter;  // 16 KiB fp8 per (tile, slot)
  static constexpr int kQwordsPerTile = kOutputsPerTile / 2;       // 32 tagged qwords of bf16x2
  static constexpr int kComputeWarps = 8;
  // Shared memory layout (aliases the attention kernel's x_norm buffer, free by then).
  static constexpr int kProjOffset = 0;                            // [kRows][kProjectionK] bf16
  static constexpr int kProjBytes = kRows * kProjectionK * int(sizeof(__nv_bfloat16));
  static constexpr int kResidualOffset = kProjOffset + kProjBytes; // [kRows][64] bf16
  static constexpr int kResidualBytes = kRows * kOutputsPerTile * int(sizeof(__nv_bfloat16));
  static constexpr int kPartialOffset = kResidualOffset + kResidualBytes;   // [8 warps][kRows][64] f32
  static constexpr int kPartialBytes = kComputeWarps * kRows * kOutputsPerTile * int(sizeof(float));
  static constexpr int kStagedOffset = kPartialOffset + kPartialBytes;      // [kRows][32] u32 payload
  static constexpr int kStagedBytes = kRows * kQwordsPerTile * int(sizeof(std::uint32_t));
  static constexpr int kSharedBytes = (kStagedOffset + kStagedBytes >
      Bf16LinearAllreduce<kRows, kHidden, kProjectionK>::kSharedBytes) ? kStagedOffset + kStagedBytes
      : Bf16LinearAllreduce<kRows, kHidden, kProjectionK>::kSharedBytes;
  static_assert(Bf16LinearAllreduce<kRows, kHidden, kProjectionK>::kSharedBytes <= kSharedBytes);
  static_assert(kRows >= 1 && kRows <= 8, "query rows live in the MMA's N = 8");
  static_assert(kHidden % kOutputsPerTile == 0 && kProjectionK % (kInputSlots * 32) == 0, "tiling");
  static_assert(kWarpK == 32, "one K=32 slice per compute warp per slot");

  struct MailboxLayout {
    __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
      return std::size_t(task) * naivert::kMailboxWarpStride +
             std::size_t(qword) * sizeof(unsigned long long);
    }
  };

  // pack_unproj's QMMA swizzle of one (tile, slot) pane (unproj_o_allreduce_body).
  __host__ __device__ static int pane_storage_offset(int local_row, int k) {
    const int b = local_row >> 4;
    const int d = local_row & 15;
    const int c = k >> 5;
    const int e = k & 31;
    const int s0 = d >> 3;
    const int s1 = d & 7;
    const int s2 = e >> 4;
    const int rem = e & 15;
    const int s3 = rem >> 2;
    const int s4 = rem & 3;
    const int y_linear = ((((s2 * 2 + s0) * 8 + s1) * 4 + s3) * 4 + s4);
    const int y0 = y_linear >> 7;
    const int y_rem = y_linear & 127;
    const int y1 = y_rem >> 2;
    const int y2 = y_rem & 3;
    const int source = (y1 * 4 + y0) * 4 + y2;
    const int x0 = source >> 4;
    const int source_rem = source & 15;
    const int x1 = source_rem >> 2;
    const int x2 = source_rem & 3;
    return (((((b * kInterChunks + c) * 32 + x0) * 4 + x1) * 4) + x2);
  }

  __device__ static __forceinline__ std::uint32_t fp8_pair_as_bf16x2(std::uint16_t raw) {
    std::uint32_t out;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(raw));
    return out;
  }

  __device__ static __forceinline__ void mma_16816_f32_bf16(
      std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
      std::uint32_t b1, float& c0, float& c1, float& c2, float& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  __device__ static unsigned generation(const UnprojTailArgs& u) {
    const unsigned epoch = u.replay_epoch == nullptr ? 0u : static_cast<unsigned>(*u.replay_epoch);
    return naivert::mailbox_generation(epoch, u.mailbox_tag);
  }

  // The four W_o panes of a tile (64 KiB, the pack_unproj swizzle verbatim) staged into shared
  // memory by cp.async ahead of the tail; issue by every thread, commit, then wait + __syncthreads.
  __device__ static void prefetch_panes(const UnprojTailArgs& u, int tile, unsigned char* panes) {
    if (u.o_weight_bf16 != nullptr) return;
    const unsigned char* source = reinterpret_cast<const unsigned char*>(u.o_weight) +
                                  std::size_t(tile) * kInputSlots * kWeightPaneBytes;
    constexpr int kChunks = kInputSlots * kWeightPaneBytes / 16;
    for (int i = int(threadIdx.x); i < kChunks; i += int(blockDim.x)) {
      unsigned long long src_global;
      asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_global) : "l"(source + i * 16));
      const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(panes + i * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src_global) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  __device__ static __forceinline__ void prefetch_wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

  // Every thread of every CTA calls this after the attention output is complete (grid barrier
  // in the caller). CTA `tile` < kTiles owns output columns [tile * 64, tile * 64 + 64).
  __device__ static void run(const UnprojTailArgs& u, const __nv_bfloat16* proj_o,
                             const __nv_bfloat16* residual, unsigned char* shared) {
    run(u, proj_o, residual, shared, int(blockIdx.x), nullptr);
  }
  // tile: the output tile this CTA owns (>= kTiles: nothing to do); panes: the tile's four W_o panes
  // already in shared memory, or nullptr to read them from global memory.
  __device__ static void run(const UnprojTailArgs& u, const __nv_bfloat16* proj_o,
                             const __nv_bfloat16* residual, unsigned char* shared, int tile,
                             const unsigned char* panes) {
    if (u.o_weight_bf16 != nullptr) {
      Bf16LinearAllreduce<kRows, kHidden, kProjectionK, (kRows == 8 && kProjectionK == 1024 ? 256 : 0)>::run(
          u, u.o_weight_bf16, proj_o, residual, shared, tile);
      return;
    }
    if (tile < 0 || tile >= kTiles) {
      return;
    }
    const int tid = int(threadIdx.x);
    const int threads = kKnownThreads == 0 ? int(blockDim.x) : kKnownThreads;
    auto* shared_proj = reinterpret_cast<__nv_bfloat16*>(shared + kProjOffset);
    auto* shared_residual = reinterpret_cast<__nv_bfloat16*>(shared + kResidualOffset);
    float* partial = reinterpret_cast<float*>(shared + kPartialOffset);
    auto* staged = reinterpret_cast<std::uint32_t*>(shared + kStagedOffset);

    // 1. Stage the attention output rows (non-finite -> 0, as unproj does) and the residual tile.
    for (int i = tid; i < kRows * kProjectionK / 8; i += threads) {
      uint4 v = reinterpret_cast<const uint4*>(proj_o)[i];
      if constexpr (kPackedFinite) {
        // Each halfword has an all-ones exponent exactly for Inf or NaN.
        // Clear those halfwords while preserving every finite BF16 bit.
        unsigned* words = reinterpret_cast<unsigned*>(&v);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const unsigned raw = words[pair];
          const unsigned nonfinite = __vcmpeq2(raw & 0x7f807f80u, 0x7f807f80u);
          words[pair] = raw & ~nonfinite;
        }
      } else {
        auto* h = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (!isfinite(__bfloat162float(h[e]))) {
            h[e] = __float2bfloat16(0.0f);
          }
        }
      }
      reinterpret_cast<uint4*>(shared_proj)[i] = v;
    }
    for (int i = tid; i < kRows * kOutputsPerTile / 8; i += threads) {
      const int row = i / (kOutputsPerTile / 8);
      const int piece = i - row * (kOutputsPerTile / 8);
      reinterpret_cast<uint4*>(shared_residual)[i] = reinterpret_cast<const uint4*>(
          residual + std::size_t(row) * kHidden + tile * kOutputsPerTile)[piece];
    }
    __syncthreads();

    // 2. Eight warps: K slices of each input slot; output rows in M, query rows in N.
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;   // output row within the 16-row m tile, and query column group
    const int tid4 = lane & 3;
    if (warp < kComputeWarps) {
      float acc[4][4] = {};
#pragma unroll
      for (int slot = 0; slot < kInputSlots; ++slot) {
        const unsigned char* pane = panes != nullptr
            ? panes + slot * kWeightPaneBytes
            : reinterpret_cast<const unsigned char*>(
                  u.o_weight + (std::size_t(tile) * kInputSlots + slot) * kWeightPaneBytes);
        const int lane_k4 = warp * kWarpK + tid4 * 4;
        // B: query row `group` (zero beyond kRows), K permuted like the standalone kernel.
        std::uint32_t b_low01 = 0u, b_low23 = 0u, b_high01 = 0u, b_high23 = 0u;
        if (group < kRows) {
          const __nv_bfloat16* vec = shared_proj + group * kProjectionK + slot * kInter;
          b_low01 = *reinterpret_cast<const std::uint32_t*>(vec + lane_k4);
          b_low23 = *reinterpret_cast<const std::uint32_t*>(vec + lane_k4 + 16);
          b_high01 = *reinterpret_cast<const std::uint32_t*>(vec + lane_k4 + 2);
          b_high23 = *reinterpret_cast<const std::uint32_t*>(vec + lane_k4 + 18);
        }
        float d[4][4] = {};
#pragma unroll
        for (int m_tile = 0; m_tile < 4; ++m_tile) {
          const int row = m_tile * 16 + group;
          const uint4 packed = panes != nullptr
              ? *reinterpret_cast<const uint4*>(pane + pane_storage_offset(row, lane_k4))
              : __ldg(reinterpret_cast<const uint4*>(pane + pane_storage_offset(row, lane_k4)));
          mma_16816_f32_bf16(fp8_pair_as_bf16x2(std::uint16_t(packed.x)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.y)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.z)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.w)), b_low01, b_low23,
                             d[m_tile][0], d[m_tile][1], d[m_tile][2], d[m_tile][3]);
          mma_16816_f32_bf16(fp8_pair_as_bf16x2(std::uint16_t(packed.x >> 16)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.y >> 16)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.z >> 16)),
                             fp8_pair_as_bf16x2(std::uint16_t(packed.w >> 16)), b_high01, b_high23,
                             d[m_tile][0], d[m_tile][1], d[m_tile][2], d[m_tile][3]);
        }
        const int scale_row = (tile * kOutputsPerTile) / 128;
        const int scale_col = slot * kBlocksPerSlot + (warp * kWarpK) / 128;
        const float scale = __bfloat162float(u.o_scales[scale_row * kScaleColumns + scale_col]);
#pragma unroll
        for (int m_tile = 0; m_tile < 4; ++m_tile) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            acc[m_tile][c] += d[m_tile][c] * scale;
          }
        }
      }
      // acc[m][0..1] = (output m*16+group, rows 2*tid4, 2*tid4+1); acc[m][2..3] = output + 8.
#pragma unroll
      for (int m_tile = 0; m_tile < 4; ++m_tile) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int row = 2 * tid4 + (c & 1);
          const int output = m_tile * 16 + group + (c >= 2 ? 8 : 0);
          if (row < kRows) {
            partial[(warp * kRows + row) * kOutputsPerTile + output] = acc[m_tile][c];
          }
        }
      }
    }
    __syncthreads();
    // 3. Sum the eight K slices per (row, output), in warp order like the standalone kernel.
    for (int i = tid; i < kRows * kOutputsPerTile; i += threads) {
      const int row = i / kOutputsPerTile;
      const int output = i - row * kOutputsPerTile;
      float sum = partial[(0 * kRows + row) * kOutputsPerTile + output];
#pragma unroll
      for (int w = 1; w < kComputeWarps; ++w) {
        sum += partial[(w * kRows + row) * kOutputsPerTile + output];
      }
      partial[row * kOutputsPerTile + output] = sum;   // rows of warp 0's block are free now? no: distinct index space per row below 8*kRows*64
    }
    __syncthreads();
    // 4. All-reduce: warp r handles row r (32 qwords of bf16x2), then residual + store.
    if (warp < kRows) {
      const int row = warp;
      const float* local_delta = partial + row * kOutputsPerTile;
      std::uint32_t* row_staged = staged + row * kQwordsPerTile;
      row_staged[lane] = naivert::pack_bf16x2(local_delta[2 * lane], local_delta[2 * lane + 1]);
      __syncwarp();
      const auto mailbox = naivert::TileMailbox<MailboxLayout>::make(
          u.mailbox_peers, u.rank, generation(u), std::size_t(row) * naivert::kMailboxSlabStride);
      mailbox.publish(tile, lane, row_staged[lane]);
      __syncwarp();
      if (lane < kQwordsPerTile / 2) {
        const int qword = lane * 2;
        naivert::Bf16x4Acc acc;
        // Issue the seven peer pair loads before inspecting any tag, so the arrival
        // round trip is paid once instead of seven times in sequence (microbench 2.9 -> 1.6 us).
        naivert::TaggedPair pair[naivert::kMailboxRanks - 1];
        const unsigned long long* ptr[naivert::kMailboxRanks - 1];
#pragma unroll
        for (int slot = 0; slot < naivert::kMailboxRanks - 1; ++slot) {
          ptr[slot] = mailbox.local_word(slot, tile, qword);
          pair[slot] = naivert::load_tagged_pair<true>(ptr[slot]);
        }
        bool pending;
        do {
          pending = false;
#pragma unroll
          for (int slot = 0; slot < naivert::kMailboxRanks - 1; ++slot) {
            if (naivert::tagged_tag(pair[slot].lo) != mailbox.generation ||
                naivert::tagged_tag(pair[slot].hi) != mailbox.generation) {
              pair[slot] = naivert::load_tagged_pair<true>(ptr[slot]);
              pending = true;
            }
          }
        } while (pending);
        if constexpr (kRows == 1) {
          // One row: the eight partials in rank order from zero (the one-row unproj recipe).
          acc = naivert::Bf16x4Acc::zero();
          naivert::accumulate_rank_order(acc, u.rank, row_staged[qword], row_staged[qword + 1],
                                         [&](int source_slot) {
                                           return naivert::PayloadPair{
                                               naivert::tagged_payload(pair[source_slot].lo),
                                               naivert::tagged_payload(pair[source_slot].hi)};
                                         });
        } else {
          // Several rows: local first, then the peer lanes.
          acc = naivert::Bf16x4Acc::of(row_staged[qword], row_staged[qword + 1]);
          naivert::accumulate_peers_in_slot_order(acc, [&](int source_slot) {
            return naivert::PayloadPair{naivert::tagged_payload(pair[source_slot].lo),
                                        naivert::tagged_payload(pair[source_slot].hi)};
          });
        }
        *reinterpret_cast<unsigned long long*>(row_staged + qword) = acc.packed();
      }
      __syncwarp();
      const auto* reduced = reinterpret_cast<const __nv_bfloat16*>(row_staged);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int output = lane * 2 + e;
        u.hidden_out[std::size_t(row) * kHidden + tile * kOutputsPerTile + output] = __float2bfloat16(
            __bfloat162float(reduced[output]) +
            __bfloat162float(shared_residual[row * kOutputsPerTile + output]));
      }
    }
  }
};

}  // namespace naivert::attn_unproj
