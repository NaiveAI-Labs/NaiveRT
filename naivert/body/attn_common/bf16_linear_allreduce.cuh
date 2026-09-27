#pragma once
#include "naivert/body/attn_common/rows_gemv_mma.cuh"
#include "naivert/body/attn_common/unproj_tail_args.hpp"
#include "naivert/common/tp8_mailbox.cuh"

namespace naivert::attn_unproj {
// BF16 checkpoint linears with FP32 local accumulation and TP partials.
// The same implementation serves the fused O tail and standalone O/dense down.
template <int Rows, int Hidden, int K, int Threads = 0>
struct Bf16LinearAllreduce {
  static constexpr int kTile = 64;
  static constexpr int kTiles = Hidden / kTile;
  static constexpr int kWarps = 8;
  static constexpr int kInputBytes = Rows * K * 2;
  static constexpr int kPartialBytes = kWarps * 128 * 4;
  static constexpr int kSharedBytes = kInputBytes + kPartialBytes + Rows * kTile * 4;
  struct Layout {
    __host__ __device__ static constexpr std::size_t tile_offset(int tile, int word) {
      return std::size_t(tile * kTile + word) * sizeof(unsigned long long);
    }
  };
  static_assert(kTiles * kTile * 8 <= naivert::kMailboxPhaseStride);

  __device__ static void run(const UnprojTailArgs& u, const __nv_bfloat16* weight,
                            const __nv_bfloat16* input, const __nv_bfloat16* residual,
                            unsigned char* shared, int tile) {
    if (tile < 0 || tile >= kTiles) return;
    static_assert(Threads == 0 || (Threads == 256 && Rows == 8));
    const int tid = int(threadIdx.x);
    if constexpr (Threads != 0) { if (tid >= Threads) return; }
    auto* x = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* partial = reinterpret_cast<float*>(shared + kInputBytes);
    auto* delta = reinterpret_cast<float*>(shared + kInputBytes + kPartialBytes);
    for (int i = tid; i < Rows * K / 8; i += (Threads ? Threads : int(blockDim.x)))
      reinterpret_cast<uint4*>(x)[i] = reinterpret_cast<const uint4*>(input)[i];
    naivert::attn_rows::projection_sync<Threads>();
    auto row_ptr = [&](int row) { return weight + std::size_t(row) * K; };
    auto emit = [&](int row, int query, float value) {
      delta[query * kTile + row % kTile] = value;
    };
    // Dense down uses shorter tensor-core chains; O retains its existing path.
    naivert::attn_rows::project_rows_mma<Rows, K, 2, 4, kWarps, K == 2048 ? 128 : 0, Threads>(
        0, Hidden / 16, row_ptr, x, partial, emit, tile);

    if constexpr (Rows == 1) {
      const unsigned epoch = u.replay_epoch ? unsigned(*u.replay_epoch) : 0u;
      const unsigned generation = naivert::mailbox_generation(epoch, u.mailbox_tag);
      for (int i = tid; i < Rows * kTile; i += (Threads ? Threads : int(blockDim.x))) {
        const int query = i / kTile, col = i % kTile;
        auto mailbox = naivert::TileMailbox<Layout>::make(
            u.mailbox_peers, u.rank, generation, std::size_t(query) * naivert::kMailboxSlabStride);
        mailbox.phase = u.replay_epoch ? epoch & 1u : mailbox.phase;
        mailbox.publish(tile, col, __float_as_uint(delta[i]));
      }
      naivert::attn_rows::projection_sync<Threads>();
      if constexpr (Rows == 1) {
        static_assert(8 * kTile * sizeof(float) <= kPartialBytes);
        // Decode has only 64 outputs. Use all eight warps to acquire peers,
        // then reuse the GEMM scratch for an ordered FP32 reduction.
        for (int item = tid; item < kTile * 8; item += (Threads ? Threads : int(blockDim.x))) {
          const int source = (item / 32) % 8;
          const int col = item / 256 * 32 + item % 32;
          auto mailbox = naivert::TileMailbox<Layout>::make(
              u.mailbox_peers, u.rank, generation, 0);
          mailbox.phase = u.replay_epoch ? epoch & 1u : mailbox.phase;
          partial[source * kTile + col] = source == u.rank ? delta[col] :
              __uint_as_float(naivert::tagged_payload(naivert::wait_tagged<false>(
                  mailbox.local_word(naivert::mailbox_source_slot(source, u.rank), tile, col), generation)));
        }
        naivert::attn_rows::projection_sync<Threads>();
      }
      for (int i = tid; i < Rows * kTile; i += (Threads ? Threads : int(blockDim.x))) {
        const int query = i / kTile, col = i % kTile;
        auto mailbox = naivert::TileMailbox<Layout>::make(
            u.mailbox_peers, u.rank, generation, std::size_t(query) * naivert::kMailboxSlabStride);
        mailbox.phase = u.replay_epoch ? epoch & 1u : mailbox.phase;
        float sum = 0.0f;
  #pragma unroll
        for (int source = 0; source < 8; ++source) {
          const float part = Rows == 1 ? partial[source * kTile + i] :
              source == u.rank ? delta[i] : __uint_as_float(naivert::tagged_payload(
              naivert::wait_tagged<false>(mailbox.local_word(naivert::mailbox_source_slot(source, u.rank),
                                                           tile, col), generation)));
          sum = __fadd_rn(sum, part);
        }
        const int index = query * Hidden + tile * kTile + col;
        // The completed linear output has the training BF16 boundary before residual addition.
        u.hidden_out[index] = __float2bfloat16_rn(
            __bfloat162float(__float2bfloat16_rn(sum)) + __bfloat162float(residual[index]));
      }
    } else {
      const unsigned epoch = u.replay_epoch ? unsigned(*u.replay_epoch) : 0u;
      const unsigned generation = naivert::mailbox_generation(epoch, u.mailbox_tag);
      auto box_for = [&](int query) {
        auto box = naivert::TileMailbox<Layout>::make(
            u.mailbox_peers, u.rank, generation, std::size_t(query) * naivert::kMailboxSlabStride);
        box.phase = u.replay_epoch ? epoch & 1u : box.phase;
        return box;
      };
      // Each owner receives eight coordinates. Final BF16 pairs use disjoint
      // word offsets 32..35 within the existing 64-word tile record.
      for (int i = tid; i < Rows * kTile; i += (Threads ? Threads : int(blockDim.x))) {
        const int query = i / kTile, col = i % kTile, owner = col / 8;
        if (owner != u.rank) {
          auto box = box_for(query);
          naivert::store_tagged(box.remote_word(owner, tile, col % 8),
              naivert::tagged_word(generation, __float_as_uint(delta[i])));
        }
      }
      naivert::attn_rows::projection_sync<Threads>();
      static_assert(Rows * 8 * 8 * sizeof(float) <= kPartialBytes);
      // Fetch rank partials in parallel, reusing dead GEMM scratch.
      if constexpr (Rows == 8) {
        // Preserve the existing shared layout and reduction. Start the two
        // independent query acquisitions before inspecting either generation.
        for (int first = tid; first < Rows * 64; first += 2 * (Threads ? Threads : int(blockDim.x))) {
          const int second = first + (Threads ? Threads : int(blockDim.x));
          const int q0 = first / 64, s0 = first / 8 % 8, c0 = first % 8;
          const int q1 = second / 64, s1 = second / 8 % 8, c1 = second % 8;
          const auto box0 = box_for(q0);
          const auto* p0 = s0 == u.rank ? nullptr : box0.local_word(
              naivert::mailbox_source_slot(s0, u.rank), tile, c0);
          const unsigned long long* p1 = nullptr;
          if (second < Rows * 64 && s1 != u.rank) {
            const auto box1 = box_for(q1);
            p1 = box1.local_word(naivert::mailbox_source_slot(s1, u.rank), tile, c1);
          }
          auto w0 = p0 ? naivert::load_tagged<false>(p0) : naivert::tagged_word(
              generation, __float_as_uint(delta[q0 * kTile + u.rank * 8 + c0]));
          auto w1 = naivert::tagged_word(generation, 0u);
          if (second < Rows * 64)
            w1 = p1 ? naivert::load_tagged<false>(p1) : naivert::tagged_word(
                generation, __float_as_uint(delta[q1 * kTile + u.rank * 8 + c1]));
          while (naivert::tagged_tag(w0) != generation ||
                 naivert::tagged_tag(w1) != generation) {
            if (naivert::tagged_tag(w0) != generation) w0 = naivert::load_tagged<false>(p0);
            if (naivert::tagged_tag(w1) != generation) w1 = naivert::load_tagged<false>(p1);
          }
          partial[first] = __uint_as_float(naivert::tagged_payload(w0));
          if (second < Rows * 64) partial[second] = __uint_as_float(naivert::tagged_payload(w1));
        }
      } else {
      for (int item = tid; item < Rows * 8 * 8; item += (Threads ? Threads : int(blockDim.x))) {
        const int query = item / 64, source = item / 8 % 8, coord = item % 8;
        auto box = box_for(query);
        partial[item] = source == u.rank ? delta[query * kTile + u.rank * 8 + coord] :
            __uint_as_float(naivert::tagged_payload(naivert::wait_tagged<false>(
                box.local_word(naivert::mailbox_source_slot(source, u.rank), tile, coord), generation)));
      }
      }
      naivert::attn_rows::projection_sync<Threads>();
      // Complete each owned output in source-rank order, preserving the BF16
      // linear boundary. The residual is added only after gathering results.
      auto* completed = reinterpret_cast<__nv_bfloat16*>(delta);
      for (int item = tid; item < Rows * 8; item += (Threads ? Threads : int(blockDim.x))) {
        const int query = item / 8, coord = item % 8;
        auto box = box_for(query);
        float sum = 0.0f;
  #pragma unroll
        for (int source = 0; source < 8; ++source) {
          const float part = partial[query * 64 + source * 8 + coord];
          sum = __fadd_rn(sum, part);
        }
        completed[item] = __float2bfloat16_rn(sum);
      }
      naivert::attn_rows::projection_sync<Threads>();
      for (int item = tid; item < Rows * 4; item += (Threads ? Threads : int(blockDim.x))) {
        const int query = item / 4, pair = item % 4;
        auto box = box_for(query);
        box.publish(tile, 32 + pair, reinterpret_cast<const unsigned*>(completed)[item]);
      }
      for (int item = tid; item < Rows * 32; item += (Threads ? Threads : int(blockDim.x))) {
        const int query = item / 32, owner = item / 4 % 8, pair = item % 4;
        auto box = box_for(query);
        const unsigned bits = owner == u.rank ? reinterpret_cast<const unsigned*>(completed)[query * 4 + pair] :
            naivert::tagged_payload(naivert::wait_tagged<false>(
                box.local_word(naivert::mailbox_source_slot(owner, u.rank), tile, 32 + pair), generation));
        const int index = query * Hidden + tile * kTile + owner * 8 + pair * 2;
        const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
        const float2 old = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(residual + index));
        *reinterpret_cast<__nv_bfloat162*>(u.hidden_out + index) = __floats2bfloat162_rn(
            __fadd_rn(value.x, old.x), __fadd_rn(value.y, old.y));
      }
    }

  }
};
}  // namespace naivert::attn_unproj
