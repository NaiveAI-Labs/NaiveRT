#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "naivert/body/down_allreduce/down_allreduce_args.hpp"
#include "naivert/body/down_allreduce/down_allreduce_traits.cuh"
#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_replica_down_allreduce {

namespace slot_ring = naivert;

struct DownAllreduceBody {
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;

  __device__ static void run(const DownAllreduceArgs& args, unsigned char* shared) {
    auto st = thread_state(shared);
    bind_shared_layout(st, args, shared);
    initialize_cta(st, args);
    if (!st.active) return;
    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
      helper_stage_output_tile(st, args, st.cta_rank);
    } else {
      consume_output_tile(st, args, st.cta_rank);
    }
  }

 protected:
  __device__ static void bind_shared_layout(slot_ring::SlotRingState& st,
                                            const DownAllreduceArgs&,
                                            unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kBarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup1Offset);
    st.early_base = base + kPayloadBase;
    st.deep_base = base + kPayloadBase;
    st.scratch_base = base;
    st.early_stride = kPayloadStride;
    st.deep_stride = kPayloadStride;
  }

  __device__ static slot_ring::SlotRingState thread_state(unsigned char* shared) {
    slot_ring::SlotRingState st{};
    st.tid = int(threadIdx.x);
    st.lane_id = slot_ring::StandardRoleLayout::lane_id(st.tid);
    st.warp_id = slot_ring::StandardRoleLayout::warp_id(st.tid);
    st.cta_rank = int(blockIdx.x);
    st.active = true;
    st.smem_base = shared;
    st.role = slot_ring::StandardRoleLayout::is_consumer_a(st.tid)
                  ? slot_ring::ThreadRole::kConsumerA
                  : slot_ring::StandardRoleLayout::is_consumer_b(st.tid)
                        ? slot_ring::ThreadRole::kConsumerB
                        : slot_ring::ThreadRole::kHelper;
    return st;
  }

  __device__ static void initialize_cta(slot_ring::SlotRingState& st, const DownAllreduceArgs& args) {
    int active = capped_active_ctas(args);
    if (active > int(gridDim.x)) active = int(gridDim.x);
    st.active = st.cta_rank < active;
    if (st.tid < kSlotDepth) Slot::init_transaction_slot(st, st.tid);
    cutlass::arch::fence_view_async_shared();
    __threadfence();
    asm volatile("bar.sync 0;" ::: "memory");
  }

  __host__ __device__ static int pane_storage_offset(int local_col, int k) {
    const int k_block = k >> 5;
    const int k_in_block = k & 31;
    int phys_row = 0;
    int phys_col = 0;
    if (local_col < kFullMTiles * 16) {
      const int chunk = local_col >> 4;
      const int row_in_chunk = local_col & 15;
      const int a = row_in_chunk >> 3;
      const int b = row_in_chunk & 7;
      const int d = k_in_block >> 4;
      const int e = (k_in_block >> 2) & 3;
      const int f = k_in_block & 3;
      const int out0 = b * 4 + e;
      const int out1 = d * 2 + a;
      phys_row = chunk * 16 + k_block * 2 + (out0 >> 4);
      phys_col = ((out0 & 15) << 4) + (out1 << 2) + f;
    } else {
      const int row8 = local_col - kFullMTiles * 16;
      const int d = k_in_block >> 4;
      const int e = (k_in_block >> 2) & 3;
      const int f = k_in_block & 3;
      phys_row = kFullMTiles * 16 + k_block;
      phys_col = row8 * 32 + e * 8 + d * 4 + f;
    }
    return phys_row * kInter + phys_col;
  }

  __device__ static int logical_hidden(const DownAllreduceArgs& args) {
    const int hidden = args.hidden > 0 ? args.hidden : kHidden;
    return hidden < kHidden ? hidden : kHidden;
  }

  __device__ static int effective_total_tiles(
      const DownAllreduceArgs& args) {
    const int logical =
        (logical_hidden(args) + kOutputsPerTile - 1) / kOutputsPerTile;
    if (args.total_tiles <= 0 || args.total_tiles > logical) {
      return logical;
    }
    return args.total_tiles;
  }

  __device__ static int capped_active_ctas(
      const DownAllreduceArgs& args) {
    const int total = effective_total_tiles(args);
    int active = args.active_ctas > 0 ? args.active_ctas : int(gridDim.x);
    return active < total ? active : total;
  }

  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static void set_inner_ring_slot(slot_ring::SlotRingState& st,
                                             int input_slot) {
    const int phase = input_slot / kSlotDepth;
    st.tx.slot = input_slot - phase * kSlotDepth;
    st.tx.iteration = input_slot;
    st.tx.phase = phase;
    st.tx.phase = phase;
    st.tx.phase = phase;
  }

  __device__ static void helper_stage_output_tile(
      slot_ring::SlotRingState& st, const DownAllreduceArgs& args, int output_tile) {
    if (output_tile < 0) {
      return;
    }
    const slot_ring::TransactionState outer_tx = st.tx;
#pragma unroll
    for (int input_slot = 0; input_slot < kInputSlots; ++input_slot) {
      set_inner_ring_slot(st, input_slot);
      Slot::producer_acquire(st);
      stage_weight_pane(st, args, output_tile, input_slot,
                        st.deep_slot_ptr());
      Slot::producer_commit(st);
    }
    st.tx = outer_tx;
  }

  __device__ static void stage_weight_pane(
      slot_ring::SlotRingState& st, const DownAllreduceArgs& args, int output_tile,
      int input_slot, unsigned char* pane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (args.w_down != nullptr) {
      const auto* src = reinterpret_cast<const unsigned char*>(
          args.w_down +
          (input_slot * kHidden +
           output_tile * kOutputsPerTile) *
              kInter);
      if (st.elected) {
        Slot::producer_expect_tx(st, std::uint32_t(kWeightPaneBytes));
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{}, pane,
            src, std::uint32_t(kWeightPaneBytes),
            Slot::slot_barrier(st.full_barriers, st.tx.slot));
        stage_scale_pane(st, args, output_tile, input_slot, pane);
      }
      return;
    }
#endif
    const int helper_tid =
        st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
    for (int off = helper_tid * int(sizeof(std::uint32_t));
         off < kWeightPaneBytes;
         off += kHelperThreads * int(sizeof(std::uint32_t))) {
      *reinterpret_cast<std::uint32_t*>(pane + off) = 0u;
    }
  }

  __device__ static void stage_scale_pane(
      slot_ring::SlotRingState& st, const DownAllreduceArgs& args, int output_tile,
      int input_slot, unsigned char* pane) {
    auto* scale = reinterpret_cast<uint4*>(pane + kScalePaneOffset);
    if (args.w_down_scale == nullptr) {
      const __nv_bfloat16 one = __float2bfloat16(1.0f);
      const std::uint32_t one_pair =
          static_cast<std::uint32_t>(
              *reinterpret_cast<const std::uint16_t*>(&one)) *
          0x10001u;
      const uint4 ones{one_pair, one_pair, one_pair, one_pair};
      scale[0] = ones;
      scale[1] = ones;
      return;
    }
    const auto* src = reinterpret_cast<const unsigned char*>(
        args.w_down_scale + input_slot * (kTiles * 16) + output_tile * 16);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const auto smem =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(scale));
    const auto gmem = reinterpret_cast<std::uint64_t>(src);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n\t"
        "cp.async.cg.shared.global [%2], [%3], 16;"
        :
        : "r"(smem), "l"(gmem), "r"(smem + 16), "l"(gmem + 16)
        : "memory");
    // cp.async.mbarrier.arrive lowers to the native
    // RED.A0T1 + ARRIVES.LDGSTSBAR transaction bridge.  It binds both
    // 16-byte LDGSTS operations above to the same group0 barrier as the
    // UBLKCP weight pane without extending RED.A0TR's 0x3800 byte count.
    Slot::producer_cp_async_commit(st);
#else
    const auto* sync_src = reinterpret_cast<const uint4*>(src);
    scale[0] = sync_src[0];
    scale[1] = sync_src[1];
#endif
  }

  __device__ static float load_staged_scale(
      const unsigned char* pane, int output_tile, int hidden_col,
      int k_group) {
    const auto* scale = reinterpret_cast<const __nv_bfloat16*>(
        pane + kScalePaneOffset);
    const int output_base = output_tile * kOutputsPerTile;
    const int output_group = (hidden_col - output_base) / 8;
    const int group = k_group < 1 ? 0 : 1;
    return __bfloat162float(scale[output_group * 2 + group]);
  }

  __device__ static std::uint32_t convert_fp8_pair_as_bf16x2(
      std::uint16_t raw) {
    std::uint32_t out;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out)
                 : "h"(raw));
    return out;
  }

  __device__ static std::uint32_t load_shared_fp8_pair_as_bf16x2(
      const unsigned char* ptr) {
    return convert_fp8_pair_as_bf16x2(
        *reinterpret_cast<const std::uint16_t*>(ptr));
  }

  __device__ static std::uint32_t load_bf16_pair(
      const __nv_bfloat16* ptr) {
    // bf16 activations feed the bf16 HMMA directly: no fp16 widening, so
    // no 65,504 ceiling.
    return *reinterpret_cast<const std::uint32_t*>(ptr);
  }

  __device__ static void mma_16816_f32_bf16(
      std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
      std::uint32_t a3, std::uint32_t b0, std::uint32_t b1, float& c0,
      float& c1, float& c2, float& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  static constexpr int kTpRanks = naivert::geometry::kTpRanks;
  // Native DownAllreduce uses one slab per group of 32 CTAs and one 0x400
  // record per CTA (four warp records; the compact payload is warp 0's).
  struct MailboxLayout {
    __host__ __device__ static constexpr std::size_t tile_offset(
        int task, int qword) {
      return std::size_t(task >> 5) * naivert::kMailboxSlabStride +
             std::size_t(task & 31) * naivert::kMailboxCtaStride +
             std::size_t(qword) * sizeof(unsigned long long);
    }
  };

  // Row-r tiles in a multi-row layout live four slabs further in.
  __device__ static naivert::TileMailbox<MailboxLayout> tile_mailbox(
      const DownAllreduceArgs& args, unsigned gen, int row) {
    return naivert::TileMailbox<MailboxLayout>::make(
        args.mailbox_peers, args.rank, gen,
        std::size_t(row) * 4 * naivert::kMailboxSlabStride);
  }
  static constexpr int kMailboxQwordsPerTile = kOutputsPerTile / 2;

  __device__ static bool allreduce_enabled(
      const DownAllreduceArgs& args) {
    // The fixed-graph host contract validates every persistent pointer before
    // capture.  Keep rank=-1 as the standalone compute-only switch, but do not
    // re-check eight mailbox pointers in every consumer thread on every replay.
    return args.rank >= 0 && args.rank < kTpRanks;
  }

  __device__ static unsigned step_gen(const DownAllreduceArgs& args) {
    if (args.replay_generation != nullptr) {
      return naivert::mailbox_generation(
          static_cast<unsigned>(*args.replay_generation),
          args.generation_salt);
    }
    if (args.step_block_base == nullptr) {
      return 1u;
    }
    return *reinterpret_cast<const unsigned*>(args.step_block_base + 8);
  }

  using TaggedPair = naivert::TaggedPair;

  __device__ static void publish_and_reduce_tile(
      slot_ring::SlotRingState& st, const DownAllreduceArgs& args, int task,
      float* local_delta) {
    auto* staged = reinterpret_cast<std::uint32_t*>(
        st.scratch_base + kStagedPayloadOffset);
    auto* staged_residual = reinterpret_cast<__nv_bfloat16*>(
        st.scratch_base + kStagedResidualOffset);
    const int output_tile_base = task * kOutputsPerTile;
    if (st.tid < kMailboxQwordsPerTile) {
      const int elem = st.tid * 2;
      staged[st.tid] =
          naivert::pack_bf16x2(local_delta[elem], local_delta[elem + 1]);
    }
    // The 28 pack/publish lanes and the 14 pair reducers all reside in
    // consumer warp 0.  The preceding full consumer barrier already made the
    // reduced float payload visible; only a warp rendezvous is needed here.
    __syncwarp();

    const auto mailbox = tile_mailbox(args, step_gen(args), 0);
    if (st.tid < kMailboxQwordsPerTile) {
      mailbox.publish(task, st.tid, staged[st.tid]);
    }
    __syncwarp();
    constexpr int kQwordPairs = kMailboxQwordsPerTile / 2;
    if (st.tid < kQwordPairs) {
      const int qword = st.tid * 2;
      // Order pinned by the golden: local first, then the peer lanes,
      // rounded to bf16, then the residual (below).
      naivert::Bf16x4Acc acc =
          naivert::Bf16x4Acc::of(staged[qword], staged[qword + 1]);
      naivert::accumulate_peers_in_slot_order(
          acc, [&](int source_slot) {
            return mailbox.wait_pair<false>(source_slot, task, qword);
          });
      *reinterpret_cast<unsigned long long*>(staged + qword) = acc.packed();
    }
    slot_ring::consumer_group_named_sync();

    // Native redistributes the final epilogue over all 56 output lanes after
    // the peer reducer has overwritten the local shared payload with the TP
    // sum.  This shortens the reducer's live range and leaves only one U16
    // output store on the production alias path.
    if (st.tid < kOutputsPerTile) {
      const int output = output_tile_base + st.tid;
      const float residual = __bfloat162float(staged_residual[st.tid]);
      const auto* reduced =
          reinterpret_cast<const __nv_bfloat16*>(staged);
      auto* result =
          args.tp_result != nullptr ? args.tp_result : args.exp_out;
      result[output] =
          __float2bfloat16(__bfloat162float(reduced[st.tid]) + residual);
      if (args.exp_out != result) {
        args.exp_out[output] =
            __float2bfloat16(local_delta[st.tid] + residual);
      }
    }
  }

  __device__ __forceinline__ static void accumulate_input_slot(
      const DownAllreduceArgs& args, int output_tile, int input_slot,
      const unsigned char* pane, const __nv_bfloat16* vec, int warp,
      int group, int tid4, float& out0, float& out1, float& out2,
      float& out3, float& out4, float& out5, float& out6, float& out7) {
    const int k_base = warp * 32;
    const int lane_k4 = k_base + tid4 * 4;
    const std::uint32_t b_low01 =
        load_bf16_pair(vec + lane_k4);
    const std::uint32_t b_low23 =
        load_bf16_pair(vec + lane_k4 + 16);
    const std::uint32_t b_high01 =
        load_bf16_pair(vec + lane_k4 + 2);
    const std::uint32_t b_high23 =
        load_bf16_pair(vec + lane_k4 + 18);
    float d[4][4] = {};
#pragma unroll
    for (int m_tile = 0; m_tile < kFullMTiles; ++m_tile) {
      const int row = m_tile * 16 + group;
      const auto packed = *reinterpret_cast<const uint4*>(
          pane + pane_storage_offset(row, lane_k4));
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w)), b_low01,
          b_low23, d[m_tile][0], d[m_tile][1], d[m_tile][2],
          d[m_tile][3]);
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w >> 16)),
          b_high01, b_high23, d[m_tile][0], d[m_tile][1],
          d[m_tile][2], d[m_tile][3]);
    }
    if constexpr (kTailRows == 8) {
      const int tail_row = kFullMTiles * 16 + group;
      mma_16816_f32_bf16(
          load_shared_fp8_pair_as_bf16x2(
              pane + pane_storage_offset(tail_row, lane_k4)),
          0u,
          load_shared_fp8_pair_as_bf16x2(
              pane + pane_storage_offset(tail_row, lane_k4 + 16)),
          0u, b_low01, b_low23, d[3][0], d[3][1], d[3][2], d[3][3]);
      mma_16816_f32_bf16(
          load_shared_fp8_pair_as_bf16x2(
              pane + pane_storage_offset(tail_row, lane_k4 + 2)),
          0u,
          load_shared_fp8_pair_as_bf16x2(
              pane + pane_storage_offset(tail_row, lane_k4 + 18)),
          0u, b_high01, b_high23, d[3][0], d[3][1], d[3][2],
          d[3][3]);
    }

    if (tid4 == 0) {
      const int output_base = output_tile * kOutputsPerTile;
      const int k_group = warp >> 2;
      out0 += d[0][0] *
              load_staged_scale(
                  pane, output_tile, output_base + group, k_group);
      out1 += d[0][2] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 8, k_group);
      out2 += d[1][0] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 16, k_group);
      out3 += d[1][2] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 24, k_group);
      out4 += d[2][0] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 32, k_group);
      out5 += d[2][2] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 40, k_group);
      out6 += d[3][0] *
              load_staged_scale(
                  pane, output_tile, output_base + group + 48, k_group);
      if constexpr (kTailRows == 0) {
        out7 += d[3][2] *
                load_staged_scale(
                    pane, output_tile, output_base + group + 56, k_group);
      }
    }
  }

  __device__ static void consume_output_tile(
      slot_ring::SlotRingState& st, const DownAllreduceArgs& args, int output_tile) {
    if (!is_consumer(st) || output_tile < 0) {
      return;
    }
    const slot_ring::TransactionState outer_tx = st.tx;
    auto* shared_up_gate =
        reinterpret_cast<__nv_bfloat16*>(st.scratch_base);
    auto* shared_residual = reinterpret_cast<uint4*>(
        st.scratch_base + kStagedResidualOffset);
    constexpr int kResidualVectorCount =
        kOutputsPerTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
    if (st.tid < kResidualVectorCount) {
      if (args.residual != nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const auto smem = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(shared_residual + st.tid));
        const auto gmem = reinterpret_cast<std::uint64_t>(
            args.residual + output_tile * kOutputsPerTile +
            st.tid * int(sizeof(uint4) / sizeof(__nv_bfloat16)));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :
                     : "r"(smem), "l"(gmem)
                     : "memory");
#else
        shared_residual[st.tid] =
            reinterpret_cast<const uint4*>(
                args.residual + output_tile * kOutputsPerTile)[st.tid];
#endif
      } else {
        shared_residual[st.tid] = uint4{0u, 0u, 0u, 0u};
      }
    }
    constexpr int kInputVectorCount =
        kLocalInter * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
    for (int vector = st.tid; vector < kInputVectorCount;
         vector += kConsumerThreads) {
      reinterpret_cast<uint4*>(shared_up_gate)[vector] =
          reinterpret_cast<const uint4*>(args.up_gate)[vector];
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Match native's LDGSTS -> independent input staging -> LDGDEPBAR order:
    // leave the residual copy in flight while the warp loads UP_GATE, then
    // retire it immediately before the existing consumer rendezvous.
    asm volatile("cp.async.wait_all;" : : : "memory");
#endif
    slot_ring::consumer_group_named_sync();

    const int warp = st.warp_id;
    const int group = st.lane_id >> 2;
    const int tid4 = st.lane_id & 3;
    float p0 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
    float p3 = 0.0f;
    float p4 = 0.0f;
    float p5 = 0.0f;
    float p6 = 0.0f;
    float p7 = 0.0f;

    for (int input_slot = 0; input_slot < kInputSlots; ++input_slot) {
      set_inner_ring_slot(st, input_slot);
      Slot::consumer_wait(st);
      accumulate_input_slot(
          args, output_tile, input_slot, st.deep_slot_ptr(),
          shared_up_gate + input_slot * kInter, warp, group, tid4, p0, p1,
          p2, p3, p4, p5, p6, p7);
      Slot::consumer_release(st);
    }

    float* partial =
        reinterpret_cast<float*>(st.scratch_base + kPartialOffset);
    if (tid4 == 0) {
      const int base = warp * kPartialRows;
      partial[base + group] = p0;
      partial[base + group + 8] = p1;
      partial[base + group + 16] = p2;
      partial[base + group + 24] = p3;
      partial[base + group + 32] = p4;
      partial[base + group + 40] = p5;
      partial[base + group + 48] = p6;
      if constexpr (kTailRows == 0) {
        partial[base + group + 56] = p7;
      }
    }
    slot_ring::consumer_group_named_sync();

    if (st.tid < kOutputsPerTile) {
      float sum = partial[st.tid];
#pragma unroll
      for (int source_warp = 1; source_warp < 8; ++source_warp) {
        sum += partial[source_warp * kPartialRows + st.tid];
      }
      const int output = output_tile * kOutputsPerTile + st.tid;
      if (output < logical_hidden(args)) {
        partial[st.tid] = sum;
      }
    }
    slot_ring::consumer_group_named_sync();
    if (allreduce_enabled(args)) {
      publish_and_reduce_tile(st, args, output_tile, partial);
    } else if (st.tid < kOutputsPerTile) {
      const int output = output_tile * kOutputsPerTile + st.tid;
      if (output < logical_hidden(args)) {
        const float residual =
            args.residual != nullptr
                ? __bfloat162float(args.residual[output])
                : 0.0f;
        const __nv_bfloat16 value =
            __float2bfloat16(partial[st.tid] + residual);
        args.exp_out[output] = value;
        if (args.tp_result != nullptr) {
          args.tp_result[output] = value;
        }
      }
    }
    st.tx = outer_tx;
  }
};

}  // namespace naivert_replica_down_allreduce
