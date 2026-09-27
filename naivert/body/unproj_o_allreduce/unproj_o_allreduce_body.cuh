#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "naivert/body/unproj_o_allreduce/unproj_o_allreduce_args.hpp"
#include "naivert/body/unproj_o_allreduce/unproj_o_allreduce_traits.cuh"
#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/common/role_layout.hpp"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_replica_unproj_o_allreduce {

namespace slot_ring = naivert;

struct UnprojOAllreduceBody;

}  // namespace naivert_replica_unproj_o_allreduce

namespace naivert_replica_unproj_o_allreduce {

struct UnprojOAllreduceBody {
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;

  __device__ static slot_ring::SlotRingWorkArgs work_args(
      const UnprojOAllreduceArgs& args) {
    slot_ring::SlotRingWorkArgs progress{};
    progress.total_tiles = effective_total_tiles(args);
    progress.active_ctas = capped_active_ctas(args);
    return progress;
  }

  __device__ static void bind_shared_layout(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs&,
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

  __device__ static void init_control_slab(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs&) {
    if (st.tid < kSlotDepth) {
      Slot::init_transaction_slot(st, st.tid);
    }
  }

  __device__ static void dispatch(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs& args) {
    if (st.cta_rank >= capped_active_ctas(args)) {
      st.active = false;
    }
  }

  __device__ static int slot_task(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs& args, int) {
    if (st.cta_rank >= capped_active_ctas(args) ||
        st.tx.iteration != 0) {
      return -1;
    }
    return st.cta_rank;
  }

  // The kernel's control flow: role split, layout binding, slot init on the
  // control lanes, async-view fence +
  // __threadfence and raw BAR0, dispatch, one weight pane staged by the
  // helpers, then one output tile per active CTA consumed by the consumer
  // warps (the helpers have nothing to refill).  kPdl: the consumers wait for
  // the primary grid (attention) before touching proj_o or the residual; the
  // helpers stage the immutable weight pane meanwhile.
  template <bool kPdl>
  __device__ static void run(const UnprojOAllreduceArgs& args,
                             unsigned char* shared) {
    slot_ring::SlotRingState st{};
    st.tid = int(threadIdx.x);
    st.lane_id = slot_ring::StandardRoleLayout::lane_id(st.tid);
    st.warp_id = slot_ring::StandardRoleLayout::warp_id(st.tid);
    st.cta_rank = int(blockIdx.x);
    st.active = true;
    st.elected = false;
    st.smem_base = shared;
    st.control_base = nullptr;
    st.full_barriers = nullptr;
    st.empty_barriers = nullptr;
    st.early_base = nullptr;
    st.deep_base = nullptr;
    st.scratch_base = nullptr;
    st.early_stride = 0;
    st.deep_stride = 0;
    st.body_acc[0] = 0.0f;
    st.body_acc[1] = 0.0f;
    st.tx = {0, 0, 0};
    if (slot_ring::StandardRoleLayout::is_consumer_a(st.tid)) {
      st.role = slot_ring::ThreadRole::kConsumerA;
    } else if (slot_ring::StandardRoleLayout::is_consumer_b(st.tid)) {
      st.role = slot_ring::ThreadRole::kConsumerB;
    } else {
      st.role = slot_ring::ThreadRole::kHelper;
    }

    bind_shared_layout(st, args, shared);
    {
      const slot_ring::SlotRingWorkArgs work = work_args(args);
      int active_ctas =
          work.active_ctas > 0 ? work.active_ctas : int(gridDim.x);
      if (active_ctas > int(gridDim.x)) {
        active_ctas = int(gridDim.x);
      }
      if (work.total_tiles < active_ctas) {
        active_ctas = work.total_tiles;
      }
      st.active = st.active && st.cta_rank < active_ctas;
    }
    if (slot_ring::StandardRoleLayout::is_control_init_lane(st.tid)) {
      init_control_slab(st, args);
    }
    cutlass::arch::fence_view_async_shared();
    __threadfence();
#if defined(__CUDA_ARCH__)
    asm volatile("bar.sync 0;" ::: "memory");  // BAR0
#endif
    if (!st.active) {
      return;
    }
    dispatch(st, args);

    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
      const int old_slot = st.tx.slot;
      const int old_iteration = st.tx.iteration;
      st.tx.slot = 0;
      st.tx.iteration = 0;
      helper_stage_output_tile(st, args, slot_task(st, args, 0));
      st.tx.slot = old_slot;
      st.tx.iteration = old_iteration;
    }

    while (true) {
      const int slot = st.tx.slot;
      if (is_consumer(st)) {
        const int task = slot_task(st, args, slot);
        if (task < 0) {
          break;
        }
        if constexpr (kPdl) {
          naivert::pdl::wait_for_primary<true>();
        }
        consume_output_tile(st, args, task);
      } else {
        break;
      }
      Slot::advance(st);
    }
  }

  // Inverse of the reference _swizzle_qmma_16x32.  The returned byte offset is
  // relative to one native [64 output, 512 K] 0x8000-byte pane.
  __host__ __device__ static int pane_storage_offset(
      int local_row, int k) {
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
    const int y_linear =
        ((((s2 * 2 + s0) * 8 + s1) * 4 + s3) * 4 + s4);
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

 protected:
  static constexpr int kTpRanks = naivert::geometry::kTpRanks;
  static constexpr int kMailboxQwordsPerTile = kOutputsPerTile / 2;
  // One 0x100-byte record per tile: 32 tagged qwords = 64 bf16 outputs.
  struct MailboxLayout {
    __host__ __device__ static constexpr std::size_t tile_offset(
        int task, int qword) {
      return std::size_t(task) * naivert::kMailboxWarpStride +
             std::size_t(qword) * sizeof(unsigned long long);
    }
  };

  __device__ static unsigned mailbox_generation(
      const UnprojOAllreduceArgs& args) {
    const auto* replay_iteration =
        reinterpret_cast<const std::uint64_t*>(args.step_block_base);
    const unsigned replay =
        replay_iteration == nullptr
            ? 0u
            : static_cast<unsigned>(*replay_iteration);
    return naivert::mailbox_generation(replay, args.mailbox_tag);
  }

  // Row-r tiles in a multi-row layout live one slab further in.
  __device__ static naivert::TileMailbox<MailboxLayout> tile_mailbox(
      const UnprojOAllreduceArgs& args, int row) {
    return naivert::TileMailbox<MailboxLayout>::make(
        args.mailbox_peers, args.rank, mailbox_generation(args),
        std::size_t(row) * naivert::kMailboxSlabStride);
  }

  __device__ static int logical_hidden(
      const UnprojOAllreduceArgs& args) {
    const int hidden = args.hidden > 0 ? args.hidden : kHidden;
    return hidden < kHidden ? hidden : kHidden;
  }

  __device__ static int effective_total_tiles(
      const UnprojOAllreduceArgs& args) {
    const int logical =
        (logical_hidden(args) + kOutputsPerTile - 1) /
        kOutputsPerTile;
    if (args.total_tiles <= 0 || args.total_tiles > logical) {
      return logical;
    }
    return args.total_tiles;
  }

  __device__ static int capped_active_ctas(
      const UnprojOAllreduceArgs& args) {
    const int total = effective_total_tiles(args);
    int active =
        args.active_ctas > 0 ? args.active_ctas : int(gridDim.x);
    return active < total ? active : total;
  }

  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static void set_inner_ring_slot(
      slot_ring::SlotRingState& st, int input_slot) {
    st.tx.slot = input_slot;
    st.tx.iteration = input_slot;
    st.tx.phase = 0;
    st.tx.phase = 0;
    st.tx.phase = 0;
  }

  __device__ static void helper_stage_output_tile(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs& args,
      int output_tile) {
    if (output_tile < 0) {
      return;
    }
    const slot_ring::TransactionState outer_tx = st.tx;
#pragma unroll
    for (int input_slot = 0; input_slot < kInputSlots; ++input_slot) {
      set_inner_ring_slot(st, input_slot);
      Slot::producer_acquire(st);
      unsigned char* pane = st.deep_slot_ptr();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (args.weight != nullptr) {
        const auto* src = reinterpret_cast<const unsigned char*>(
            args.weight +
            (output_tile * kInputSlots + input_slot) *
                kWeightPaneBytes);
        if (st.elected) {
          Slot::producer_expect_tx(
              st, std::uint32_t(kWeightPaneBytes));
          // The four weight panes are consumed once by this CTA.  Marking
          // the 14 MiB graph-wide stream evict-first prevents it from
          // displacing the next layer's reused MLA weights from L2.
          const std::uint32_t smem_dst =
              static_cast<std::uint32_t>(
                  __cvta_generic_to_shared(pane));
          auto* slot_barrier = Slot::slot_barrier(
              st.full_barriers, st.tx.slot);
          const std::uint32_t smem_barrier =
              static_cast<std::uint32_t>(
                  __cvta_generic_to_shared(slot_barrier));
          std::uint64_t cache_policy = 0;
          asm volatile(
              "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
              : "=l"(cache_policy));
          asm volatile(
              "cp.async.bulk.shared::cta.global."
              "mbarrier::complete_tx::bytes.L2::cache_hint "
              "[%0], [%1], %2, [%3], %4;"
              :
              : "r"(smem_dst), "l"(src),
                "r"(std::uint32_t(kWeightPaneBytes)),
                "r"(smem_barrier), "l"(cache_policy)
              : "memory");
        }
      } else
#endif
      {
        const int helper_tid =
            st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
        for (int off =
                 helper_tid * int(sizeof(std::uint32_t));
             off < kWeightPaneBytes;
             off +=
                 kHelperThreads * int(sizeof(std::uint32_t))) {
          *reinterpret_cast<std::uint32_t*>(pane + off) = 0u;
        }
      }
      Slot::producer_commit(st);
    }
    st.tx = outer_tx;
  }

  __device__ static std::uint32_t convert_fp8_pair_as_bf162(
      std::uint16_t raw) {
    std::uint32_t out;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out)
                 : "h"(raw));
    return out;
  }

  __device__ static std::uint32_t load_bf16_pair(
      const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const std::uint32_t*>(ptr);
  }

  __device__ static void mma_16816_f32_bf16(
      std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
      std::uint32_t a3, std::uint32_t b0, std::uint32_t b1,
      float& c0, float& c1, float& c2, float& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
  }

  __device__ static float scale_for_partition(
      const UnprojOAllreduceArgs& args, int output_tile,
      int input_slot, int warp) {
    if (args.scales == nullptr) {
      return 1.0f;
    }
    const int scale_row = (output_tile * kOutputsPerTile) / 128;
    const int scale_col =
        input_slot * kBlocksPerSlot + (warp * kWarpK) / 128;
    return __bfloat162float(
        args.scales[scale_row * kScaleColumns + scale_col]);
  }

  __device__ __forceinline__ static void accumulate_input_slot(
      const UnprojOAllreduceArgs& args, int output_tile,
      int input_slot, const unsigned char* pane,
      const __nv_bfloat16* vec, int warp, int group, int tid4,
      float& out0, float& out1, float& out2, float& out3,
      float& out4, float& out5, float& out6, float& out7) {
    float d[4][4] = {};
    const int warp_k_base = warp * kWarpK;
#pragma unroll
    for (int half = 0; half < kWarpK / 32; ++half) {
      const int lane_k4 =
          warp_k_base + half * 32 + tid4 * 4;
      const std::uint32_t b_low01 =
          load_bf16_pair(vec + lane_k4);
      const std::uint32_t b_low23 =
          load_bf16_pair(vec + lane_k4 + 16);
      const std::uint32_t b_high01 =
          load_bf16_pair(vec + lane_k4 + 2);
      const std::uint32_t b_high23 =
          load_bf16_pair(vec + lane_k4 + 18);
#pragma unroll
      for (int m_tile = 0; m_tile < 4; ++m_tile) {
        const int row = m_tile * 16 + group;
        const auto packed = *reinterpret_cast<const uint4*>(
            pane + pane_storage_offset(row, lane_k4));
        mma_16816_f32_bf16(
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.x)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.y)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.z)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.w)),
            b_low01, b_low23, d[m_tile][0], d[m_tile][1],
            d[m_tile][2], d[m_tile][3]);
        mma_16816_f32_bf16(
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.x >> 16)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.y >> 16)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.z >> 16)),
            convert_fp8_pair_as_bf162(
                std::uint16_t(packed.w >> 16)),
            b_high01, b_high23, d[m_tile][0], d[m_tile][1],
            d[m_tile][2], d[m_tile][3]);
      }
    }
    if (tid4 == 0) {
      const float scale =
          scale_for_partition(args, output_tile, input_slot, warp);
      out0 += d[0][0] * scale;
      out1 += d[0][2] * scale;
      out2 += d[1][0] * scale;
      out3 += d[1][2] * scale;
      out4 += d[2][0] * scale;
      out5 += d[2][2] * scale;
      out6 += d[3][0] * scale;
      out7 += d[3][2] * scale;
    }
  }

  __device__ static bool allreduce_enabled(
      const UnprojOAllreduceArgs& args) {
    return args.rank >= 0 && args.rank < kTpRanks;
  }

  using TaggedPair = naivert::TaggedPair;

  __device__ static void publish_and_reduce_tile(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs& args,
      int task, float* local_delta) {
    auto* staged = reinterpret_cast<std::uint32_t*>(
        st.scratch_base + kStagedPayloadOffset);
    auto* staged_residual =
        reinterpret_cast<const __nv_bfloat16*>(
            st.scratch_base + kStagedResidualOffset);
    if (st.tid < kMailboxQwordsPerTile) {
      const int elem = st.tid * 2;
      staged[st.tid] = naivert::pack_bf16x2(
          local_delta[elem], local_delta[elem + 1]);
    }
    __syncwarp();
    const auto mailbox = tile_mailbox(args, 0);
    if (st.tid < kMailboxQwordsPerTile) {
      mailbox.publish(task, st.tid, staged[st.tid]);
    }
    __syncwarp();
    if (st.tid < kMailboxQwordsPerTile / 2) {
      const int qword = st.tid * 2;
      // Order pinned by the golden: the eight partials in rank order from
      // zero, rounded to bf16, then the residual (below).
      naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::zero();
      naivert::accumulate_rank_order(
          acc, args.rank, staged[qword], staged[qword + 1],
          [&](int source_slot) {
            return mailbox.wait_pair<true>(source_slot, task, qword);
          });
      *reinterpret_cast<unsigned long long*>(staged + qword) = acc.packed();
    }
    slot_ring::consumer_group_named_sync();
    if (st.tid < kOutputsPerTile) {
      const int output = task * kOutputsPerTile + st.tid;
      const auto* reduced =
          reinterpret_cast<const __nv_bfloat16*>(staged);
      const float residual =
          __bfloat162float(staged_residual[st.tid]);
      args.output[output] = __float2bfloat16(
          __bfloat162float(reduced[st.tid]) + residual);
    }
  }

  __device__ static void consume_output_tile(
      slot_ring::SlotRingState& st, const UnprojOAllreduceArgs& args,
      int output_tile) {
    if (!is_consumer(st) || output_tile < 0) {
      return;
    }
    const slot_ring::TransactionState outer_tx = st.tx;
    auto* shared_proj = reinterpret_cast<__nv_bfloat16*>(
        st.scratch_base + kInputOffset);
    auto* shared_residual = reinterpret_cast<uint4*>(
        st.scratch_base + kStagedResidualOffset);

    constexpr int kInputVectorCount =
        kProjectionK * int(sizeof(__nv_bfloat16)) /
        int(sizeof(uint4));
    if (st.tid < kInputVectorCount) {
      if (args.proj_o != nullptr) {
        reinterpret_cast<uint4*>(shared_proj)[st.tid] =
            reinterpret_cast<const uint4*>(args.proj_o)[st.tid];
      } else {
        reinterpret_cast<uint4*>(shared_proj)[st.tid] =
            uint4{0u, 0u, 0u, 0u};
      }
      // Variant-1 short MLA leaves a complete invalid head as BF16 NaNs when
      // that head has no valid attention contribution.  Native UnprojO treats
      // the invalid contribution as zero; feeding it directly to HMMA instead
      // poisons every output row and then every TP rank through AllReduce.
      // Sanitize the same uint4 this thread staged, avoiding a cross-thread
      // dependency before the existing consumer-group barrier below.
      constexpr int kBf16PerInputVector =
          int(sizeof(uint4)) / int(sizeof(__nv_bfloat16));
      const int element_base = st.tid * kBf16PerInputVector;
#pragma unroll
      for (int lane = 0; lane < kBf16PerInputVector; ++lane) {
        const int element = element_base + lane;
        const float value = __bfloat162float(shared_proj[element]);
        if (!isfinite(value)) {
          shared_proj[element] = __float2bfloat16(0.0f);
        }
      }
    }

    constexpr int kResidualVectorCount =
        kOutputsPerTile * int(sizeof(__nv_bfloat16)) /
        int(sizeof(uint4));
    if (st.tid < kResidualVectorCount) {
      if (args.residual != nullptr) {
        shared_residual[st.tid] =
            reinterpret_cast<const uint4*>(
                args.residual +
                output_tile * kOutputsPerTile)[st.tid];
      } else {
        shared_residual[st.tid] =
            uint4{0u, 0u, 0u, 0u};
      }
    }
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

#pragma unroll
    for (int input_slot = 0; input_slot < kInputSlots;
         ++input_slot) {
      set_inner_ring_slot(st, input_slot);
      Slot::consumer_wait(st);
      accumulate_input_slot(
          args, output_tile, input_slot, st.deep_slot_ptr(),
          shared_proj + input_slot * kInter, warp, group, tid4,
          p0, p1, p2, p3, p4, p5, p6, p7);
      Slot::consumer_release(st);
    }

    float* partial = reinterpret_cast<float*>(
        st.scratch_base + kPartialOffset);
    if (tid4 == 0) {
      const int base = warp * kOutputsPerTile;
      partial[base + group] = p0;
      partial[base + group + 8] = p1;
      partial[base + group + 16] = p2;
      partial[base + group + 24] = p3;
      partial[base + group + 32] = p4;
      partial[base + group + 40] = p5;
      partial[base + group + 48] = p6;
      partial[base + group + 56] = p7;
    }
    slot_ring::consumer_group_named_sync();

    if (st.tid < kOutputsPerTile) {
      float sum = partial[st.tid];
#pragma unroll
      for (int source_warp = 1; source_warp < 8;
           ++source_warp) {
        sum +=
            partial[source_warp * kOutputsPerTile + st.tid];
      }
      partial[st.tid] = sum;
    }
    slot_ring::consumer_group_named_sync();

    if (allreduce_enabled(args)) {
      publish_and_reduce_tile(st, args, output_tile, partial);
    } else if (st.tid < kOutputsPerTile) {
      const int output =
          output_tile * kOutputsPerTile + st.tid;
      const float residual =
          args.residual != nullptr
              ? __bfloat162float(args.residual[output])
              : 0.0f;
      args.output[output] =
          __float2bfloat16(partial[st.tid] + residual);
    }
    st.tx = outer_tx;
  }
};

}  // namespace naivert_replica_unproj_o_allreduce
