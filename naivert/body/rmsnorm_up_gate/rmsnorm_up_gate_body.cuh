#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_args.hpp"
#include "naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_traits.cuh"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_replica_rmsnorm_up_gate {

namespace slot_ring = naivert;

struct RMSNormUpGateBody {
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;

  __device__ static void run(const RMSNormUpGateArgs& args, unsigned char* shared) {
    auto st = thread_state(shared);
    bind_shared_layout(st, args, shared);
    initialize_cta(st, args);
    // Keep rows1 validation at its original phase boundaries: hoisting it
    // before input staging regresses SM103a instruction scheduling.
    if (!st.active) return;
    if (!valid_args(args) || st.cta_rank >= args.total_tiles) st.active = false;
    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
#pragma unroll
      for (int page = 0; page < kSlotDepth; ++page) {
        const auto outer_tx = st.tx;
        st.tx.slot = page;
        st.tx.iteration = page;
        const int task = valid_args(args) && st.cta_rank < args.total_tiles && page < kPages
                             ? page : -1;
        helper_stage(st, args, task, true);
        st.tx.slot = outer_tx.slot;
        st.tx.iteration = outer_tx.iteration;
      }
    }
    if (is_consumer(st)) stage_input(st, args);
    consumer_sync(st);
    while (true) {
      if (is_consumer(st)) {
        if (!valid_args(args) || st.cta_rank >= args.total_tiles || st.tx.iteration != 0) break;
        internal_consumer_pipeline(st, args);
      } else {
        auto refill = st;
        Slot::prepare_same_slot_reuse(refill, kSlotDepth);
        if (!valid_args(args) || st.cta_rank >= args.total_tiles || refill.tx.iteration >= kPages) break;
        helper_stage(refill, args, refill.tx.iteration, true);
      }
      Slot::advance(st);
    }
  }

 protected:
  __device__ static void bind_shared_layout(slot_ring::SlotRingState& st,
                                            const RMSNormUpGateArgs&,
                                            unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kBarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup1Offset);
    st.early_base = base + kMetadataOffset;
    st.deep_base = base + kPaneBaseOffset;
    st.scratch_base = base + kScratchOffset;
    st.early_stride = 0;
    st.deep_stride = kPaneStride;
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

  __device__ static void initialize_cta(slot_ring::SlotRingState& st, const RMSNormUpGateArgs& args) {
    int active = args.active_ctas > 0 ? args.active_ctas : int(gridDim.x);
    if (active > int(gridDim.x)) active = int(gridDim.x);
    if (args.total_tiles < active) active = args.total_tiles;
    st.active = st.cta_rank < active;
    if (st.tid < kSlotDepth) Slot::init_transaction_slot(st, st.tid);
    cutlass::arch::fence_view_async_shared();
    __threadfence();
    __syncthreads();
  }

  __device__ static void stage_input(slot_ring::SlotRingState& st,
                                            const RMSNormUpGateArgs& args) {
    if (!is_consumer(st)) {
      return;
    }
    const int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* dst16 = reinterpret_cast<uint4*>(bootstrap_x_ptr(st));
    const auto* src16 = reinterpret_cast<const uint4*>(args.unproj_o);
    for (int idx = consumer_tid; idx < kHidden / 8;
         idx += kConsumerThreads) {
      __pipeline_memcpy_async(dst16 + idx, src16 + idx, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }

  __device__ static void stage_weight_pages(slot_ring::SlotRingState& st,
                                            const RMSNormUpGateArgs& args) {
    // Preserve the four-slot bootstrap and the later same-slot refills.
#pragma unroll
    for (int page = 0; page < kSlotDepth; ++page) {
      st.tx.slot = page;
      st.tx.iteration = page;
      helper_stage(st, args, page < kPages ? page : -1, true);
    }
    st.tx = {0, 0, 0};
    while (true) {
      auto refill = st;
      Slot::prepare_same_slot_reuse(refill, kSlotDepth);
      if (refill.tx.iteration >= kPages) break;
      helper_stage(refill, args, refill.tx.iteration, true);
      Slot::advance(st);
    }
  }

  __device__ static bool valid_args(const RMSNormUpGateArgs& args) {
    return args.workspace_base != nullptr && args.dim == kHidden &&
           args.inter_dim == kInter &&
           args.total_tiles == kInter / kOutputsPerTile &&
           args.active_ctas == args.total_tiles;
  }

  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static __nv_bfloat16* bootstrap_x_ptr(
      const slot_ring::SlotRingState& st) {
    return reinterpret_cast<__nv_bfloat16*>(st.scratch_base);
  }

  __device__ static float* sum_state_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x4000);
  }

  __device__ static __half* activation_half_ptr(
      const slot_ring::SlotRingState& st) {
    return reinterpret_cast<__half*>(st.scratch_base + 0x5000);
  }

  __device__ static float* output_accum_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0xC000);
  }

  __device__ static void consumer_sync(slot_ring::SlotRingState& st) {
    if (is_consumer(st)) slot_ring::consumer_group_named_sync();
  }

  __device__ static void helper_stage(slot_ring::SlotRingState& st,
                                      const RMSNormUpGateArgs& args, int page,
                                      bool wait_for_release) {
    if (st.role != slot_ring::ThreadRole::kHelper || page < 0) {
      return;
    }
    if (wait_for_release) {
      Slot::producer_acquire(st);
    }
    void* dst = st.deep_slot_ptr();
    const void* src =
        args.workspace_base +
        (st.cta_rank * kPages + page) * kPaneVisibleBytes;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (st.elected) {
      Slot::producer_expect_tx(st, std::uint32_t(kPaneVisibleBytes));
      auto* barrier = Slot::slot_barrier(st.full_barriers, st.tx.slot);
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{}, dst, src,
          std::uint32_t(kPaneVisibleBytes), barrier);
    }
#else
    const int helper_tid =
        st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
    auto* dst4 = reinterpret_cast<uint4*>(dst);
    const auto* src4 = reinterpret_cast<const uint4*>(src);
    for (int idx = helper_tid;
         idx < kPaneVisibleBytes / int(sizeof(uint4));
         idx += kHelperThreads) {
      dst4[idx] = src4[idx];
    }
#endif
    Slot::producer_commit(st);
  }

  __device__ static float warp_sum(float v) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, delta);
    }
    return v;
  }

  __device__ static float fast_exp2(float v) {
    float out;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(v));
    return out;
  }

  __device__ static float fast_rsqrt(float v) {
    float out;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(v));
    return out;
  }

  __device__ static void initialize_projection_state(
      slot_ring::SlotRingState& st, const RMSNormUpGateArgs& args) {
    auto* state = sum_state_ptr(st);
    const int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* x_bf16 = bootstrap_x_ptr(st);
    const auto* x_bf16x2 =
        reinterpret_cast<const __nv_bfloat162*>(x_bf16);
    float partial = 0.0f;
    for (int i = consumer_tid; i < kHidden / 2;
         i += kConsumerThreads) {
      const float2 x = __bfloat1622float2(x_bf16x2[i]);
      partial += x.x * x.x + x.y * x.y;
    }
    partial = warp_sum(partial);
    if (st.lane_id == 0) {
      state[2 + st.warp_id] = partial;
    }
    consumer_sync(st);
    if (st.warp_id == 0) {
      float total = st.lane_id < 8 ? state[2 + st.lane_id] : 0.0f;
      total = warp_sum(total);
      if (st.lane_id == 0) {
        state[1] = fast_rsqrt(total / float(kHidden) + naivert::geometry::kNormEps);
      }
    }
    consumer_sync(st);
    auto* x_norm = activation_half_ptr(st);
    const auto* gamma2 = reinterpret_cast<const float2*>(args.post_norm);
    for (int i = consumer_tid; i < kHidden / 2;
         i += kConsumerThreads) {
      const float2 x = __bfloat1622float2(x_bf16x2[i]);
      const float2 gamma = gamma2[i];
      // The reference Type6 kernel rounds the normalized activation through BF16
      // before widening it to F16 for HMMA.  Preserve both boundaries: using
      // the unrounded FP32 value here changes the decode trajectory.
      const __nv_bfloat16 norm0 =
          __float2bfloat16(x.x * gamma.x * state[1]);
      const __nv_bfloat16 norm1 =
          __float2bfloat16(x.y * gamma.y * state[1]);
      x_norm[i * 2 + 0] =
          __float2half(__bfloat162float(norm0));
      x_norm[i * 2 + 1] =
          __float2half(__bfloat162float(norm1));
    }
    consumer_sync(st);
  }

  __device__ static std::uint32_t convert_fp8_pair_as_half2(
      std::uint16_t raw) {
    std::uint32_t out;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                 : "=r"(out)
                 : "h"(raw));
    return out;
  }

  __device__ static void mma_16816_f32_f16(
      std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
      std::uint32_t a3, std::uint32_t b0, std::uint32_t b1,
      float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  __device__ static float projection_scale(const slot_ring::SlotRingState& st,
                                           int branch, int scale_group) {
    const auto* scales = reinterpret_cast<const __nv_bfloat16*>(
        st.deep_slot_ptr() + kScaleSidebandOffset +
        branch * kScaleBranchBytes);
    return __bfloat162float(scales[scale_group]);
  }

  __device__ static void accumulate_projection_page_native_hmma(
      slot_ring::SlotRingState& st, int page, float& gate0, float& gate1,
      float& up0, float& up1) {
    if (st.warp_id > 7) {
      return;
    }
    const auto* pane = st.deep_slot_ptr();
    const auto* x_half_bytes = reinterpret_cast<const unsigned char*>(
        activation_half_ptr(st));
    const int warp = st.warp_id;
    const int lane = st.lane_id;
    // Exact Type6 fragment addresses:
    //   A: lane*8 + warp*0x100 + scale_group*0x800
    //   B: lane*4 + warp*0x20 + scale_group*0x100
    // Both gate and up are evaluated by every consumer warp.  The old body
    // split four warps per branch and fed a direct FP8 MMA, which did not
    // reproduce the captured native pane/fragment contract.
    const int a_lane_offset = warp * 0x100 + lane * 8;
    const int b_lane_offset =
        page * 0x800 + warp * 0x20 + lane * 4;
#pragma unroll
    for (int scale_group = 0; scale_group < kScaleValuesPerPage;
         ++scale_group) {
      const auto gate_raw = *reinterpret_cast<const uint2*>(
          pane + scale_group * 0x800 + a_lane_offset);
      const auto up_raw = *reinterpret_cast<const uint2*>(
          pane + kBranchBytes + scale_group * 0x800 +
          a_lane_offset);
      const auto* b_ptr =
          x_half_bytes + b_lane_offset + scale_group * 0x100;
      const std::uint32_t b0 =
          *reinterpret_cast<const std::uint32_t*>(b_ptr);
      const std::uint32_t b1 =
          *reinterpret_cast<const std::uint32_t*>(b_ptr + 0x10);

      float gd0 = 0.0f;
      float gd1 = 0.0f;
      float gd2 = 0.0f;
      float gd3 = 0.0f;
      mma_16816_f32_f16(
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.x)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.x >> 16)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.y)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.y >> 16)),
          b0, b1, gd0, gd1, gd2, gd3);

      float ud0 = 0.0f;
      float ud1 = 0.0f;
      float ud2 = 0.0f;
      float ud3 = 0.0f;
      mma_16816_f32_f16(
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.x)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.x >> 16)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.y)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.y >> 16)),
          b0, b1, ud0, ud1, ud2, ud3);

      const float gate_scale =
          projection_scale(st, 0, scale_group);
      const float up_scale =
          projection_scale(st, 1, scale_group);
      gate0 = fmaf(gd0, gate_scale, gate0);
      gate1 = fmaf(gd2, gate_scale, gate1);
      up0 = fmaf(ud0, up_scale, up0);
      up1 = fmaf(ud2, up_scale, up1);
    }
  }

  __device__ static void internal_consumer_pipeline(
      slot_ring::SlotRingState& st, const RMSNormUpGateArgs& args) {
    initialize_projection_state(st, args);
    slot_ring::SlotRingState page_st = st;
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
#pragma nounroll
    for (int page = 0; page < kPages; ++page) {
      if (page != 0) {
        Slot::advance(page_st);
      }
      Slot::consumer_wait(page_st);
      accumulate_projection_page_native_hmma(
          page_st, page, gate0, gate1, up0, up1);
      Slot::consumer_release(page_st);
    }
    auto* accum = output_accum_ptr(st);
    if (st.warp_id <= 7 && (st.lane_id & 3) == 0) {
      const int row = st.lane_id >> 2;
      const int gate_base = st.warp_id * (2 * kOutputsPerTile);
      accum[gate_base + row] = gate0;
      accum[gate_base + row + 8] = gate1;
      accum[gate_base + kOutputsPerTile + row] = up0;
      accum[gate_base + kOutputsPerTile + row + 8] = up1;
    }
    consumer_sync(st);
    auto* dst = args.up_gate + st.cta_rank * kOutputsPerTile;
    if (st.warp_id == 0 && st.lane_id < kOutputsPerTile) {
      float gate = 0.0f;
      float up = 0.0f;
#pragma unroll
      for (int warp = 0; warp < 8; ++warp) {
        const int base = warp * (2 * kOutputsPerTile);
        gate += accum[base + st.lane_id];
        up += accum[base + kOutputsPerTile + st.lane_id];
      }
      const float sig =
          __fdividef(1.0f,
                     1.0f + fast_exp2(-gate * 1.4426950408889634f));
      dst[st.lane_id] = __float2bfloat16(gate * sig * up);
    }
  }
};

}  // namespace naivert_replica_rmsnorm_up_gate
