#pragma once

// Eight-row dense up-gate for verify-8.  One CTA owns one 16-output tile for all eight
// rows; the four weight pages of the tile stream once through the ring and every page feeds
// the m16n8k16 with all eight N columns populated.  The pane
// ring keeps the rows1 fragment layout and only packs its panes at 0x8100 instead of
// 0x9000 so that the eight fp16 activation rows (64 KiB) fit below the SM100 limit.
#include "naivert/body/rmsnorm_up_gate/rmsnorm_up_gate_body.cuh"
#include "naivert/pdl/grid_dependency.cuh"

namespace naivert_replica_rmsnorm_up_gate {

template <bool kWaitForPrimary>
struct RMSNormUpGateRows8BodyT;
using RMSNormUpGateRows8Body = RMSNormUpGateRows8BodyT<false>;
using RMSNormUpGateRows8PdlBody = RMSNormUpGateRows8BodyT<true>;


template <bool kWaitForPrimary>
struct RMSNormUpGateRows8BodyT : RMSNormUpGateBody {
  static constexpr int kRows = kRows8Rows;
  static constexpr int kConsumerWarps = kConsumerThreads / 32;
  static constexpr int kAccumRowStride = 2 * kOutputsPerTile;   // gate[16] | up[16] per warp

  __device__ static void run(const RMSNormUpGateArgs& args, unsigned char* shared) {
    auto st = thread_state(shared);
    bind_shared_layout(st, args, shared);
    initialize_cta(st, args);
    if (!st.active || !valid_args(args) || !kRows8Fits) return;
    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
      stage_weight_pages(st, args);
    } else {
      stage_input(st, args);
      slot_ring::consumer_group_named_sync();
      pipeline_rows8(st, args);
    }
  }

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
    st.scratch_base = base + kRows8ScratchOffset;
    st.early_stride = 0;
    st.deep_stride = kRows8PaneStride;
  }

  __device__ static void stage_input(slot_ring::SlotRingState& st,
                                            const RMSNormUpGateArgs& args) {
    if (!kRows8Fits || !is_consumer(st)) {
      return;
    }
    // The helpers already stream the weight pages; only the activations depend on the primary.
    naivert::pdl::wait_for_primary<kWaitForPrimary>();
    const int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* dst16 = reinterpret_cast<uint4*>(st.scratch_base);
    const auto* src16 = reinterpret_cast<const uint4*>(args.unproj_o);
    constexpr int kVectors = kRows * kHidden / 8;
    for (int idx = consumer_tid; idx < kVectors; idx += kConsumerThreads) {
      __pipeline_memcpy_async(dst16 + idx, src16 + idx, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }

 private:
  __device__ static void sync_rows8(slot_ring::SlotRingState& st) {
    if (is_consumer(st)) slot_ring::consumer_group_named_sync();
  }

  __device__ static unsigned char* activation_row_ptr(
      const slot_ring::SlotRingState& st, int row) {
    return st.scratch_base + row * kRowActivationBytes;
  }

  __device__ static float* sum_state_ptr_rows8(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.smem_base + kRows8SumStateOffset);
  }

  __device__ static float* output_accum_ptr_rows8(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.smem_base + kRows8OutputAccumOffset);
  }

  // RMSNorm of the eight rows in place (bf16 -> bf16-rounded fp16).
  __device__ static void normalize_rows8(slot_ring::SlotRingState& st,
                                         const RMSNormUpGateArgs& args) {
    auto* state = sum_state_ptr_rows8(st);
    const int consumer_tid = st.warp_id * 32 + st.lane_id;
    const auto* gamma2 = reinterpret_cast<const float2*>(args.post_norm);
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const auto* x_bf16x2 = reinterpret_cast<const __nv_bfloat162*>(
          activation_row_ptr(st, row));
      float partial = 0.0f;
      for (int i = consumer_tid; i < kHidden / 2; i += kConsumerThreads) {
        const float2 x = __bfloat1622float2(x_bf16x2[i]);
        partial += x.x * x.x + x.y * x.y;
      }
      partial = warp_sum(partial);
      if (st.lane_id == 0) {
        state[row * kConsumerWarps + st.warp_id] = partial;
      }
    }
    sync_rows8(st);
    if (st.warp_id == 0) {
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        float total = st.lane_id < kConsumerWarps
                          ? state[row * kConsumerWarps + st.lane_id]
                          : 0.0f;
        total = warp_sum(total);
        if (st.lane_id == 0) {
          state[kRows * kConsumerWarps + row] =
              fast_rsqrt(total / float(kHidden) + naivert::geometry::kNormEps);
        }
      }
    }
    sync_rows8(st);
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      const auto* x_bf16x2 = reinterpret_cast<const __nv_bfloat162*>(
          activation_row_ptr(st, row));
      auto* x_norm = reinterpret_cast<__half*>(activation_row_ptr(st, row));
      const float inv_rms = state[kRows * kConsumerWarps + row];
      for (int i = consumer_tid; i < kHidden / 2; i += kConsumerThreads) {
        const float2 x = __bfloat1622float2(x_bf16x2[i]);
        const float2 gamma = gamma2[i];
        const __nv_bfloat16 norm0 = __float2bfloat16(x.x * gamma.x * inv_rms);
        const __nv_bfloat16 norm1 = __float2bfloat16(x.y * gamma.y * inv_rms);
        x_norm[i * 2 + 0] = __float2half(__bfloat162float(norm0));
        x_norm[i * 2 + 1] = __float2half(__bfloat162float(norm1));
      }
    }
    sync_rows8(st);
  }

  // One 16x1024 gate/up page against all eight rows: lane (g = lane >> 2, q = lane & 3) holds
  // (output g, rows 2q / 2q+1) in d0 / d1 and (output g+8, rows 2q / 2q+1) in d2 / d3.
  __device__ static void accumulate_page_rows8(slot_ring::SlotRingState& st, int page,
                                               float (&gate)[4], float (&up)[4]) {
    if (st.warp_id > 7) {
      return;
    }
    const auto* pane = st.deep_slot_ptr();
    const int warp = st.warp_id;
    const int lane = st.lane_id;
    const int input_row = lane >> 2;
    const int k_pair = lane & 3;
    const int a_lane_offset = warp * 0x100 + lane * 8;
    const auto* x_half =
        reinterpret_cast<const __half*>(activation_row_ptr(st, input_row));
#pragma unroll
    for (int scale_group = 0; scale_group < kScaleValuesPerPage; ++scale_group) {
      const auto gate_raw = *reinterpret_cast<const uint2*>(
          pane + scale_group * 0x800 + a_lane_offset);
      const auto up_raw = *reinterpret_cast<const uint2*>(
          pane + kBranchBytes + scale_group * 0x800 + a_lane_offset);
      const int k_base = page * kPageK + scale_group * 128 + warp * 16;
      const std::uint32_t b0 =
          *reinterpret_cast<const std::uint32_t*>(x_half + k_base + k_pair * 2);
      const std::uint32_t b1 =
          *reinterpret_cast<const std::uint32_t*>(x_half + k_base + k_pair * 2 + 8);

      float gd0 = 0.0f, gd1 = 0.0f, gd2 = 0.0f, gd3 = 0.0f;
      mma_16816_f32_f16(
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.x)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.x >> 16)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.y)),
          convert_fp8_pair_as_half2(std::uint16_t(gate_raw.y >> 16)),
          b0, b1, gd0, gd1, gd2, gd3);
      float ud0 = 0.0f, ud1 = 0.0f, ud2 = 0.0f, ud3 = 0.0f;
      mma_16816_f32_f16(
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.x)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.x >> 16)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.y)),
          convert_fp8_pair_as_half2(std::uint16_t(up_raw.y >> 16)),
          b0, b1, ud0, ud1, ud2, ud3);

      const float gate_scale = projection_scale(st, 0, scale_group);
      const float up_scale = projection_scale(st, 1, scale_group);
      gate[0] = fmaf(gd0, gate_scale, gate[0]);
      gate[1] = fmaf(gd1, gate_scale, gate[1]);
      gate[2] = fmaf(gd2, gate_scale, gate[2]);
      gate[3] = fmaf(gd3, gate_scale, gate[3]);
      up[0] = fmaf(ud0, up_scale, up[0]);
      up[1] = fmaf(ud1, up_scale, up[1]);
      up[2] = fmaf(ud2, up_scale, up[2]);
      up[3] = fmaf(ud3, up_scale, up[3]);
    }
  }

  __device__ static void pipeline_rows8(slot_ring::SlotRingState& st,
                                        const RMSNormUpGateArgs& args) {
    normalize_rows8(st, args);
    slot_ring::SlotRingState page_st = st;
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma nounroll
    for (int page = 0; page < kPages; ++page) {
      if (page != 0) {
        Slot::advance(page_st);
      }
      Slot::consumer_wait(page_st);
      accumulate_page_rows8(page_st, page, gate, up);
      Slot::consumer_release(page_st);
    }

    // accum[row][warp][gate 16 | up 16]; the per-(row, output) sum runs over the warps in
    // order, matching the single-row reduction order.
    auto* accum = output_accum_ptr_rows8(st);
    if (st.warp_id <= 7) {
      const int g = st.lane_id >> 2;
      const int q = st.lane_id & 3;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = 2 * q + half;
        float* slot = accum + (row * kConsumerWarps + st.warp_id) * kAccumRowStride;
        slot[g] = gate[half];
        slot[g + 8] = gate[2 + half];
        slot[kOutputsPerTile + g] = up[half];
        slot[kOutputsPerTile + g + 8] = up[2 + half];
      }
    }
    sync_rows8(st);
    const int tid = st.warp_id * 32 + st.lane_id;
    if (tid < kRows * kOutputsPerTile) {
      const int row = tid / kOutputsPerTile;
      const int o = tid - row * kOutputsPerTile;
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
#pragma unroll
      for (int warp = 0; warp < kConsumerWarps; ++warp) {
        const float* slot = accum + (row * kConsumerWarps + warp) * kAccumRowStride;
        gate_sum += slot[o];
        up_sum += slot[kOutputsPerTile + o];
      }
      const float sig =
          __fdividef(1.0f, 1.0f + fast_exp2(-gate_sum * 1.4426950408889634f));
      args.up_gate[row * args.inter_dim + st.cta_rank * kOutputsPerTile + o] =
          __float2bfloat16(gate_sum * sig * up_sum);
    }
  }
};

}  // namespace naivert_replica_rmsnorm_up_gate
