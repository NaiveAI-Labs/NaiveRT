#pragma once

// Eight-row dense down projection + TP8 all-reduce for verify-8.  One output tile per
// CTA; the eight rows fill all eight N columns of the m16n8k16. Every warp publishes and reduces one row's tile through the mailbox (four
// slabs per row), and the epilogue writes 8 x kOutputsPerTile outputs.
#include "naivert/body/down_allreduce/down_allreduce_body.cuh"
#include "naivert/pdl/grid_dependency.cuh"

namespace naivert_replica_down_allreduce {

template <bool kWaitForPrimary>
struct DownAllreduceRows8BodyT;
using DownAllreduceRows8Body = DownAllreduceRows8BodyT<false>;
using DownAllreduceRows8PdlBody = DownAllreduceRows8BodyT<true>;


template <bool kWaitForPrimary>
struct DownAllreduceRows8BodyT : DownAllreduceBody {
  static constexpr int kRows = kRows8Rows;
  static constexpr int kAllOutputs = kRows * kOutputsPerTile;

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
      consume_output_tile_rows8(st, args, st.cta_rank);
    }
  }

  __device__ static void bind_shared_layout(slot_ring::SlotRingState& st,
                                            const DownAllreduceArgs&,
                                            unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kRows8BarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kRows8BarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kRows8BarrierGroup1Offset);
    st.early_base = base + kRows8PayloadBase;
    st.deep_base = base + kRows8PayloadBase;
    st.scratch_base = base;
    st.early_stride = kPayloadStride;
    st.deep_stride = kPayloadStride;
  }

 private:
  __device__ static void sync_rows8(slot_ring::SlotRingState& st) {
    if (is_consumer(st)) slot_ring::consumer_group_named_sync();
  }

  __device__ static unsigned char* staged_payload(const slot_ring::SlotRingState& st, int row) {
    return st.scratch_base + kRows8StagedPayloadOffset + row * kRowEpilogueStride;
  }

  __device__ static unsigned char* staged_residual(const slot_ring::SlotRingState& st, int row) {
    return st.scratch_base + kRows8StagedResidualOffset + row * kRowEpilogueStride;
  }

  // Lane (group = lane >> 2, tid4 = lane & 3) accumulates rows 2 * tid4 (even) and
  // 2 * tid4 + 1 (odd) for weight rows group + 8 m of the tile.
  __device__ __forceinline__ static void accumulate_input_slot_rows8(
      const DownAllreduceArgs& args, int output_tile, int input_slot,
      const unsigned char* pane, const __nv_bfloat16* all_rows, int warp,
      int group, int tid4, float (&even)[8], float (&odd)[8]) {
    const int k_base = warp * 32;
    const int lane_k4 = k_base + tid4 * 4;
    const auto* vec = all_rows + group * kLocalInter + input_slot * kInter;
    const std::uint32_t b_low01 = load_bf16_pair(vec + lane_k4);
    const std::uint32_t b_low23 = load_bf16_pair(vec + lane_k4 + 16);
    const std::uint32_t b_high01 = load_bf16_pair(vec + lane_k4 + 2);
    const std::uint32_t b_high23 = load_bf16_pair(vec + lane_k4 + 18);

    float d[4][4] = {};
#pragma unroll
    for (int m_tile = 0; m_tile < kFullMTiles; ++m_tile) {
      const int weight_row = m_tile * 16 + group;
      const auto packed = *reinterpret_cast<const uint4*>(
          pane + pane_storage_offset(weight_row, lane_k4));
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w)),
          b_low01, b_low23, d[m_tile][0], d[m_tile][1], d[m_tile][2], d[m_tile][3]);
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w >> 16)),
          b_high01, b_high23, d[m_tile][0], d[m_tile][1], d[m_tile][2], d[m_tile][3]);
    }
    if constexpr (kTailRows == 8) {
      const int tail_row = kFullMTiles * 16 + group;
      mma_16816_f32_bf16(
          load_shared_fp8_pair_as_bf16x2(pane + pane_storage_offset(tail_row, lane_k4)), 0u,
          load_shared_fp8_pair_as_bf16x2(pane + pane_storage_offset(tail_row, lane_k4 + 16)), 0u,
          b_low01, b_low23, d[3][0], d[3][1], d[3][2], d[3][3]);
      mma_16816_f32_bf16(
          load_shared_fp8_pair_as_bf16x2(pane + pane_storage_offset(tail_row, lane_k4 + 2)), 0u,
          load_shared_fp8_pair_as_bf16x2(pane + pane_storage_offset(tail_row, lane_k4 + 18)), 0u,
          b_high01, b_high23, d[3][0], d[3][1], d[3][2], d[3][3]);
    }

    const int output_base = output_tile * kOutputsPerTile;
    const int k_group = warp >> 2;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (m < kFullMTiles || (kTailRows == 8 && m == 3)) {
        const float scale_lo =
            load_staged_scale(pane, output_tile, output_base + group + 16 * m, k_group);
        even[2 * m] += d[m][0] * scale_lo;
        odd[2 * m] += d[m][1] * scale_lo;
        if (m < kFullMTiles) {
          const float scale_hi =
              load_staged_scale(pane, output_tile, output_base + group + 16 * m + 8, k_group);
          even[2 * m + 1] += d[m][2] * scale_hi;
          odd[2 * m + 1] += d[m][3] * scale_hi;
        }
      }
    }
  }

  __device__ static void issue_residual_rows8(slot_ring::SlotRingState& st,
                                              const DownAllreduceArgs& args,
                                              int output_tile) {
    constexpr int kResidualVectorCount =
        kOutputsPerTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
    constexpr int kAllResidualVectors = kRows * kResidualVectorCount;
    if (st.tid < kAllResidualVectors) {
      const int row = st.tid / kResidualVectorCount;
      const int vector = st.tid - row * kResidualVectorCount;
      auto* shared_residual = reinterpret_cast<uint4*>(staged_residual(st, row));
      if (args.residual != nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const auto smem = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(shared_residual + vector));
        const auto gmem = reinterpret_cast<std::uint64_t>(
            args.residual + row * args.hidden + output_tile * kOutputsPerTile +
            vector * int(sizeof(uint4) / sizeof(__nv_bfloat16)));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :
                     : "r"(smem), "l"(gmem)
                     : "memory");
#else
        shared_residual[vector] = reinterpret_cast<const uint4*>(
            args.residual + row * args.hidden + output_tile * kOutputsPerTile)[vector];
#endif
      } else {
        shared_residual[vector] = uint4{0u, 0u, 0u, 0u};
      }
    }
  }

  __device__ static void wait_residual_rows8(slot_ring::SlotRingState& st) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int kResidualVectorCount =
        kOutputsPerTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
    if (st.tid < kRows * kResidualVectorCount) {
      asm volatile("cp.async.wait_all;" : : : "memory");
    }
#endif
    sync_rows8(st);
  }

  // Warp w publishes and reduces row w's tile.
  __device__ static void publish_and_reduce_rows8(slot_ring::SlotRingState& st,
                                                  const DownAllreduceArgs& args,
                                                  int task, float* partial) {
    const int worker_row = st.warp_id;
    const int lane = st.lane_id;
    const int output_tile_base = task * kOutputsPerTile;
    if (worker_row < kRows) {
      auto* staged = reinterpret_cast<std::uint32_t*>(staged_payload(st, worker_row));
      const auto* local_delta = partial + worker_row * 8 * kPartialRows;
      if (lane < kMailboxQwordsPerTile) {
        const int elem = lane * 2;
        staged[lane] = naivert::pack_bf16x2(local_delta[elem], local_delta[elem + 1]);
      }
      __syncwarp();
      const auto mailbox = tile_mailbox(args, step_gen(args), worker_row);
      if (lane < kMailboxQwordsPerTile) {
        mailbox.publish(task, lane, staged[lane]);
      }
      __syncwarp();
      constexpr int kQwordPairs = kMailboxQwordsPerTile / 2;
      if (lane < kQwordPairs) {
        const int qword = lane * 2;
        // Order pinned by the golden: local first, then the peer lanes.
        naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::of(staged[qword], staged[qword + 1]);
        naivert::accumulate_peers_in_slot_order(acc, [&](int source_slot) {
          return mailbox.wait_pair<false>(source_slot, task, qword);
        });
        *reinterpret_cast<unsigned long long*>(staged + qword) = acc.packed();
      }
    }
    sync_rows8(st);

    for (int idx = st.tid; idx < kAllOutputs; idx += kConsumerThreads) {
      const int row = idx / kOutputsPerTile;
      const int local_output = idx - row * kOutputsPerTile;
      const int output = output_tile_base + local_output;
      if (output < logical_hidden(args)) {
        const auto* reduced =
            reinterpret_cast<const __nv_bfloat16*>(staged_payload(st, row));
        const auto* residual_row =
            reinterpret_cast<const __nv_bfloat16*>(staged_residual(st, row));
        const float residual = __bfloat162float(residual_row[local_output]);
        auto* result = args.tp_result != nullptr ? args.tp_result : args.exp_out;
        const int row_output = row * args.hidden + output;
        result[row_output] =
            __float2bfloat16(__bfloat162float(reduced[local_output]) + residual);
        if (args.exp_out != result) {
          const auto* local_delta = partial + row * 8 * kPartialRows;
          args.exp_out[row_output] =
              __float2bfloat16(local_delta[local_output] + residual);
        }
      }
    }
  }

  __device__ static void consume_output_tile_rows8(slot_ring::SlotRingState& st,
                                                   const DownAllreduceArgs& args,
                                                   int output_tile) {
    if (!is_consumer(st) || output_tile < 0) {
      return;
    }
    // The weight panes stream regardless; the activations and the residual come from the
    // primary (the rows8 up-gate, itself past the attention tail).
    naivert::pdl::wait_for_primary<kWaitForPrimary>();
    const slot_ring::TransactionState outer_tx = st.tx;
    auto* shared_up_gate = reinterpret_cast<__nv_bfloat16*>(st.scratch_base);
    issue_residual_rows8(st, args, output_tile);
    constexpr int kInputVectorCount =
        kRows * kLocalInter * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
    for (int vector = st.tid; vector < kInputVectorCount; vector += kConsumerThreads) {
      reinterpret_cast<uint4*>(shared_up_gate)[vector] =
          reinterpret_cast<const uint4*>(args.up_gate)[vector];
    }
    wait_residual_rows8(st);

    const int warp = st.warp_id;
    const int group = st.lane_id >> 2;
    const int tid4 = st.lane_id & 3;
    float even[8] = {};
    float odd[8] = {};
    for (int input_slot = 0; input_slot < kInputSlots; ++input_slot) {
      set_inner_ring_slot(st, input_slot);
      Slot::consumer_wait(st);
      accumulate_input_slot_rows8(args, output_tile, input_slot, st.deep_slot_ptr(),
                                  shared_up_gate, warp, group, tid4, even, odd);
      Slot::consumer_release(st);
    }

    auto* partial = reinterpret_cast<float*>(st.scratch_base + kRows8PartialOffset);
    {
      const int even_row = tid4 * 2;
      const int odd_row = even_row + 1;
      const int even_base = (even_row * 8 + warp) * kPartialRows;
      const int odd_base = (odd_row * 8 + warp) * kPartialRows;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        if (m < kFullMTiles || (kTailRows == 8 && m == 3)) {
          partial[even_base + group + 16 * m] = even[2 * m];
          partial[odd_base + group + 16 * m] = odd[2 * m];
          if (m < kFullMTiles) {
            partial[even_base + group + 16 * m + 8] = even[2 * m + 1];
            partial[odd_base + group + 16 * m + 8] = odd[2 * m + 1];
          }
        }
      }
    }
    sync_rows8(st);

    for (int idx = st.tid; idx < kAllOutputs; idx += kConsumerThreads) {
      const int row = idx / kOutputsPerTile;
      const int local_output = idx - row * kOutputsPerTile;
      float sum = partial[row * 8 * kPartialRows + local_output];
#pragma unroll
      for (int source_warp = 1; source_warp < 8; ++source_warp) {
        sum += partial[(row * 8 + source_warp) * kPartialRows + local_output];
      }
      partial[row * 8 * kPartialRows + local_output] = sum;
    }
    sync_rows8(st);

    if (allreduce_enabled(args)) {
      publish_and_reduce_rows8(st, args, output_tile, partial);
    } else {
      for (int idx = st.tid; idx < kAllOutputs; idx += kConsumerThreads) {
        const int row = idx / kOutputsPerTile;
        const int local_output = idx - row * kOutputsPerTile;
        const int output = output_tile * kOutputsPerTile + local_output;
        if (output < logical_hidden(args)) {
          const auto* residual_row =
              reinterpret_cast<const __nv_bfloat16*>(staged_residual(st, row));
          const float residual = __bfloat162float(residual_row[local_output]);
          const __nv_bfloat16 value = __float2bfloat16(
              partial[row * 8 * kPartialRows + local_output] + residual);
          const int row_output = row * args.hidden + output;
          args.exp_out[row_output] = value;
          if (args.tp_result != nullptr) {
            args.tp_result[row_output] = value;
          }
        }
      }
    }
    st.tx = outer_tx;
  }
};

}  // namespace naivert_replica_down_allreduce
