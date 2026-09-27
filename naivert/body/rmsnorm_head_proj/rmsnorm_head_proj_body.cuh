#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "naivert/body/execute_top_p/sampler_mailbox.hpp"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_args.hpp"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_traits.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/common/role_layout.hpp"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_replica_rmsnorm_head_proj {

namespace slot_ring = naivert;

struct RMSNormHeadProjBody;

}  // namespace naivert_replica_rmsnorm_head_proj

namespace naivert_replica_rmsnorm_head_proj {

struct RMSNormHeadProjBody {
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;

  __device__ static slot_ring::SlotRingWorkArgs work_args(
      const RMSNormHeadProjArgs& args) {
    slot_ring::SlotRingWorkArgs progress{};
    progress.total_tiles = args.total_tiles;
    progress.active_ctas = args.active_ctas;
    return progress;
  }

  __device__ static void bind_shared_layout(slot_ring::SlotRingState& st,
                                            const RMSNormHeadProjArgs&,
                                            unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kBarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup1Offset);
    st.early_base = base + kScratchOffset;
    st.deep_base = base + kScratchOffset;
    st.scratch_base = base + kNormalizedOffset;
    st.early_stride = kScratchBytes;
    st.deep_stride = kScratchBytes;
  }

  __device__ static void init_control_slab(slot_ring::SlotRingState& st,
                                           const RMSNormHeadProjArgs&) {
    if (st.tid < kSlotDepth) {
      Slot::init_transaction_slot(st, st.tid);
    }
  }

  __device__ static void dispatch(slot_ring::SlotRingState& st,
                                  const RMSNormHeadProjArgs& args) {
    if (!valid_args(args) || st.cta_rank >= kGridCtas) {
      st.active = false;
    }
  }

  __device__ static int slot_task(slot_ring::SlotRingState& st,
                                  const RMSNormHeadProjArgs& args, int) {
    if (!valid_args(args) || st.cta_rank >= kGridCtas) {
      return -1;
    }
    const int wave = st.tx.iteration / kWeightPages;
    const int page = st.tx.iteration - wave * kWeightPages;
    const int output_tile = st.cta_rank + wave * kGridCtas;
    if (output_tile >= args.total_tiles) {
      return -1;
    }
    if (is_consumer(st)) {
      return page == 0 ? output_tile : -1;
    }
    return output_tile * kWeightPages + page;
  }

  // The kernel's control flow: role split, layout binding, control-slab and
  // barrier init on the control lanes, async-proxy fence + BAR0 (raw
  // bar.sync), dispatch, helper bootstrap of kSlotDepth pages while the
  // consumers normalise the hidden state, the consumer-group sync, then the
  // transport loop (consumers project one output tile per task, helpers
  // refill the page they staged kSlotDepth iterations ago) and the top-1
  // publication in finalize.  kPdl: the consumers wait for the primary grid
  // (the last expert_down) before reading exp_out; the helpers stream weight
  // pages meanwhile.
  template <bool kPdl>
  __device__ static void run(const RMSNormHeadProjArgs& args,
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
    // Publish the barrier init to the async proxy before BAR0, plus the
    // __threadfence the recovered SASS models as its cache-control point.
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
#pragma unroll
      for (int s = 0; s < kSlotDepth; ++s) {
        const int old_slot = st.tx.slot;
        const int old_iteration = st.tx.iteration;
        st.tx.slot = s;
        st.tx.iteration = s;
        helper_stage(st, args, slot_task(st, args, s));
        st.tx.slot = old_slot;
        st.tx.iteration = old_iteration;
      }
    }
    if (is_consumer(st)) {
      if constexpr (kPdl) {
        naivert::pdl::wait_for_primary<true>();
      }
      consumer_bootstrap(st, args);
    }
    consumer_sync(st);  // BAR4: consumers rendezvous, helpers proceed

    while (true) {
      const int slot = st.tx.slot;
      if (is_consumer(st)) {
        const int task = slot_task(st, args, slot);
        if (task < 0) {
          break;
        }
        project_output_tile(st, args, task);
      } else {
        slot_ring::SlotRingState refill_st = st;
        Slot::prepare_same_slot_reuse(refill_st, kSlotDepth);
        const int task = slot_task(refill_st, args, slot);
        if (task < 0) {
          break;
        }
        helper_stage(refill_st, args, task);
      }
      Slot::advance(st);
    }
    finalize(st, args);
  }

  __device__ static void consumer_bootstrap(slot_ring::SlotRingState& st,
                                            const RMSNormHeadProjArgs& args) {
    auto* counters =
        reinterpret_cast<unsigned int*>(args.workspace_base);
    auto* cta_winners = cta_winners_for(st);
    auto* normalized =
        reinterpret_cast<__nv_bfloat16*>(st.scratch_base);
    auto* partial = reinterpret_cast<float*>(
        st.scratch_base + kNormalizedBytes);
    if (is_consumer(st) && st.lane_id == 0) {
      cta_winners[st.warp_id] = 0ull;
    }
    if (st.cta_rank < kRmsBootstrapCtas) {
      const float inv_rms = compute_inv_rms(st, args);
      for (int k = st.cta_rank * kConsumerThreads + st.tid;
           k < kHidden; k += kRmsBootstrapCtas * kConsumerThreads) {
        const float x = __bfloat162float(args.exp_out[k]);
        args.last_hidden_states[k] =
            __float2bfloat16(x * args.final_norm[k] * inv_rms);
      }
      consumer_sync(st);
      if (st.tid == 0) {
        __threadfence();
        atomicAdd(counters, 1u);
      }
    }
    if (st.tid == 0) {
      const auto* ready =
          reinterpret_cast<const volatile unsigned int*>(counters);
      while (*ready != unsigned(kRmsBootstrapCtas)) {
      }
      // Pair the producer-side device fence + atomic completion publish with
      // an acquire fence before this CTA reads the normalized-hidden payload.
      __threadfence();
    }
    consumer_sync(st);
    auto* dst16 = reinterpret_cast<uint4*>(normalized);
    const auto* src16 =
        reinterpret_cast<const uint4*>(args.last_hidden_states);
    for (int idx = st.tid; idx < kNormalizedBytes / int(sizeof(uint4));
         idx += kConsumerThreads) {
      __pipeline_memcpy_async(dst16 + idx, src16 + idx, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }

  __device__ static void finalize(slot_ring::SlotRingState& st,
                                  const RMSNormHeadProjArgs& args) {
    if (!valid_args(args)) {
      return;
    }
    // Consumer warp leaders produce the local winners while helpers are still
    // staging the tail pages.  Join the complete CTA before one thread reduces
    // those eight values and publishes its contribution to the rank winner.
    __syncthreads();
    if (st.tid == 0) {
      auto* counters =
          reinterpret_cast<unsigned int*>(args.workspace_base);
      auto* cta_winners = cta_winners_for(st);
      unsigned long long cta_best = 0ull;
#pragma unroll
      for (int warp = 0; warp < kConsumerThreads / 32; ++warp) {
        const unsigned long long candidate = cta_winners[warp];
        if (candidate > cta_best) {
          cta_best = candidate;
        }
      }
      atomicMax(shard_top1_for(args), cta_best);
      __threadfence();
      if (atomicAdd(counters + 1, 1u) == kGridCtas - 1) {
        // The last CTA sees every preceding local atomicMax after their
        // fence-and-counter publication.  Publish this rank's winner while
        // HeadProj is still executing, so the later sampler graph need not
        // issue any remote P2P load or atomic.
        publish_top1_inbox(args, counters);
        // The inbox now owns the completed payload. Prepare the rank-local
        // reduction word for the next graph replay here, after every CTA has
        // joined the final counter, instead of making 147 CTAs spin behind a
        // bootstrap-time reset.
        *shard_top1_for(args) = 0ull;
        counters[0] = 0u;
        counters[1] = 0u;
        __threadfence();
      }
    }
  }

 private:
  __device__ static bool valid_args(const RMSNormHeadProjArgs& args) {
    return args.exp_out != nullptr && args.final_norm != nullptr &&
           args.lm_head != nullptr && args.logits_out != nullptr &&
           args.last_hidden_states != nullptr &&
           args.workspace_base != nullptr &&
           args.dim == kHidden && args.vocab == kVocabShard &&
           args.total_tiles == kOutputTiles &&
           args.active_ctas == kGridCtas;
  }

  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static void consumer_sync(slot_ring::SlotRingState& st) {
    if (is_consumer(st)) {
      slot_ring::consumer_group_named_sync();  // BAR4
    }
  }

  __device__ static unsigned long long* cta_winners_for(
      const slot_ring::SlotRingState& st) {
    return reinterpret_cast<unsigned long long*>(
        st.smem_base + kMetadataOffset);
  }

  __device__ static unsigned long long* shard_top1_for(
      const RMSNormHeadProjArgs& args) {
    // The first two 32-bit words retain the established normalization and
    // final-CTA counters.  The following aligned 64-bit word is a stable
    // per-rank handoff consumed by the later sampler graph.
    return reinterpret_cast<unsigned long long*>(
        args.workspace_base + sizeof(unsigned int) * 2);
  }

  // workspace[0..1] are the established normalization/final-CTA counters;
  // bytes 8..15 hold the sampler handoff. Keep the monotonic inbox generation
  // after that 64-bit payload.
  static constexpr int kTop1InboxGenerationCounter = 5;

  __device__ static void publish_top1_inbox(
      const RMSNormHeadProjArgs& args, unsigned int* counters) {
    using SamplerPeerState =
        naivert_replica_execute_top_p::SamplerPeerState;
    constexpr int kTpRanks =
        naivert_replica_execute_top_p::kSamplerTpRanks;
    if (args.tp_rank < 0 || args.tp_rank >= kTpRanks) {
      return;
    }
    for (int destination = 0; destination < kTpRanks; ++destination) {
      if (args.sampler_inbox_peers[destination] == nullptr) {
        return;
      }
    }

    // This counter intentionally remains monotonic across graph replays:
    // the sampler rejects an observed future generation, making stale or
    // cross-slot payload acceptance fail closed.
    unsigned generation =
        atomicAdd(counters + kTop1InboxGenerationCounter, 1u) + 1u;
    // Wraparound is practically unreachable for a request, but retain a
    // non-zero generation so the consumer produces an ordinary fail-closed
    // token instead of escalating a recoverable protocol error to Xid 43.
    if (generation == 0u) {
      generation = 1u;
    }
    const unsigned long long packed = *shard_top1_for(args);

    for (int destination = 0; destination < kTpRanks; ++destination) {
      auto* inbox = reinterpret_cast<SamplerPeerState*>(
          const_cast<std::uint8_t*>(
              args.sampler_inbox_peers[destination]));
      inbox->top1_inbox_packed[args.tp_rank] = packed;
    }
    __threadfence_system();
#pragma unroll
    for (int destination = 0; destination < kTpRanks; ++destination) {
      auto* inbox = reinterpret_cast<SamplerPeerState*>(
          const_cast<std::uint8_t*>(
              args.sampler_inbox_peers[destination]));
      atomicExch_system(
          &inbox->top1_inbox_generation[args.tp_rank], generation);
    }
  }

  __device__ static unsigned ordered_float_key(float value) {
    const unsigned bits = __float_as_uint(value);
    const unsigned flip =
        (static_cast<unsigned>(static_cast<int>(bits) >> 31) |
         0x80000000u);
    return bits ^ flip;
  }

  __device__ static unsigned long long packed_candidate(float score,
                                                         int local_index) {
    return (static_cast<unsigned long long>(ordered_float_key(score)) << 32) |
           static_cast<unsigned>(0xffffffffu -
                                 static_cast<unsigned>(local_index));
  }

  __device__ static float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
  }

  __device__ static float fast_rsqrt(float value) {
    float out;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
  }

  __device__ static float compute_inv_rms(
      slot_ring::SlotRingState& st, const RMSNormHeadProjArgs& args) {
    float partial_sum = 0.0f;
    for (int k = st.tid; k < kHidden; k += kConsumerThreads) {
      const float x = __bfloat162float(args.exp_out[k]);
      partial_sum = fmaf(x, x, partial_sum);
    }
    partial_sum = warp_sum(partial_sum);
    auto* partial = reinterpret_cast<float*>(
        st.scratch_base + kNormalizedBytes);
    if (st.lane_id == 0) {
      partial[st.warp_id] = partial_sum;
    }
    consumer_sync(st);
    if (st.tid == 0) {
      float total = 0.0f;
#pragma unroll
      for (int warp = 0; warp < kConsumerThreads / 32; ++warp) {
        total += partial[warp];
      }
      partial[0] =
          fast_rsqrt(total / static_cast<float>(kHidden) + naivert::geometry::kNormEps);
    }
    consumer_sync(st);
    return partial[0];
  }

  __device__ static void helper_stage(slot_ring::SlotRingState& st,
                                      const RMSNormHeadProjArgs& args,
                                      int task) {
    if (st.role != slot_ring::ThreadRole::kHelper || task < 0) {
      return;
    }
    Slot::producer_acquire(st);
    const int helper_tid =
        st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
    auto* pane = st.deep_slot_ptr();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (st.elected) {
      Slot::producer_expect_tx(st, std::uint32_t(kWeightPaneBytes));
      const auto* src =
          reinterpret_cast<const unsigned char*>(
              args.lm_head + task * kOutputsPerTile * kWeightPage);
      auto* dst = pane + kWeightPaneOffset;
      cp_async_bulk_evict_first(
          dst, src, std::uint32_t(kWeightPaneBytes),
          Slot::slot_barrier(st.full_barriers, st.tx.slot));
    }
#else
    auto* dst4 = reinterpret_cast<uint4*>(pane + kWeightPaneOffset);
    const auto* src4 = reinterpret_cast<const uint4*>(
        args.lm_head + task * kOutputsPerTile * kWeightPage);
    for (int i = helper_tid; i < kWeightPaneBytes / int(sizeof(uint4));
         i += kHelperThreads) {
      dst4[i] = src4[i];
    }
#endif
    Slot::producer_commit(st);
  }

  __device__ static __forceinline__ void
  cp_async_bulk_evict_first(
      void* destination, const void* source, std::uint32_t bytes,
      std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t shared_destination =
        static_cast<std::uint32_t>(
            __cvta_generic_to_shared(destination));
    const std::uint64_t global_source =
        __cvta_generic_to_global(source);
    const std::uint32_t shared_barrier =
        static_cast<std::uint32_t>(
            __cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n"
        "  .reg .b64 policy;\n"
        "  createpolicy.fractional.L2::evict_first.b64 "
        "policy, 1.0;\n"
        "  cp.async.bulk.shared::cta.global."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], policy;\n"
        "}\n"
        :
        : "r"(shared_destination), "l"(global_source),
          "r"(bytes), "r"(shared_barrier)
        : "memory");
#else
    (void)destination;
    (void)source;
    (void)bytes;
    (void)barrier;
#endif
  }

  __device__ static void accumulate_page(
      slot_ring::SlotRingState& st, int page, float (&acc0)[8],
      float (&acc1)[8]) {
    const int lane = st.lane_id;
    const int warp = st.warp_id;
    const int k_base = page * kWeightPage;
    const auto* normalized =
        reinterpret_cast<const __nv_bfloat16*>(st.scratch_base);
    const auto* weight_page = reinterpret_cast<const __nv_bfloat16*>(
        st.deep_slot_ptr() + kWeightPaneOffset);
    const auto* w0 = weight_page + warp * kWeightPage;
    const auto* w1 = weight_page + (warp + 8) * kWeightPage;
    const int kk0 = lane * 2;
#pragma unroll
    for (int step = 0; step < kWeightPage / 64; ++step) {
      const int kk = kk0 + step * 64;
      const int k = k_base + kk;
      const __nv_bfloat162 x =
          *reinterpret_cast<const __nv_bfloat162*>(normalized + k);
      const __nv_bfloat162 weight0 =
          *reinterpret_cast<const __nv_bfloat162*>(w0 + kk);
      const __nv_bfloat162 weight1 =
          *reinterpret_cast<const __nv_bfloat162*>(w1 + kk);
      const float2 x_float = __bfloat1622float2(x);
      const float2 weight0_float = __bfloat1622float2(weight0);
      const float2 weight1_float = __bfloat1622float2(weight1);
      const int bucket = step & 7;
      acc0[bucket] =
          fmaf(x_float.x, weight0_float.x, acc0[bucket]);
      acc0[bucket] =
          fmaf(x_float.y, weight0_float.y, acc0[bucket]);
      acc1[bucket] =
          fmaf(x_float.x, weight1_float.x, acc1[bucket]);
      acc1[bucket] =
          fmaf(x_float.y, weight1_float.y, acc1[bucket]);
    }
  }

  __device__ static void project_output_tile(
      slot_ring::SlotRingState& st, const RMSNormHeadProjArgs& args,
      int output_tile) {
    slot_ring::SlotRingState page_st = st;
    float acc0[8];
    float acc1[8];
#pragma unroll
    for (int bucket = 0; bucket < 8; ++bucket) {
      acc0[bucket] = 0.0f;
      acc1[bucket] = 0.0f;
    }
#pragma unroll
    for (int page = 0; page < kWeightPages; ++page) {
      if (page != 0) {
        Slot::advance(page_st);
      }
      Slot::consumer_wait(page_st);
      accumulate_page(page_st, page, acc0, acc1);
      Slot::consumer_release(page_st);
    }
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#pragma unroll
    for (int bucket = 0; bucket < 8; ++bucket) {
      sum0 += acc0[bucket];
      sum1 += acc1[bucket];
    }
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (st.lane_id == 0) {
      const int out0 = output_tile * kOutputsPerTile + st.warp_id;
      args.logits_out[out0] = sum0;
      args.logits_out[out0 + 8] = sum1;
      auto* cta_winners = cta_winners_for(st);
      const unsigned long long candidate0 = packed_candidate(sum0, out0);
      const unsigned long long candidate1 =
          packed_candidate(sum1, out0 + 8);
      const unsigned long long previous = cta_winners[st.warp_id];
      cta_winners[st.warp_id] =
          candidate0 > previous
              ? (candidate1 > candidate0 ? candidate1 : candidate0)
              : (candidate1 > previous ? candidate1 : previous);
    }
    st.tx = page_st.tx;
  }
};

}  // namespace naivert_replica_rmsnorm_head_proj
