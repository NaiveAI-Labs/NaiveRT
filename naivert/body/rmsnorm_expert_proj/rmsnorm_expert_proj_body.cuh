#pragma once
#include "naivert/common/load_fp32x8.cuh"

#include "naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_args.hpp"
#include "naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_traits.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/common/role_layout.hpp"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

#include "naivert/common/prepared_mlp_input_layout.hpp"
#include "naivert/common/block128_activation.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#include <cuda/ptx>
#endif

namespace naivert_replica_rmsnorm_expert_proj {

namespace slot_ring = naivert;

struct RMSNormExpertProjBody;

}  // namespace naivert_replica_rmsnorm_expert_proj

namespace naivert_replica_rmsnorm_expert_proj {

__device__ static __forceinline__ void
cp_async_bulk_weight_evict_first(
    void* destination, const void* source, std::uint32_t bytes,
    std::uint64_t* barrier) {
  const std::uint32_t destination_shared =
      std::uint32_t(__cvta_generic_to_shared(destination));
  const std::uint32_t barrier_shared =
      std::uint32_t(__cvta_generic_to_shared(barrier));
  std::uint64_t cache_policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first."
      "L2::evict_unchanged.b64 %0, 1.0;"
      : "=l"(cache_policy));
  asm volatile(
      "cp.async.bulk.shared::cta.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :
      : "r"(destination_shared), "l"(source), "r"(bytes),
        "r"(barrier_shared), "l"(cache_policy)
      : "memory");
}

struct RMSNormExpertProjBody {
  using Slot = slot_ring::SlotProtocol<kRmsSlotDepth>;

  __device__ static slot_ring::SlotRingWorkArgs work_args(
      const RMSNormExpertProjArgs&) {
    slot_ring::SlotRingWorkArgs progress{};
    progress.total_tiles = kRmsTaskCtas;
    progress.active_ctas = kRmsTaskCtas;
    return progress;
  }

  __device__ static void bind_shared_layout(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs&,
      unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kRmsBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kRmsBarrierGroup1Offset);
    st.early_base = base + kRmsScratchBase;
    st.deep_base = base + kRmsScratchBase;
    st.scratch_base = base;
    st.early_stride = kRmsScratchStride;
    st.deep_stride = kRmsScratchStride;
  }

  __device__ static void init_control_slab(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs&) {
    if (st.tid < kRmsSlotDepth) {
      Slot::init_transaction_slot(st, st.tid);
    }
  }

  __device__ static int slot_task(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs&, int) {
    if (st.role == slot_ring::ThreadRole::kHelper) {
      if (st.tx.iteration < kRmsExpertsPerTask) {
        return st.cta_rank * kRmsExpertsPerTask + st.tx.iteration;
      }
      return -1;
    }
    return st.tx.iteration == 0 ? st.cta_rank : -1;
  }

  // The kernel's control flow: role split, layout binding, slot init on the
  // control lanes, async-proxy fence (no __threadfence: the native sequence
  // has none) and raw BAR0, helper bootstrap of both fixed-shape rows (no
  // refill exists), the consumer bootstrap (normalised hidden) and BAR4, then
  // one paired-slot task per consumer: wait for both panes, the split-group
  // router GEMV, release both, and the score epilogue in finalize.
  __device__ static void run(const RMSNormExpertProjArgs& args,
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
#if defined(__CUDA_ARCH__)
    asm volatile("bar.sync 0;" ::: "memory");  // BAR0
#endif
    if (!st.active) {
      return;
    }

    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
#pragma unroll
      for (int s = 0; s < kRmsSlotDepth; ++s) {
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
      consumer_bootstrap(st, args);
    }
    if (is_consumer(st)) {
      slot_ring::consumer_group_named_sync();  // BAR4: consumers rendezvous, helpers proceed
    }

    // One paired-slot task per consumer iteration; the helpers have nothing
    // to refill and leave the loop at once.
    while (true) {
      const int slot = st.tx.slot;
      if (is_consumer(st)) {
        const int task = slot_task(st, args, slot);
        if (task < 0) {
          break;
        }
        Slot::consumer_wait(st);
        slot_ring::SlotRingState secondary_st = st;
        Slot::advance(secondary_st);
        Slot::consumer_wait(secondary_st);
        if (is_consumer(st) && task >= 0) {
          // The two panes are contiguous in the fixed native layout, so the
          // GEMV addresses them from the primary state alone.
          router_gemv_split_groups(st, task);
          slot_ring::consumer_group_named_sync();
        }
        Slot::consumer_release(st);
        Slot::consumer_release(secondary_st);
      } else {
        break;
      }
      Slot::advance(st);
      Slot::advance(st);
    }
    finalize(st, args);
  }

  __device__ static void consumer_bootstrap(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs& args) {
    if (st.tid == 0) {
      auto* score_tasks = reinterpret_cast<int*>(
          st.scratch_base + kRmsScoreTaskOffset);
#pragma unroll
      for (int i = 0; i < kRmsMaxTasksPerCta; ++i) {
        score_tasks[i] = -1;
      }
    }
    const auto* hidden4 =
        reinterpret_cast<const uint4*>(args.hidden);
    compute_normalized(st, args, hidden4);
  }

  __device__ static void finalize(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs& args) {
    if (!is_consumer(st)) {
      return;
    }

    // Publish the per-iteration score scratch before the elected consumer
    // stores it.  This is the kernel epilogue; no alternate task loop or
    // transport path is hidden in finalize.
    slot_ring::consumer_group_named_sync();
    if (st.tid == 0) {
      const auto* score_tasks = reinterpret_cast<const int*>(
          st.scratch_base + kRmsScoreTaskOffset);
      const auto* score_values = reinterpret_cast<const float*>(
          st.scratch_base + kRmsScoreValueOffset);
#pragma unroll
      for (int i = 0; i < kRmsMaxTasksPerCta; ++i) {
        const int task = score_tasks[i];
        if (task >= 0) {
          const int output = task * kRmsExpertsPerTask;
#pragma unroll
          for (int expert = 0; expert < kRmsExpertsPerTask; ++expert) {
            args.scores[output + expert] =
                score_values[i * kRmsExpertsPerTask + expert];
          }
        }
      }
    }

    // X_MLP_IN is token-level output, so only the CTA owning task zero
    // publishes the cached normalized vector.
    if (st.cta_rank != 0 || args.x_mlp == nullptr) {
      return;
    }
    constexpr int vector_count = kRmsHiddenDim / 8;
    const auto* x_norm4 = reinterpret_cast<const uint4*>(
        st.scratch_base + kRmsXNormPaneOffset);
    auto* x_mlp4 = reinterpret_cast<uint4*>(args.x_mlp);
    for (int chunk = st.tid; chunk < vector_count;
         chunk += kRmsConsumerThreads) {
      x_mlp4[chunk] = x_norm4[chunk];
    }
  }

 private:
  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static float warp_sum(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(kMask, value, offset);
    }
    return value;
  }

  __device__ static float consumer_sum(
      slot_ring::SlotRingState& st, float value) {
    auto* partial =
        reinterpret_cast<float*>(
            st.scratch_base + kRmsRmsPartialOffset);
    value = warp_sum(value);
    if (st.lane_id == 0) {
      partial[st.warp_id] = value;
    }
    slot_ring::consumer_group_named_sync();
    float combined = 0.0f;
#pragma unroll
    for (int warp = 0; warp < kRmsConsumerThreads / 32; ++warp) {
      combined += partial[warp];
    }
    return combined;
  }

  __device__ static void consumer_subgroup_sync(int group) {
#if defined(__CUDA_ARCH__)
    const int barrier = 8 + group;
    naivert::named_barrier_sync(128u, unsigned(barrier));
#else
    (void)group;
#endif
  }

  __device__ static void consumer_group_score(
      slot_ring::SlotRingState& st, float value, int group, int task) {
    auto* partial = reinterpret_cast<float*>(
        st.scratch_base + kRmsRouterPartialOffset +
        st.tx.slot * kRmsPartialBytes);
    value = warp_sum(value);
    const int group_tid = st.tid - group * kRmsConsumerGroupThreads;
    const int group_warp = group_tid / 32;
    if (st.lane_id == 0) {
      partial[group * kRmsConsumerGroupWarps + group_warp] = value;
    }
    consumer_subgroup_sync(group);
    float combined = 0.0f;
    if (group_tid == 0) {
#pragma unroll
      for (int warp = 0; warp < kRmsConsumerGroupWarps; ++warp) {
        combined +=
            partial[group * kRmsConsumerGroupWarps + warp];
      }
      auto* score_tasks = reinterpret_cast<int*>(
          st.scratch_base + kRmsScoreTaskOffset);
      auto* score_values = reinterpret_cast<float*>(
          st.scratch_base + kRmsScoreValueOffset);
      if (group == 0) {
        score_tasks[0] = task;
      }
      score_values[group] = combined;
    }
  }

  __device__ static void helper_stage(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs& args,
      int task) {
    if (st.role != slot_ring::ThreadRole::kHelper) {
      return;
    }
    Slot::producer_acquire(st);
    if (task >= 0) {
      constexpr int bytes = kRmsRouterPaneBytes;
      void* dst = st.deep_slot_ptr();
      const void* src =
          args.router_weight + task * kRmsHiddenDim;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      // One elected lane owns both transaction declaration and the complete
      // row copy.  Multiple expect-tx issuers accumulate byte counts on the
      // same mbarrier and were the source of the unsafe earlier experiment.
      if (st.elected) {
        Slot::producer_expect_tx(st, std::uint32_t(bytes));
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
            static_cast<unsigned char*>(dst),
            static_cast<const unsigned char*>(src), std::uint32_t(bytes),
            Slot::slot_barrier(st.full_barriers, st.tx.slot));
      }
#else
      const int helper_tid =
          st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
      auto* out = reinterpret_cast<naivert::RouterWeight*>(dst);
      const auto* in = reinterpret_cast<const naivert::RouterWeight*>(src);
      for (int col = helper_tid; col < kRmsHiddenDim;
           col += kRmsHelperThreads) {
        out[col] = in[col];
      }
#endif
    }
    Slot::producer_commit(st);
  }

  __device__ static void compute_normalized(
      slot_ring::SlotRingState& st, const RMSNormExpertProjArgs& args,
      const uint4* hidden4) {
    constexpr int vector_count = kRmsHiddenDim / 8;
    const float4* gamma4 = reinterpret_cast<const float4*>(args.gamma);
    uint4* x_norm4 =
        reinterpret_cast<uint4*>(st.scratch_base + kRmsXNormPaneOffset);
    constexpr int kMaxChunks =
        (kRmsHiddenDim / 8 + kRmsConsumerThreads - 1) /
        kRmsConsumerThreads;
    uint4 hidden_cache[kMaxChunks];
    float4 gamma0_cache[kMaxChunks];
    float4 gamma1_cache[kMaxChunks];
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      const int chunk = st.tid + i * kRmsConsumerThreads;
      if (chunk < vector_count) {
        hidden_cache[i] = hidden4[chunk];
        gamma0_cache[i] = gamma4[2 * chunk];
        gamma1_cache[i] = gamma4[2 * chunk + 1];
      }
    }

    float square_acc[kMaxChunks];
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      square_acc[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      const int chunk = st.tid + i * kRmsConsumerThreads;
      if (chunk < vector_count) {
        const __nv_bfloat162* values =
            reinterpret_cast<const __nv_bfloat162*>(&hidden_cache[i]);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float2 x = __bfloat1622float2(values[pair]);
          square_acc[i] =
              __fmaf_rn(x.x, x.x, square_acc[i]);
          square_acc[i] =
              __fmaf_rn(x.y, x.y, square_acc[i]);
        }
      }
    }
    float local_square = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      local_square += square_acc[i];
    }
    const float inv_rms = rsqrtf(
        consumer_sum(st, local_square) /
            static_cast<float>(kRmsHiddenDim) +
        naivert::geometry::kNormEps);

#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      const int chunk = st.tid + i * kRmsConsumerThreads;
      if (chunk < vector_count) {
        const __nv_bfloat162* values =
            reinterpret_cast<const __nv_bfloat162*>(&hidden_cache[i]);
        const float4 gamma0 = gamma0_cache[i];
        const float4 gamma1 = gamma1_cache[i];
        const float gamma[8] = {
            gamma0.x, gamma0.y, gamma0.z, gamma0.w,
            gamma1.x, gamma1.y, gamma1.z, gamma1.w};
        uint4 packed;
        __nv_bfloat162* out =
            reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float2 x = __bfloat1622float2(values[pair]);
          out[pair] = __floats2bfloat162_rn(
              x.x * gamma[2 * pair] * inv_rms,
              x.y * gamma[2 * pair + 1] * inv_rms);
        }
        x_norm4[chunk] = packed;
      }
    }
    slot_ring::consumer_group_named_sync();
  }

  __device__ static void router_gemv_split_groups(
      slot_ring::SlotRingState& st, int task) {
    constexpr int vector_count = kRmsHiddenDim / 8;
    const int group =
        st.role == slot_ring::ThreadRole::kConsumerA ? 0 : 1;
    const int group_tid =
        st.tid - group * kRmsConsumerGroupThreads;
    const auto* router4 = reinterpret_cast<const uint4*>(
        st.deep_base + group * st.deep_stride);
    const uint4* x_norm4 = reinterpret_cast<const uint4*>(
        st.scratch_base + kRmsXNormPaneOffset);
    constexpr int kMaxChunks =
        (kRmsHiddenDim / 8 + kRmsConsumerGroupThreads - 1) /
        kRmsConsumerGroupThreads;
    float accumulators[kMaxChunks];
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      accumulators[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      const int chunk =
          group_tid + i * kRmsConsumerGroupThreads;
      if (chunk < vector_count) {
        const uint4 x_vector = x_norm4[chunk];
        const auto* router_values = reinterpret_cast<const naivert::RouterWeight*>(router4) + chunk * 8;
        const __nv_bfloat162* x_values =
            reinterpret_cast<const __nv_bfloat162*>(&x_vector);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float2 x = __bfloat1622float2(x_values[pair]);
          const float2 weight =
              naivert::router_weight_pair(router_values, pair);
          accumulators[i] =
              __fmaf_rn(weight.x, x.x, accumulators[i]);
          accumulators[i] =
              __fmaf_rn(weight.y, x.y, accumulators[i]);
        }
      }
    }
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      dot += accumulators[i];
    }
    consumer_group_score(st, dot, group, task);
  }
};

// Multi-row router: one CTA owns an expert pair across all rows.
// Helper warps stage each router weight once for reuse by the consumer dot products.
template <int kRows_, bool kWideGamma = false, bool kPackedWeights = false>
struct RMSNormExpertProjRowsBody {
  static constexpr int kRows = kRows_;
  static constexpr int kXNormOffset = kRmsRowsXNormOffset;
  static constexpr int kXNormStride = kRmsRowsXNormStride;
  // Prepared rows keep eight-term FP32 dot fragments after the BF16 row image.
  static constexpr int kDotPartialOffset = kRmsHiddenDim * int(sizeof(__nv_bfloat16));
  static constexpr int kRouterOffset = kRows * kXNormStride;
  static constexpr int kRouterStride = kRmsRowsRouterStride;
  static constexpr int kPartialOffset = kRouterOffset + kRmsExpertsPerTask * kRouterStride;
  static constexpr int kBarrierOffset = kPartialOffset;
  static constexpr int kReductionOffset = kPartialOffset + 0x10;
  static_assert(kRows % 2 == 0, "rows are normalized in pairs");
  static_assert(kReductionOffset + (kRows * kRmsConsumerWarps + kRmsExpertsPerTask * kRows * kRmsConsumerGroupWarps) * 4 <= kRmsSharedBytes,
                "row-count reduction state must fit in dynamic shared memory");

  template <bool kWaitForPrimary = false, bool kPrepareFp8 = false>
  __device__ static void run(
      const RMSNormExpertProjArgs& args, unsigned char* smem,
      unsigned char* prepared = nullptr) {
    static_assert(!kPackedWeights || kPrepareFp8);
    const int tid = static_cast<int>(threadIdx.x);
    const int task = static_cast<int>(blockIdx.x);
    if (task >= kRmsTaskCtas) {
      return;
    }
    // Publish initialization before any consumer can poll the transaction
    // barrier. Normalization latency is not an initialization handshake.
    if (tid == kRmsConsumerThreads) {
      slot_ring::Barrier::init(reinterpret_cast<std::uint64_t*>(smem + kBarrierOffset), 1);
    }
    __syncthreads();
    if (tid < kRmsConsumerThreads) {
      compute_normalized_rows<kWaitForPrimary, kPrepareFp8>(args, smem, tid);
      slot_ring::consumer_group_named_sync();
      if constexpr (kPrepareFp8) {
        static_assert(kRows == 8 && naivert::kExpertActivationFp32);
        // Every owner CTA publishes one K128 block for all eight tokens.
        // Normalized BF16 values are acquired above, before any reader uses them.
        if (task < naivert::prepared_mlp::kBlocks) {
          const int token = tid >> 5, lane = tid & 31;
          const auto* source = reinterpret_cast<const __nv_bfloat16*>(
              smem + kXNormOffset + token*kXNormStride) + task*128;
          auto* destination = reinterpret_cast<__nv_fp8_e4m3*>(
              prepared + token*naivert::prepared_mlp::kRowStride) + task*128;
          auto* scales = reinterpret_cast<float*>(prepared + naivert::prepared_mlp::kScaleOffset)
                         + task*naivert::prepared_mlp::kScaleBlockStride + token;
          naivert::quantize_block128_row(source,destination,scales,128,lane,32);
          if (task == naivert::prepared_mlp::kBlocks-1 && lane == 0)
            *reinterpret_cast<uint4*>(prepared + token*naivert::prepared_mlp::kRowStride + 4096)
                = make_uint4(0,0,0,0);
        }
      }
      if constexpr (!kPrepareFp8) wait_router_pair(smem);
      compute_router_scores<kPrepareFp8>(args, smem, task, tid);
    } else {
      stage_router_pair(args, smem, task, tid - kRmsConsumerThreads);
    }
  }

 private:
  __device__ static float fast_rsqrt(float value) {
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value));
    return result;
  }

  __device__ static float warp_sum(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(kMask, value, offset);
    }
    return value;
  }

  __device__ static void consumer_sum_pair(
      unsigned char* smem, int tid, int row,
      float value0, float value1,
      float& combined0, float& combined1) {
    auto* partial = reinterpret_cast<float*>(
        smem + kReductionOffset);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    value0 = warp_sum(value0);
    value1 = warp_sum(value1);
    if (lane == 0) {
      partial[row * kRmsConsumerWarps + warp] = value0;
      partial[(row + 1) * kRmsConsumerWarps + warp] = value1;
    }
    slot_ring::consumer_group_named_sync();
    combined0 = 0.0f;
    combined1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kRmsConsumerWarps; ++i) {
      combined0 += partial[row * kRmsConsumerWarps + i];
      combined1 += partial[(row + 1) * kRmsConsumerWarps + i];
    }
  }

  template <bool kWaitForPrimary, bool kDirectScores = false>
  __device__ static void compute_normalized_rows(
      const RMSNormExpertProjArgs& args, unsigned char* smem, int tid) {
    constexpr int kNormBatch = kRows == 8 ? 4 : 2;
    constexpr int vector_count = kRmsHiddenDim / 8;
    constexpr int kMaxChunks =
        (vector_count + kRmsConsumerThreads - 1) /
        kRmsConsumerThreads;
    const auto* hidden4 =
        reinterpret_cast<const uint4*>(args.hidden);
    const auto* gamma4 =
        reinterpret_cast<const float4*>(args.gamma);

    float4 gamma0_cache[kMaxChunks];
    float4 gamma1_cache[kMaxChunks];
#pragma unroll
    for (int i = 0; i < kMaxChunks; ++i) {
      const int chunk = tid + i * kRmsConsumerThreads;
      if (chunk < vector_count) {
        if constexpr (kWideGamma) {
          naivert::load_fp32x8_aligned32(
              args.gamma + chunk * 8, gamma0_cache[i], gamma1_cache[i]);
        } else {
          gamma0_cache[i] = gamma4[2 * chunk];
          gamma1_cache[i] = gamma4[2 * chunk + 1];
        }
      }
    }
    float2 router0_cache[kMaxChunks][4];
    float2 router1_cache[kMaxChunks][4];
    naivert::pdl::wait_for_primary<kWaitForPrimary>();

#pragma unroll
    for (int row = 0; row < kRows; row += kNormBatch) {
      uint4 hidden_cache[kNormBatch][kMaxChunks];
#pragma unroll
      for (int i = 0; i < kMaxChunks; ++i) {
        const int chunk = tid + i * kRmsConsumerThreads;
        if (chunk < vector_count) {
#pragma unroll
          for (int pair_row = 0; pair_row < kNormBatch; ++pair_row)
            hidden_cache[pair_row][i] = hidden4[(row + pair_row) * vector_count + chunk];
        }
      }

      float square_acc[kNormBatch][kMaxChunks] = {};
#pragma unroll
      for (int pair_row = 0; pair_row < kNormBatch; ++pair_row) {
#pragma unroll
        for (int i = 0; i < kMaxChunks; ++i) {
          const int chunk = tid + i * kRmsConsumerThreads;
          if (chunk < vector_count) {
            const auto* values =
                reinterpret_cast<const __nv_bfloat162*>(
                    &hidden_cache[pair_row][i]);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const float2 x = __bfloat1622float2(values[pair]);
              square_acc[pair_row][i] =
                  __fmaf_rn(
                      x.x, x.x, square_acc[pair_row][i]);
              square_acc[pair_row][i] =
                  __fmaf_rn(
                      x.y, x.y, square_acc[pair_row][i]);
            }
          }
        }
      }
      auto* partial = reinterpret_cast<float*>(smem + kReductionOffset);
      const int warp = tid >> 5, lane = tid & 31;
#pragma unroll
      for (int pair_row = 0; pair_row < kNormBatch; ++pair_row) {
        float local_square = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxChunks; ++i)
          local_square += square_acc[pair_row][i];
        const float value = warp_sum(local_square);
        if (lane == 0)
          partial[(row + pair_row) * kRmsConsumerWarps + warp] = value;
      }
      // Each row has its own partial slots; publish this batch together.
      slot_ring::consumer_group_named_sync();
      // Acquire weights before the first fused dot, after sum work overlaps delivery.
      if constexpr (kDirectScores) {
        if (row == 0) {
          wait_router_pair(smem);
#pragma unroll
          for (int i = 0; i < kMaxChunks; ++i) {
            const int chunk = tid + i * kRmsConsumerThreads;
            if constexpr (kPackedWeights) {
              // The second weight image swaps alternating float4 halves.
              // Each eight-lane LDS.128 subgroup then uses every bank once.
              const int lo = (chunk * 2) ^ ((chunk >> 2) & 1);
              const int hi = lo ^ 1;
              const auto* w0 = reinterpret_cast<const float4*>(smem + kRouterOffset);
              const auto* w1 = reinterpret_cast<const float4*>(smem + kRouterOffset + kRouterStride);
              const float4 a0 = w0[lo], b0 = w0[hi], a1 = w1[lo], b1 = w1[hi];
              router0_cache[i][0] = make_float2(a0.x, a0.y);
              router0_cache[i][1] = make_float2(a0.z, a0.w);
              router0_cache[i][2] = make_float2(b0.x, b0.y);
              router0_cache[i][3] = make_float2(b0.z, b0.w);
              router1_cache[i][0] = make_float2(a1.x, a1.y);
              router1_cache[i][1] = make_float2(a1.z, a1.w);
              router1_cache[i][2] = make_float2(b1.x, b1.y);
              router1_cache[i][3] = make_float2(b1.z, b1.w);
            } else {
              const auto* w0 = reinterpret_cast<const naivert::RouterWeight*>(
                  smem + kRouterOffset) + chunk * 8;
              const auto* w1 = reinterpret_cast<const naivert::RouterWeight*>(
                  smem + kRouterOffset + kRouterStride) + chunk * 8;
  #pragma unroll
              for (int pair = 0; pair < 4; ++pair) {
                router0_cache[i][pair] = naivert::router_weight_pair(w0, pair);
                router1_cache[i][pair] = naivert::router_weight_pair(w1, pair);
              }
            }
          }
        }
      }
      float inv_rms[kNormBatch];
#pragma unroll
      for (int pair_row = 0; pair_row < kNormBatch; ++pair_row) {
        float combined = 0.0f;
#pragma unroll
        for (int i = 0; i < kRmsConsumerWarps; ++i)
          combined += partial[(row + pair_row) * kRmsConsumerWarps + i];
        inv_rms[pair_row] = fast_rsqrt(
            combined / static_cast<float>(kRmsHiddenDim) + naivert::geometry::kNormEps);
      }

#pragma unroll
      for (int pair_row = 0; pair_row < kNormBatch; ++pair_row) {
        auto* x_norm4 = reinterpret_cast<uint4*>(
            smem + kXNormOffset +
            (row + pair_row) * kXNormStride);
#pragma unroll
        for (int i = 0; i < kMaxChunks; ++i) {
          const int chunk = tid + i * kRmsConsumerThreads;
          if (chunk < vector_count) {
            const auto* values =
                reinterpret_cast<const __nv_bfloat162*>(
                    &hidden_cache[pair_row][i]);
            const float4 gamma0 = gamma0_cache[i];
            const float4 gamma1 = gamma1_cache[i];
            const float gamma[8] = {
                gamma0.x, gamma0.y, gamma0.z, gamma0.w,
                gamma1.x, gamma1.y, gamma1.z, gamma1.w};
            float dot0 = 0.0f, dot1 = 0.0f;
            uint4 packed;
            auto* out =
                reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const float2 x = __bfloat1622float2(values[pair]);
              out[pair] = __floats2bfloat162_rn(
                  x.x * gamma[2 * pair] * inv_rms[pair_row],
                  x.y * gamma[2 * pair + 1] *
                      inv_rms[pair_row]);
              if constexpr (kDirectScores) {
                // Decode the rounded BF16 pair exactly as the original GEMV does.
                const float2 rounded = __bfloat1622float2(out[pair]);
                const float2 a0 = router0_cache[i][pair];
                const float2 a1 = router1_cache[i][pair];
                dot0 = __fmaf_rn(a0.x, rounded.x, dot0);
                dot0 = __fmaf_rn(a0.y, rounded.y, dot0);
                dot1 = __fmaf_rn(a1.x, rounded.x, dot1);
                dot1 = __fmaf_rn(a1.y, rounded.y, dot1);
              }
            }
            if (!kDirectScores || (chunk >> 4) == blockIdx.x ||
                (blockIdx.x == 0 && args.x_mlp != nullptr))
              x_norm4[chunk] = packed;
            if constexpr (kDirectScores) {
              static_assert(kDotPartialOffset + 2 * vector_count * 4 <= kXNormStride);
              auto* dots = reinterpret_cast<float*>(
                  smem + kXNormOffset + (row + pair_row) * kXNormStride + kDotPartialOffset);
              dots[chunk] = dot0;
              dots[vector_count + chunk] = dot1;
            }
          }
        }
      }
    }
  }

  __device__ static void stage_router_pair(
      const RMSNormExpertProjArgs& args, unsigned char* smem,
      int task, int helper_tid) {
    if (helper_tid != 0) {
      return;
    }
    auto* router_barrier = reinterpret_cast<std::uint64_t*>(
        smem + kBarrierOffset);
    // Initialization was published by the full CTA before entering this path.
    slot_ring::Barrier::arrive_and_expect_tx(
        router_barrier,
        kRmsExpertsPerTask * kRmsRouterPaneBytes);
#pragma unroll
    for (int local_expert = 0;
         local_expert < kRmsExpertsPerTask; ++local_expert) {
      const int expert =
          task * kRmsExpertsPerTask + local_expert;
      const void* src =
          args.router_weight + expert * kRmsHiddenDim +
          (kPackedWeights ? kRmsExperts * kRmsHiddenDim : 0);
      void* dst =
          smem + kRouterOffset +
          local_expert * kRouterStride;
      cp_async_bulk_weight_evict_first(
          dst, src, std::uint32_t(kRmsRouterPaneBytes),
          router_barrier);
    }
  }

  __device__ static void wait_router_pair(unsigned char* smem) {
    auto* router_barrier = reinterpret_cast<std::uint64_t*>(
        smem + kBarrierOffset);
    while (!slot_ring::Barrier::try_wait(
        router_barrier, 0)) {
    }
  }

  __device__ static void subgroup_sync(int group) {
#if defined(__CUDA_ARCH__)
    const int barrier = 8 + group;
    naivert::named_barrier_sync(128u, unsigned(barrier));
#else
    (void)group;
#endif
  }

  template <bool kDirectScores = false>
  __device__ static void compute_router_scores(
      const RMSNormExpertProjArgs& args, unsigned char* smem,
      int task, int tid) {
    constexpr int vector_count = kRmsHiddenDim / 8;
    constexpr int kMaxChunks =
        (vector_count + kRmsConsumerGroupThreads - 1) /
        kRmsConsumerGroupThreads;
    const int group = tid / kRmsConsumerGroupThreads;
    const int group_tid =
        tid - group * kRmsConsumerGroupThreads;
    const int lane = tid & 31;
    const int group_warp = group_tid >> 5;
    float accumulators[kRows][kMaxChunks];
    if constexpr (kDirectScores) {
  #pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const auto* dots = reinterpret_cast<const float*>(
            smem + kXNormOffset + row * kXNormStride + kDotPartialOffset) + group * vector_count;
  #pragma unroll
        for (int i = 0; i < kMaxChunks; ++i)
          accumulators[row][i] = dots[group_tid + i * kRmsConsumerGroupThreads];
      }

    } else {
      const auto* router4 = reinterpret_cast<const uint4*>(
          smem + kRouterOffset +
          group * kRouterStride);
  #pragma unroll
      for (int row = 0; row < kRows; ++row) {
  #pragma unroll
        for (int i = 0; i < kMaxChunks; ++i) {
          accumulators[row][i] = 0.0f;
        }
      }

  #pragma unroll
      for (int i = 0; i < kMaxChunks; ++i) {
        const int chunk =
            group_tid + i * kRmsConsumerGroupThreads;
        if (chunk < vector_count) {
          const auto* router_values = reinterpret_cast<const naivert::RouterWeight*>(router4) + chunk * 8;
  #pragma unroll
          for (int row = 0; row < kRows; ++row) {
            const auto* x_norm4 =
                reinterpret_cast<const uint4*>(
                    smem + kXNormOffset +
                    row * kXNormStride);
            const uint4 x_vector = x_norm4[chunk];
            const auto* x_values =
                reinterpret_cast<const __nv_bfloat162*>(
                    &x_vector);
  #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const float2 x =
                  __bfloat1622float2(x_values[pair]);
              const float2 weight =
                  naivert::router_weight_pair(router_values, pair);
              accumulators[row][i] = __fmaf_rn(
                  weight.x, x.x, accumulators[row][i]);
              accumulators[row][i] = __fmaf_rn(
                  weight.y, x.y, accumulators[row][i]);
            }
          }
        }
      }

    }

    auto* partial = reinterpret_cast<float*>(
        smem + kReductionOffset +
        kRows * kRmsConsumerWarps *
            static_cast<int>(sizeof(float)));
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < kMaxChunks; ++i) {
        dot += accumulators[row][i];
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        partial[(group * kRows + row) *
                    kRmsConsumerGroupWarps +
                group_warp] = dot;
      }
    }
    subgroup_sync(group);

    if (group_tid == 0) {
      const int expert =
          task * kRmsExpertsPerTask + group;
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        float combined = 0.0f;
#pragma unroll
        for (int warp = 0;
             warp < kRmsConsumerGroupWarps; ++warp) {
          combined +=
              partial[(group * kRows + row) *
                          kRmsConsumerGroupWarps +
                      warp];
        }
        args.scores[row * kRmsExperts + expert] = combined;
      }
    }

    if (task == 0 && args.x_mlp != nullptr) {
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const auto* x_norm4 =
            reinterpret_cast<const uint4*>(
                smem + kXNormOffset +
                row * kXNormStride);
        auto* x_mlp4 = reinterpret_cast<uint4*>(
            args.x_mlp + row * kRmsHiddenDim);
        for (int chunk = tid; chunk < vector_count;
             chunk += kRmsConsumerThreads) {
          x_mlp4[chunk] = x_norm4[chunk];
        }
      }
    }
  }

};

using RMSNormExpertProjRows8Body = RMSNormExpertProjRowsBody<8>;

}  // namespace naivert_replica_rmsnorm_expert_proj
