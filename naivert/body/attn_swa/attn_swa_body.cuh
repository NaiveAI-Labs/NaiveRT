#pragma once
#include <cute/arch/copy_sm90_tma.hpp>
// Naive N0.5 sliding-window attention layer step for one query row on one
// TP8 rank, as a cooperative grid of one CTA per SM:
//
//   0. every CTA RMSNorms the row into shared memory;
//   1. fused bf16 GEMV over [q 1536 | k 192 | v 128] (one warp per row);
//   2. rope on q and k (first 64 dims of each head, theta 1e4), k / v appended
//      to the ring cache at position % 136 (the 128-token window plus room for
//      eight pending query rows);
//   3. one CTA per head: scores over the visible window (at most 128 keys),
//      softmax with the head's sink logit in the denominator, P x V -> proj_o.
//
// The RMSNorm / GEMV / publish stages mirror attn_dsa (same rounding points:
// bf16 x_norm, bf16 fused projections, bf16 cache rows); phases are separated
// by grid syncs.
#include "naivert/body/attn_swa/attn_swa_args.hpp"

#include <cooperative_groups.h>

#include "naivert/body/attn_common/rows_gemv_mma.cuh"
#include "naivert/body/attn_common/unproj_tail.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace naivert::attn_swa {

namespace cg = cooperative_groups;

template <bool kWideGamma>
struct AttnSwaBodyImpl {
  using Args = AttnSwaArgs;
  using Scratch = AttnSwaScratchLayout;
  static constexpr int kWmmaWarps = 12;
  static constexpr int kThreads = kWmmaWarps * 32;
  static constexpr float kNormEps = 1.0e-5f;
  static constexpr float kScoreScale = 0.07216878364870322f;  // 192^-0.5

  // Shared with the attention bodies (block reductions, bf16 loads).
#include "naivert/body/attn_common/cta_helpers.cuh"

  // ---------------------------------------------------------------- helpers
  __device__ __forceinline__ static float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
  }

  // acc += dot of eight bf16 pairs packed in two uint4.
  __device__ __forceinline__ static float dot8(float acc, uint4 w, uint4 x) {
    const unsigned ws[4] = {w.x, w.y, w.z, w.w};
    const unsigned xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      float2 wf = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(ws + pair));
      float2 xf = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(xs + pair));
      acc = fmaf(wf.x, xf.x, acc);
      acc = fmaf(wf.y, xf.y, acc);
    }
    return acc;
  }

  __device__ __forceinline__ static void rotate_pair(
      float& real, float& imaginary, float cosine, float sine) {
    // Pin the contraction order across row-count variants. Otherwise the
    // compiler can fuse either product into the add and change the BF16 K
    // cache at a rounding midpoint, even with identical projection values.
    float rotated_real = __fmaf_rn(-imaginary, sine, __fmul_rn(real, cosine));
    float rotated_imaginary = __fmaf_rn(real, sine, __fmul_rn(imaginary, cosine));
    real = rotated_real;
    imaginary = rotated_imaginary;
  }

  __device__ __forceinline__ static int ring_row(int position) {
    return position % Scratch::kWindowRows;
  }

  // ---------------------------------------------------------- 0: RMSNorm
  __device__ static void normalize_x_local(const Args& args, __nv_bfloat16* x_norm,
                                           float*) {
    naivert::attn_rows::normalize_rows<Scratch::kQueryRows, Scratch::kDim, kWideGamma>(
        args.residual, args.input_norm, x_norm, kNormEps);
  }

  // ------------------------------------------------ 1: fused bf16 GEMV
  __device__ __forceinline__ static const __nv_bfloat16* projection_row(
      const Args& args, int row) {
    if (row < Scratch::kKBase) {
      return args.wq + std::size_t(row) * Scratch::kDim;
    }
    if (row < Scratch::kVBase) {
      return args.wk + std::size_t(row - Scratch::kKBase) * Scratch::kDim;
    }
    return args.wv + std::size_t(row - Scratch::kVBase) * Scratch::kDim;
  }

  // One warp per output row; a lane owns kChunksPerLane 16-byte pieces of the
  // 8 KB weight row and issues all of their loads before reducing.
  static constexpr int kChunksPerLane = Scratch::kDim / 8 / 32;  // 16
  // Weight tiles of 16 rows in M, query rows in N; 8 warps split K (rows_gemv_mma.cuh).
  static constexpr int kGemvSplits = 8;
  static_assert(Scratch::kFused % 16 == 0 && Scratch::kKBase % 16 == 0 &&
                    Scratch::kVBase % 16 == 0,
                "every projection matrix must start on a 16-row tile");

  // Stage one 16-row weight tile (128 KiB) before RMSNorm. Verify-8 uses TMA;
  // decode retains per-thread cp.async. Both produce the same StagedTile layout.
  static constexpr int kProjectionTiles = Scratch::kFused / 16;   // 116
  static constexpr int kAttentionCtas = Scratch::kQueryRows * Scratch::kLocalHeads;
  using StagedTile = naivert::attn_rows::StagedTile<Scratch::kDim>;
  static constexpr int kDynamicSharedBytes = StagedTile::kBytes;

  __device__ static void stage_projection_tile(const Args& args, unsigned char* stage,
                                               std::uint64_t* barrier) {
    const int tile = int(blockIdx.x);
    if (tile < kProjectionTiles) {
      if constexpr (Scratch::kUseProjectionTma) {
        static_assert(Scratch::kDim == 4096);
        if (threadIdx.x == 0) {
          using Barrier = cutlass::arch::ClusterTransactionBarrier;
          Barrier::init(barrier, 1);
          cutlass::arch::fence_barrier_init();
          Barrier::arrive_and_expect_tx(barrier, StagedTile::kBytes);
          const int matrix = tile < Scratch::kKBase / 16 ? 0 : (tile < Scratch::kVBase / 16 ? 1 : 2);
          const int local_tile = tile - (matrix == 0 ? 0 : (matrix == 1 ? Scratch::kKBase : Scratch::kVBase) / 16);
          cute::SM90_TMA_LOAD_4D::copy(
              args.workspace + Scratch::kProjectionMapsOffset + matrix * Scratch::kProjectionMapBytes,
              barrier, 0, stage, 0, 0, 0, local_tile);
        }
      } else {
        StagedTile::issue([&](int row) { return projection_row(args, row); }, tile * 16, stage);
      }
    }
  }

  __device__ static void project_fused(const Args& args, const unsigned char* stage,
                                       const __nv_bfloat16* x_norm, __nv_bfloat16* fused,
                                       float* partials) {
    naivert::attn_rows::project_rows_mma_staged<Scratch::kQueryRows, Scratch::kDim, kGemvSplits,
                                                kWmmaWarps, (Scratch::kQueryRows == 8 ? kThreads : 0)>(
        0, kProjectionTiles, stage, x_norm, partials, [&](int row, int query, float value) {
          if (row >= Scratch::kVBase) value *= args.value_scale == 0.0f ? 1.0f : args.value_scale;
          fused[std::size_t(query) * Scratch::kFused + row] = __float2bfloat16(value);
        });
  }

  // ------------------------------------------ 2: rope and cache append
  // q: rope on the first 64 dims of each of the 8 heads -> q_rot [8, 192].

  // k (rope on the first 64 dims) and v into the ring cache row of `position`.

  // -------------------------------------------- 3: windowed attention
  // One CTA per local head. visible = min(position + 1, 128) keys ending at the
  // query's own (just appended) key; the sink logit joins the softmax
  // denominator only (HF: appended to the logits, dropped after the softmax).
  // One CTA per (query row, local head). visible = min(position + 1, 128) keys ending at
  // the query's own key; the sink logit joins the softmax denominator only (HF: appended to
  // the logits, dropped after the softmax). The keys and values of this launch's rows come
  // from the projection output (rounded to bf16 like the cache rows they become); the head-0
  // CTA of each row appends that row to the ring cache for later steps.
  __device__ static void attend_head(const Args& args, const __nv_bfloat16* fused,
                                     __nv_bfloat16* q_rot, int query, int head, int position,
                                     int base_pos, float* scores, float* q_head,
                                     float* reduction, float* pv_partial, float* k_fresh,
                                     float* v_fresh, float* partials) {
    int tid = int(threadIdx.x);
    int warp = tid >> 5;
    int lane = tid & 31;
    const int visible = position + 1 < Scratch::kWindow ? position + 1 : Scratch::kWindow;
    const int first = position + 1 - visible;
    const int cached = base_pos > first ? base_pos - first : 0;
    constexpr int kRowsPerWarp = (Scratch::kWindow + kWmmaWarps - 1) / kWmmaWarps;  // 11
    constexpr int kLanesPerKey = Scratch::kHeadDim / 8;  // 24
    const bool key_lane = lane < kLanesPerKey;
    // 0. the phase is a chain of dependent global round trips, so this
    //    warp's cached window rows t = warp + 12 j -- key (lanes 0..23, 16 bytes each) and
    //    value (a lane owns four dims) -- and the sink logit are loaded first and consumed
    //    from registers after the shared-memory sync.
    uint4 key_loaded[kRowsPerWarp];
    uint2 value_loaded[kRowsPerWarp];
#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j) {
      const int t = warp + j * kWmmaWarps;
      if (t >= cached) {
        break;
      }
      {
        const int row = ring_row(first + t);
        if (key_lane) {
          key_loaded[j] = *reinterpret_cast<const uint4*>(
              args.k_window + std::size_t(row) * Scratch::kHeadDim + lane * 8);
        }
        value_loaded[j] = *reinterpret_cast<const uint2*>(
            args.v_window + std::size_t(row) * Scratch::kVDim + lane * 4);
      }
    }
    const float sink = args.sink_bias[head];
    // 1. This head's query: rope on the first 64 dims (as publish_query did), kept as fp32
    //    of the bf16-rounded values, and written to q_rot for the harness.
    {
      const __nv_bfloat16* source =
          fused + std::size_t(query) * Scratch::kFused + Scratch::kQBase + head * Scratch::kHeadDim;
      __nv_bfloat16* destination = q_rot + std::size_t(query) * Scratch::kQ + head * Scratch::kHeadDim;
      const float* rope = args.rope_freqs + std::size_t(position) * Scratch::kRope;
      if (tid < Scratch::kHeadDim / 2) {
        int element = tid * 2;
        float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(source + element));
        if (element < Scratch::kRope) {
          rotate_pair(value.x, value.y, rope[element], rope[element + 1]);
        }
        const __nv_bfloat162 rounded = __floats2bfloat162_rn(value.x, value.y);
        *reinterpret_cast<__nv_bfloat162*>(destination + element) = rounded;
        const float2 back = __bfloat1622float2(rounded);
        q_head[element] = back.x;
        q_head[element + 1] = back.y;
      }
    }
    // 2. Keys and values of rows 0..query of this launch (as publish_kv computes them); the
    //    head-0 CTA of row `query` also appends its row to the ring cache.
    //    All rows' loads are issued before any is consumed.
    if (tid < Scratch::kHeadDim / 2) {
      const int element = tid * 2;
      __nv_bfloat162 raw[Scratch::kQueryRows];
      float2 rope_pair[Scratch::kQueryRows];
#pragma unroll
      for (int row = 0; row < Scratch::kQueryRows; ++row) {
        if (row > query) {
          break;
        }
        {
          raw[row] = *reinterpret_cast<const __nv_bfloat162*>(
              fused + std::size_t(row) * Scratch::kFused + Scratch::kKBase + element);
          if (element < Scratch::kRope) {
            rope_pair[row] = *reinterpret_cast<const float2*>(
                args.rope_freqs + std::size_t(base_pos + row) * Scratch::kRope + element);
          }
        }
      }
#pragma unroll
      for (int row = 0; row < Scratch::kQueryRows; ++row) {
        if (row > query) {
          break;
        }
        {
          float2 value = __bfloat1622float2(raw[row]);
          if (element < Scratch::kRope) {
            rotate_pair(value.x, value.y, rope_pair[row].x, rope_pair[row].y);
          }
          const __nv_bfloat162 rounded = __floats2bfloat162_rn(value.x, value.y);
          const float2 back = __bfloat1622float2(rounded);
          k_fresh[row * Scratch::kHeadDim + element] = back.x;
          k_fresh[row * Scratch::kHeadDim + element + 1] = back.y;
          if (head == 0 && row == query) {
            *reinterpret_cast<__nv_bfloat162*>(
                args.k_window + std::size_t(ring_row(base_pos + row)) * Scratch::kHeadDim +
                element) = rounded;
          }
        }
      }
    } else if (tid < Scratch::kHeadDim / 2 + Scratch::kVDim / 2) {
      const int element = (tid - Scratch::kHeadDim / 2) * 2;
      __nv_bfloat162 raw[Scratch::kQueryRows];
#pragma unroll
      for (int row = 0; row < Scratch::kQueryRows; ++row) {
        if (row > query) {
          break;
        }
        {
          raw[row] = *reinterpret_cast<const __nv_bfloat162*>(
              fused + std::size_t(row) * Scratch::kFused + Scratch::kVBase + element);
        }
      }
#pragma unroll
      for (int row = 0; row < Scratch::kQueryRows; ++row) {
        if (row > query) {
          break;
        }
        {
          const float2 value = __bfloat1622float2(raw[row]);
          v_fresh[row * Scratch::kVDim + element] = value.x;
          v_fresh[row * Scratch::kVDim + element + 1] = value.y;
          if (head == 0 && row == query) {
            *reinterpret_cast<__nv_bfloat162*>(
                args.v_window + std::size_t(ring_row(base_pos + row)) * Scratch::kVDim +
                element) = raw[row];
          }
        }
      }
    }
    __syncthreads();
    // 3. Scores: one warp per key, eight dims per key lane, from the keys loaded in 0; then
    //    the <= kQueryRows fresh rows from shared memory.
    {
      float q8[8];
      if constexpr (Scratch::kQueryRows == 8) {
        if (key_lane) {
          // q_head is 16-byte aligned; every lane owns eight floats.
          const float4 lo = *reinterpret_cast<const float4*>(q_head + lane * 8);
          const float4 hi = *reinterpret_cast<const float4*>(q_head + lane * 8 + 4);
          q8[0] = lo.x;
          q8[1] = lo.y;
          q8[2] = lo.z;
          q8[3] = lo.w;
          q8[4] = hi.x;
          q8[5] = hi.y;
          q8[6] = hi.z;
          q8[7] = hi.w;
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            q8[i] = 0.0f;
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          q8[i] = key_lane ? q_head[lane * 8 + i] : 0.0f;
        }
      }
#pragma unroll
      for (int j = 0; j < kRowsPerWarp; ++j) {
        const int t = warp + j * kWmmaWarps;
        if (t >= cached) {
          break;
        }
        {
          float acc = 0.0f;
          if (key_lane) {
            const unsigned words[4] = {key_loaded[j].x, key_loaded[j].y, key_loaded[j].z,
                                       key_loaded[j].w};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const float2 kf = __bfloat1622float2(
                  *reinterpret_cast<const __nv_bfloat162*>(words + pair));
              acc = fmaf(q8[pair * 2], kf.x, acc);
              acc = fmaf(q8[pair * 2 + 1], kf.y, acc);
            }
          }
          acc = warp_sum(acc);
          if (lane == 0) {
            scores[t] = acc * kScoreScale;
          }
        }
      }
    }
    for (int t = cached + warp; t < visible; t += kWmmaWarps) {
      const float* key = k_fresh + (first + t - base_pos) * Scratch::kHeadDim;
      float acc = 0.0f;
      // Match the cached-key reduction: eight consecutive dimensions per
      // active lane. A key must have the same score after entering the cache
      // as it has while it is a fresh row in a batched launch.
      if (key_lane) {
#pragma unroll
        for (int piece = 0; piece < 8; ++piece) {
          const int dimension = lane * 8 + piece;
          acc = fmaf(q_head[dimension], key[dimension], acc);
        }
      }
      acc = warp_sum(acc);
      if (lane == 0) {
        scores[t] = acc * kScoreScale;
      }
    }
    __syncthreads();
    // 4. Softmax with the sink logit in the denominator.
    if constexpr (Scratch::kQueryRows == 8) {
      // Preserve the block reduction tree over the 128-key window, while
      // keeping intermediate maxima and sums inside the first warp.
      if (warp == 0) {
        float value[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          value[i] = lane + i * 32 < visible ? scores[lane + i * 32] : -3.402823466e38f;
        float maximum = fmaxf(fmaxf(value[0], value[2]), fmaxf(value[1], value[3]));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        maximum = fmaxf(__shfl_sync(0xffffffffu, maximum, 0), sink);
        float exponent[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          exponent[i] = lane + i * 32 < visible ? __expf(value[i] - maximum) : 0.0f;
        float denominator = (exponent[0] + exponent[2]) + (exponent[1] + exponent[3]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          denominator += __shfl_down_sync(0xffffffffu, denominator, offset);
        denominator = __shfl_sync(0xffffffffu, denominator, 0) + __expf(sink - maximum);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (lane + i * 32 < visible) scores[lane + i * 32] = exponent[i] / denominator;
      }
    } else {
      float local_max = tid < visible ? scores[tid] : -3.402823466e38f;
      float maximum = fmaxf(block_max(local_max, reduction), sink);
      // All warps must read the maximum before block_sum reuses reduction[0].
      __syncthreads();
      float exponent = tid < visible ? __expf(scores[tid] - maximum) : 0.0f;
      float denominator = block_sum(exponent, reduction) + __expf(sink - maximum);
      if (tid < visible) {
        scores[tid] = exponent / denominator;
      }
    }
    __syncthreads();
    // 5. P x V: one warp per key row (rows t = warp mod 12) from the values loaded in 0 (a
    //    lane owns four consecutive dims), then the fresh rows from shared memory; the 12
    //    per-warp partials are reduced through the projection's partials buffer.
    {
      const int dim0 = lane * 4;
      float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
      for (int j = 0; j < kRowsPerWarp; ++j) {
        const int t = warp + j * kWmmaWarps;
        if (t >= cached) {
          break;
        }
        {
          const float probability = scores[t];
          const float2 low = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&value_loaded[j].x));
          const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&value_loaded[j].y));
          acc0 = fmaf(probability, low.x, acc0);
          acc1 = fmaf(probability, low.y, acc1);
          acc2 = fmaf(probability, high.x, acc2);
          acc3 = fmaf(probability, high.y, acc3);
        }
      }
      // Fresh rows: the first t >= cached with t = warp (mod 12).
      for (int t = cached + (((warp - cached) % kWmmaWarps) + kWmmaWarps) % kWmmaWarps; t < visible;
           t += kWmmaWarps) {
        const float probability = scores[t];
        const float* value = v_fresh + (first + t - base_pos) * Scratch::kVDim + dim0;
        acc0 = fmaf(probability, value[0], acc0);
        acc1 = fmaf(probability, value[1], acc1);
        acc2 = fmaf(probability, value[2], acc2);
        acc3 = fmaf(probability, value[3], acc3);
      }
      float* mine = partials + warp * Scratch::kVDim + dim0;
      mine[0] = acc0;
      mine[1] = acc1;
      mine[2] = acc2;
      mine[3] = acc3;
    }
    __syncthreads();
    if (tid < Scratch::kVDim) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kWmmaWarps; ++w) {
        total += partials[w * Scratch::kVDim + tid];
      }
      args.proj_o[(std::size_t(query) * Scratch::kLocalHeads + head) * Scratch::kVDim +
                  tid] = __float2bfloat16(total);
    }
  }

  __device__ static bool arguments_valid(const Args& args, int base_pos) {
    return args.workspace != nullptr && args.residual != nullptr &&
           args.input_norm != nullptr && args.wq != nullptr && args.wk != nullptr &&
           args.wv != nullptr && args.rope_freqs != nullptr &&
           args.sink_bias != nullptr && args.k_window != nullptr &&
           args.v_window != nullptr && args.proj_o != nullptr && base_pos >= 0 &&
           args.max_seq_len <= Scratch::kMaxSeqLen &&
           args.max_seq_len >= Scratch::kQueryRows &&
           base_pos <= args.max_seq_len - Scratch::kQueryRows;
  }

  __device__ static void run(const Args& args) {
    // PDL: let the next kernel launch now; wait for the previous grid (its outputs are
    // this kernel's inputs) before reading anything. No-ops under a plain launch.
    naivert::pdl::trigger_dependents<true>();
    naivert::pdl::wait_for_primary<true>();
    int base_pos = args.cur_pos == nullptr ? -1 : *args.cur_pos;
    if (!arguments_valid(args, base_pos)) {
      return;
    }
    __shared__ __align__(8) float reduction[512];
    __shared__ alignas(16) float gemv_partials[kWmmaWarps * 128];
    __shared__ alignas(16) __nv_bfloat16 x_norm[Scratch::kQueryRows * Scratch::kDim];
    __shared__ float scores[Scratch::kWindow];
    __shared__ alignas(16) float q_head[Scratch::kHeadDim];
    __shared__ float pv_partial[kThreads];
    __shared__ float k_fresh[Scratch::kQueryRows * Scratch::kHeadDim];
    __shared__ float v_fresh[Scratch::kQueryRows * Scratch::kVDim];
    auto* fused = reinterpret_cast<__nv_bfloat16*>(args.workspace + Scratch::kFusedOffset);
    auto* q_rot = reinterpret_cast<__nv_bfloat16*>(args.workspace + Scratch::kQRotOffset);
    // TMA's 128-byte swizzle matches StagedTile only at a 1-KiB-aligned origin.
    extern __shared__ __align__(Scratch::kUseProjectionTma ? 1024 : 128) unsigned char attn_swa_dynamic[];
    auto grid = cg::this_grid();

    // The weight tile streams into shared memory while every CTA normalises
    // into its own x_norm. The normalization CTA join also publishes TMA barrier
    // initialization. reduction is unused until attention, after the grid join.
    stage_projection_tile(args, attn_swa_dynamic, reinterpret_cast<std::uint64_t*>(reduction));
    normalize_x_local(args, x_norm, reduction);
    if constexpr (Scratch::kUseProjectionTma) {
      if (int(blockIdx.x) < kProjectionTiles)
        cutlass::arch::ClusterTransactionBarrier::wait(reinterpret_cast<std::uint64_t*>(reduction), 0);
    }
    project_fused(args, attn_swa_dynamic, x_norm, fused, gemv_partials);
    if constexpr (Scratch::kUseProjectionTma) {
      if (int(blockIdx.x) < kProjectionTiles && threadIdx.x == 0)
        cutlass::arch::ClusterTransactionBarrier::invalidate(reinterpret_cast<std::uint64_t*>(reduction));
    }
    grid.sync();
    // No publish stage; each attention CTA ropes its query and this launch's keys and
    // values itself (the head-0 CTA of a row appends it to the ring cache).
    {
      int worker = int(blockIdx.x);
      if (worker < Scratch::kQueryRows * Scratch::kLocalHeads) {
        int query = worker / Scratch::kLocalHeads;
        int head = worker - query * Scratch::kLocalHeads;
        attend_head(args, fused, q_rot, query, head, base_pos + query, base_pos, scores,
                    q_head, reduction, pv_partial, k_fresh, v_fresh, gemv_partials);
      }
    }
    if (args.unproj.o_weight != nullptr) {
      // O-projection + TP8 all-reduce as the kernel's last phase (unproj_tail.cuh);
      // the attention output is complete after the barrier, x_norm's shared memory is free.
      grid.sync();
      using Tail = naivert::attn_unproj::UnprojTail<Scratch::kQueryRows, Scratch::kDim,
                                                    Scratch::kLocalHeads * Scratch::kVDim,
                                (Scratch::kQueryRows == 8), (Scratch::kQueryRows == 8 ? kThreads : 0)>;
      static_assert(Tail::kSharedBytes <= int(sizeof(x_norm)),
                    "the fused o-projection reuses x_norm's shared memory");
      Tail::run(args.unproj, args.proj_o, args.residual,
                reinterpret_cast<unsigned char*>(x_norm));
    }
  }
};

using AttnSwaBody = AttnSwaBodyImpl<false>;

}  // namespace naivert::attn_swa
