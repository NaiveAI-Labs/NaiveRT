#pragma once
#include <cute/arch/copy_sm90_tma.hpp>
// Naive N0.5 full-attention layer step (GQA 8 query heads : 1 kv head per
// TP8 rank, head 192 = 64 rotary + 128 plain, value 128) with the
// sparse-attention indexer (16 or 64 replicated index heads x 128, top-k 2048).
//
// One cooperative grid (kSmCount CTAs x 384 threads) runs the whole layer
// step for one query row:
//
//   0  normalize_x            RMSNorm(x) -> x_norm (bf16)
//   1  project_fused          x_norm x [wq | wk | wv | wik | wiq | wis] (bf16 GEMV)
//   2  publish_*              rope on q / k / index-q / index-k, LayerNorm on
//                             index-k, cache append of k, v, ki
//   3  index logits + top-k   MLA-v28 stages from naivert/body/attn_common
//                             (identity below 2048 visible tokens)
//   4  compute_attention      split-KV softmax over the selected tokens,
//                             combine -> proj_o [rows, 8, 128]
//
// Phases are separated by grid syncs in this first version; the ready-flag
// overlap of the MLA one-row body is a later optimisation.
#include "naivert/body/attn_dsa/attn_dsa_args.hpp"

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda/ptx>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
// The tcgen05 index sweep uses cute for the UMMA descriptors, the MMA atom, TMEM.
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/float8.h>

#include "naivert/body/attn_common/rows_gemv_mma.cuh"
#include "naivert/body/attn_common/unproj_tail.cuh"
#include "naivert/pdl/grid_dependency.cuh"

#include <cstddef>
#include <cstdint>

namespace naivert::attn_dsa {

namespace cg = cooperative_groups;

template <bool kWideGamma>
struct AttnDsaBodyImpl {
  using Args = AttnDsaArgs;
  using Scratch = AttnDsaScratchLayout;

  static constexpr int kWmmaWarps = 12;
  static constexpr int kThreads = kWmmaWarps * 32;
  static constexpr float kNormEps = 1.0e-5f;
  // weights_proj(x) * n_heads^-0.5, as in the Naive N indexer reference.
  static constexpr float kIndexWeightScale = Scratch::kIndexHeads == 16 ? 0.25f : 0.125f;

  // RadixControl (defined by the shared top-k fragment) keeps the MLA-v28
  // field layout, including its fused-ready shard counters, so that
  // Scratch::kRadixControlBytes stays shared between the two bodies.
  static constexpr int kFusedReadyShards = 10;
  static constexpr int kTopKSortItemsPerThread = 22;
  using TopKBlockRadixSort =
      cub::BlockRadixSort<unsigned, kThreads, kTopKSortItemsPerThread,
                          cub::NullType, 5>;
  static constexpr int kTopKWindowItemsPerLane = 3;
  static constexpr int kPhaseTimerFloatOffset = Scratch::kPhaseTimerFloatOffset;
  static constexpr int kPhaseTimerCount = Scratch::kPhaseTimerCount;

  // Attention scratch inside one query row's kAttentionTempStride floats.
  // One CTA per (query row, split): rows x splits must fit the persistent grid.
  // BF16 probability rounding happens inside each softmax partition.
  // Naive N0.5 decode and verify must use identical partitions to compute the same
  // function from identical Q/K/V and selected keys.
  static constexpr int kAttentionSplits = Scratch::kIndexHeads == 16
      ? 16 : (Scratch::kQueryRows > 4 ? 16 : 32);
  static_assert(Scratch::kQueryRows * kAttentionSplits <= 148,
                "attention splits x query rows must fit one CTA per SM");
  static constexpr int kTokenTile = 16;
  static constexpr int kScoreFloats = Scratch::kIndexTopK * Scratch::kLocalHeads;
  static constexpr int kProbabilityStorageFloats =
      kScoreFloats * int(sizeof(__nv_bfloat16)) / int(sizeof(float));
  static constexpr int kSplitOutputFloats =
      kAttentionSplits * Scratch::kPaddedHeads * Scratch::kVDim;
  static constexpr int kSplitStatsFloats =
      kAttentionSplits * Scratch::kLocalHeads * 2;
  static constexpr int kSplitOutputOffset = kScoreFloats + kProbabilityStorageFloats;
  static constexpr int kSplitStatsOffset = kSplitOutputOffset + kSplitOutputFloats;
  static_assert(kSplitStatsOffset + kSplitStatsFloats <= kPhaseTimerFloatOffset,
                "attention temporaries must not reach the timer/flag tail");
  static_assert(kPhaseTimerFloatOffset + 64 + Scratch::kLocalHeads +
                        Scratch::kQueryRows <=
                    Scratch::kAttentionTempStride,
                "ready flags must fit the attention scratch tail");

  struct alignas(32) ProjectionTile {
    __nv_bfloat16 a[kWmmaWarps][16 * 16];
    __nv_bfloat16 b[kWmmaWarps][16 * 16];
    float c[kWmmaWarps][16 * 16];
  };
  // The normalised rows as e4m3 (rows 16 B apart mod 128: conflict-free B loads).
  static constexpr int kX8Stride = Scratch::kDim + 16;
  struct SharedWorkspace {
    float reduction[512];
    float ki_values[Scratch::kIndexDim];
    alignas(16) float attention_scores[Scratch::kIndexTopK];
    ProjectionTile projection_tile;
    alignas(16) __nv_fp8_e4m3 x8[Scratch::kQueryRows * kX8Stride];
  };
  struct alignas(16) CombineWorkspace {
    float values[kAttentionSplits * (Scratch::kVDim / 2)];
    float corrections[kAttentionSplits + 1];
  };
  // One split's selected keys / values, the query row, the split's scores and
  // probability tiles, staged by one cp.async burst; row strides padded by 8 bf16 (16 B) so
  // consecutive rows land on different bank segments for the wmma loads.
  static constexpr int kSplitTokens = Scratch::kIndexTopK / kAttentionSplits;
  static_assert(kSplitTokens * kAttentionSplits == Scratch::kIndexTopK && kSplitTokens % kTokenTile == 0,
                "a split owns whole 16-token tiles");
  static constexpr int kSplitTiles = kSplitTokens / kTokenTile;
  static constexpr int kGatherKeyStride = Scratch::kHeadDim + 8;
  static constexpr int kGatherValueStride = Scratch::kVDim + 8;
  struct alignas(16) GatherStorage {
    __nv_bfloat16 keys[kSplitTokens * kGatherKeyStride];
    __nv_bfloat16 values[kSplitTokens * kGatherValueStride];
    __nv_bfloat16 query[Scratch::kPaddedHeads * kGatherKeyStride];
    float scores[kSplitTokens * Scratch::kPaddedHeads];
    __nv_bfloat16 probabilities[kSplitTiles * Scratch::kPaddedHeads * kTokenTile];
  };
  // Shared memory of the tcgen05 index sweep; the phase sits between two grid barriers,
  // so it aliases the other members.
  struct alignas(1024) SweepStorage {
    static constexpr int kCols = Scratch::kQueryRows * Scratch::kIndexHeads;
    // TMEM reads 32 columns at a time. Decode pads 16 heads to 32 columns;
    // verify packs two real query rows into each 32-column read.
    static constexpr int kPaddedCols = kCols < 32 ? 32 : kCols;
    static constexpr int kPassN = kPaddedCols > 256 ? 256 : kPaddedCols;
    static constexpr int kPasses = kPaddedCols / kPassN;
    static constexpr int kStages = 3;
    static constexpr int kTileM = 128;
    static constexpr int kTileBytes = kTileM * Scratch::kIndexDim;   // 16 KB e4m3
    static constexpr int kTmemCols = 2 * kPassN >= 32 ? 2 * kPassN : 32;
    alignas(1024) unsigned char b_image[kPaddedCols * Scratch::kIndexDim];
    alignas(1024) unsigned char a_ring[kStages * kTileBytes];
    float weights[kPaddedCols];
    unsigned long long full[kStages];
    unsigned long long empty[kStages];
    unsigned long long acc_full[kPasses];
    unsigned long long acc_empty[kPasses];
    unsigned long long b_full;
    unsigned tmem_slot;
  };
  // Alias the later attention/index workspace. Keep partials and the barrier
  // outside the incoming weight image; the image's swizzle needs 1-KiB alignment.
  using ProjectionStage = naivert::attn_rows::StagedTile<Scratch::kDim>;
  struct alignas(1024) ProjectionStorage {
    unsigned char weights[Scratch::kUseProjectionTma ? ProjectionStage::kBytes : 1];
    float partials[Scratch::kUseProjectionTma ? kWmmaWarps * 128 : 1];
    alignas(8) std::uint64_t barrier;
  };
  union SharedStorage {
    ProjectionStorage projection_stage;
    SharedWorkspace workspace;
    GatherStorage gather;
    TopKBlockRadixSort::TempStorage topk_sort;
    CombineWorkspace combine;
    SweepStorage sweep;
  };
  static_assert(sizeof(SharedStorage) >=
                    (kThreads * kTopKSortItemsPerThread + 66) * sizeof(unsigned),
                "exact TopK refinement must fit shared storage");

  // Shared with the MLA-v28 body (verbatim fragments).
#include "naivert/body/attn_common/cta_helpers.cuh"
#include "naivert/body/attn_common/index_topk_stage.cuh"
#include "naivert/body/attn_dsa/index_sweep_tcgen05.cuh"
#include "naivert/body/attn_dsa/topk_grid_select.cuh"
#include "naivert/body/attn_dsa/topk_local_select.cuh"

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
    // Keep the same multiply/FMA rounding order in every row-count variant.
    float rotated_real = __fmaf_rn(-imaginary, sine, __fmul_rn(real, cosine));
    float rotated_imaginary = __fmaf_rn(real, sine, __fmul_rn(imaginary, cosine));
    real = rotated_real;
    imaginary = rotated_imaginary;
  }

  // ---------------------------------------------------------- 0: RMSNorm

  // Every CTA normalises the query rows into its own shared copy (8 KB per
  // row): cheaper than a global round trip plus a grid barrier, and the GEMV
  // then reads x_norm from shared memory.
  __device__ static void normalize_x_local(const Args& args, __nv_bfloat16* x_norm,
                                           float*) {
    naivert::attn_rows::normalize_rows<Scratch::kQueryRows, Scratch::kDim, kWideGamma>(
        args.residual, args.input_norm, x_norm, kNormEps);
  }

  // ------------------------------------------------ 1: fused bf16 GEMV

  static constexpr int kProjectionRows = Scratch::kFused + Scratch::kIndexHeads;

  __device__ __forceinline__ static const __nv_bfloat16* projection_row(
      const Args& args, int row) {
    if (row < Scratch::kKBase) {
      return args.wq + std::size_t(row) * Scratch::kDim;
    }
    if (row < Scratch::kVBase) {
      return args.wk + std::size_t(row - Scratch::kKBase) * Scratch::kDim;
    }
    if (row < Scratch::kIkBase) {
      return args.wv + std::size_t(row - Scratch::kVBase) * Scratch::kDim;
    }
    if (row < Scratch::kIqBase) {
      return args.wik + std::size_t(row - Scratch::kIkBase) * Scratch::kDim;
    }
    if (row < Scratch::kFused) {
      return args.wiq_bf16 + std::size_t(row - Scratch::kIqBase) * Scratch::kDim;
    }
    return args.wis + std::size_t(row - Scratch::kFused) * Scratch::kDim;
  }

  // X_norm (bf16, shared) -> e4m3 rows (direct conversion: the normalised values are O(1),
  // the weights carry the block scales), all threads, ends with __syncthreads.
  __device__ static void convert_x8(const __nv_bfloat16* x_norm, __nv_fp8_e4m3* x8) {
    constexpr int kPairs = Scratch::kDim / 2;
    for (int index = int(threadIdx.x); index < Scratch::kQueryRows * kPairs; index += kThreads) {
      const int row = index / kPairs;
      const int pair = index - row * kPairs;
      reinterpret_cast<__nv_fp8x2_e4m3*>(x8 + row * kX8Stride)[pair] = __nv_fp8x2_e4m3(
          reinterpret_cast<const __nv_bfloat162*>(x_norm + std::size_t(row) * Scratch::kDim)[pair]);
    }
    __syncthreads();
  }

  __device__ __forceinline__ static void mma_e4m3_16832(float (&c)[4], unsigned a0, unsigned a1,
                                                        unsigned a2, unsigned a3, unsigned b0,
                                                        unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  // The 8192 index-query rows from e4m3 weights [rows][4096] (natural row-major) with
  // block-128 fp32 scales [rows/128][32]. Tiles of 16 rows, kIndexTilesPerCta per CTA, two K
  // halves per tile (8 warps). A lane loads 16 consecutive K of rows g and g + 8 per 64-K step
  // (two m16n8k32, the same K permutation in A and B, so the dot product is exact); the fp32
  // accumulator of each 128-K chunk is scaled by the chunk's block scale. Partials over the K
  // halves are summed in shared memory in a fixed order and emitted as bf16 into `fused`.
  __device__ static void project_index_fp8(const Args& args, const __nv_fp8_e4m3* x8,
                                           __nv_bfloat16* fused, float* partials) {
    constexpr int kTiles = Scratch::kIqRows / 16;
    constexpr int kSplits = 2;
    constexpr int kChunks = Scratch::kDim / 128 / kSplits;   // 128-K chunks per warp: 16
    constexpr int kActiveWarps = kIndexTilesPerCta * kSplits;
    static_assert(kActiveWarps <= kWmmaWarps && Scratch::kDim % 256 == 0, "");
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int q = lane & 3;
    const int local_tile = warp / kSplits;
    const int split = warp - local_tile * kSplits;
    const uint4* x_row = reinterpret_cast<const uint4*>(x8 + std::size_t(g) * kX8Stride);
    for (int tile_base = int(blockIdx.x) * kIndexTilesPerCta; tile_base < kTiles;
         tile_base += int(gridDim.x) * kIndexTilesPerCta) {
      const int tile = tile_base + local_tile;
      if (warp < kActiveWarps && tile < kTiles) {
        const int row0 = tile * 16 + g;
        const uint4* w0 = reinterpret_cast<const uint4*>(args.wiq + std::size_t(row0) * Scratch::kDim);
        const uint4* w1 = reinterpret_cast<const uint4*>(args.wiq + std::size_t(row0 + 8) * Scratch::kDim);
        const float* scales = args.wiq_scale + (tile * 16 / 128) * (Scratch::kDim / 128);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int chunk_begin = split * kChunks;
#pragma unroll 4
        for (int chunk = chunk_begin; chunk < chunk_begin + kChunks; ++chunk) {
          // 128 K = 2 steps of 64 K; per step this lane owns K [k, k + 16) of its 4-lane quad.
          uint4 a0[2], a1[2], b[2];
#pragma unroll
          for (int step = 0; step < 2; ++step) {
            const int k = chunk * 128 + step * 64 + q * 16;   // bytes == K for e4m3
            a0[step] = __ldg(w0 + (k >> 4));
            a1[step] = __ldg(w1 + (k >> 4));
            b[step] = g < Scratch::kQueryRows ? x_row[k >> 4] : make_uint4(0u, 0u, 0u, 0u);
          }
          float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
          for (int step = 0; step < 2; ++step) {
            // K slots (4q..4q+3) <- k..k+3, (4q+16..) <- k+4..k+7; then k+8..k+15.
            mma_e4m3_16832(c, a0[step].x, a1[step].x, a0[step].y, a1[step].y, b[step].x, b[step].y);
            mma_e4m3_16832(c, a0[step].z, a1[step].z, a0[step].w, a1[step].w, b[step].z, b[step].w);
          }
          const float scale = __ldg(scales + chunk);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc[i] = fmaf(c[i], scale, acc[i]);
          }
        }
        float* mine = partials + warp * 128;
        mine[g * 8 + 2 * q] = acc[0];
        mine[g * 8 + 2 * q + 1] = acc[1];
        mine[(g + 8) * 8 + 2 * q] = acc[2];
        mine[(g + 8) * 8 + 2 * q + 1] = acc[3];
      }
      __syncthreads();
      for (int item = int(threadIdx.x); item < kIndexTilesPerCta * 16 * Scratch::kQueryRows;
           item += kThreads) {
        const int t = item / (16 * Scratch::kQueryRows);
        const int rest = item - t * 16 * Scratch::kQueryRows;
        const int row = rest / Scratch::kQueryRows;
        const int query = rest - row * Scratch::kQueryRows;
        if (tile_base + t < kTiles) {
          float sum = 0.0f;
#pragma unroll
          for (int sp = 0; sp < kSplits; ++sp) {
            sum += partials[(t * kSplits + sp) * 128 + row * 8 + query];
          }
          fused[std::size_t(query) * Scratch::kFused + Scratch::kIqBase + (tile_base + t) * 16 + row] =
              __float2bfloat16(sum);
        }
      }
      __syncthreads();
    }
  }

  // One warp per output row. A lane owns kChunksPerLane 16-byte pieces of the
  // 8 KB weight row and issues all of their loads before reducing, so each
  // warp keeps the whole row in flight (the GEMV is HBM-bound). Below 2048
  // visible tokens the selection is the identity, so the 8192 replicated
  // index-query rows (67 of the 84 MB) and weights_proj are skipped; only
  // q, k, v and the index key (cache append) are projected.
  static constexpr int kChunksPerLane = Scratch::kDim / 8 / 32;  // 16

  // Weight tiles of 16 rows in M, query rows in N; 8 warps split K (rows_gemv_mma.cuh).
  // Below the sparse regime the index-query rows (wiq) are skipped, as before.
  static constexpr int kGemvSplits = 8;
  static constexpr int kIndexGemvSplits = 2;
  static constexpr int kIndexTilesPerCta = 4;
  static_assert(Scratch::kKBase % 16 == 0 && Scratch::kVBase % 16 == 0 &&
                    Scratch::kIkBase % 16 == 0 && Scratch::kIqBase % 16 == 0 &&
                    Scratch::kFused % 16 == 0 && kProjectionRows % 16 == 0,
                "every projection matrix must start on a 16-row tile");

  __device__ static void issue_projection(const Args& args, ProjectionStorage& stage) {
    const int tile = int(blockIdx.x);
    if (tile < Scratch::kIqBase / 16 && threadIdx.x == 0) {
      using Barrier = cutlass::arch::ClusterTransactionBarrier;
      Barrier::init(&stage.barrier, 1);
      cutlass::arch::fence_barrier_init();
      Barrier::arrive_and_expect_tx(&stage.barrier, ProjectionStage::kBytes);
      const int matrix = tile < Scratch::kKBase / 16 ? 0 :
          (tile < Scratch::kVBase / 16 ? 1 : (tile < Scratch::kIkBase / 16 ? 2 : 3));
      const int first = matrix == 0 ? 0 : (matrix == 1 ? Scratch::kKBase :
          (matrix == 2 ? Scratch::kVBase : Scratch::kIkBase));
      cute::SM90_TMA_LOAD_4D::copy(args.workspace + Scratch::kProjectionMapsOffset + matrix * Scratch::kProjectionMapBytes,
          &stage.barrier, 0, stage.weights, 0, 0, 0, tile - first / 16);
    }
  }

  __device__ static void project_fused(const Args& args, const __nv_bfloat16* x_norm,
                                       __nv_bfloat16* fused, float* index_weights,
                                       bool project_index, float* partials,
                                       __nv_fp8_e4m3* x8, ProjectionStorage& stage) {
    auto row_ptr = [&](int row) { return projection_row(args, row); };
    auto emit = [&](int row, int query, float value) {
      if (row >= Scratch::kVBase && row < Scratch::kIkBase)
        value *= args.value_scale == 0.0f ? 1.0f : args.value_scale;
      if (row < Scratch::kFused) {
        fused[std::size_t(query) * Scratch::kFused + row] = __float2bfloat16(value);
      } else {
        index_weights[query * Scratch::kIndexHeads + (row - Scratch::kFused)] =
            value * kIndexWeightScale;
      }
    };
    // q, k, v, ik: 124 tiles -> one tile per CTA, eight K splits.
    if constexpr (Scratch::kUseProjectionTma) {
      if (int(blockIdx.x) < Scratch::kIqBase / 16)
        cutlass::arch::ClusterTransactionBarrier::wait(&stage.barrier, 0);
      naivert::attn_rows::project_rows_mma_staged<Scratch::kQueryRows, Scratch::kDim,
          kGemvSplits, kWmmaWarps, (Scratch::kQueryRows == 8 ? kThreads : 0)>(0, Scratch::kIqBase / 16, stage.weights, x_norm,
                                 stage.partials, emit);
      // The staged projection ends with a CTA join; no shared reader remains.
      if (int(blockIdx.x) < Scratch::kIqBase / 16 && threadIdx.x == 0)
        cutlass::arch::ClusterTransactionBarrier::invalidate(&stage.barrier);
    } else {
      naivert::attn_rows::project_rows_mma<Scratch::kQueryRows, Scratch::kDim, kGemvSplits, 1,
                                           kWmmaWarps>(
          0, Scratch::kIqBase / 16, row_ptr, x_norm, partials, emit);
    }
    if (project_index) {
      // Four 16-row FP8 tiles per CTA, with two K halves each: 32 CTAs for
      // 16 index heads or 128 CTAs for 64. The BF16 wis tiles use the idle CTAs.
      constexpr int kIndexCtas = Scratch::kIqRows / 16 / kIndexTilesPerCta;
      if (int(blockIdx.x) >= kIndexCtas) {
        naivert::attn_rows::project_rows_mma<Scratch::kQueryRows, Scratch::kDim, kGemvSplits, 1,
                                             kWmmaWarps>(
            Scratch::kFused, (kProjectionRows - Scratch::kFused) / 16, row_ptr, x_norm, partials,
            emit, int(blockIdx.x) - kIndexCtas);
      }
      if (args.wiq_bf16 != nullptr) {
        naivert::attn_rows::project_rows_mma<Scratch::kQueryRows, Scratch::kDim,
            kIndexGemvSplits, kIndexTilesPerCta, kWmmaWarps>(
                Scratch::kIqBase, Scratch::kIqRows / 16, row_ptr, x_norm, partials, emit);
      } else {
        convert_x8(x_norm, x8);
        project_index_fp8(args, x8, fused, partials);
      }
    }
  }

  // ------------------------------------ 2: rope, norms, cache append

  // q: rope on the first 64 dims of each of the 8 heads, written as
  // [kPaddedHeads, kHeadDim] with zero pad rows for the wmma N tile.
  __device__ static void publish_query(const Args& args,
                                       const __nv_bfloat16* fused,
                                       __nv_bfloat16* q_rot, int query,
                                       int position) {
    int tid = int(threadIdx.x);
    const __nv_bfloat16* source =
        fused + std::size_t(query) * Scratch::kFused + Scratch::kQBase;
    __nv_bfloat16* destination =
        q_rot + std::size_t(query) * Scratch::kPaddedHeads * Scratch::kHeadDim;
    const float* rope = args.rope_freqs + std::size_t(position) * Scratch::kRope;
    for (int pair = tid; pair < Scratch::kQ / 2; pair += kThreads) {
      int element = pair * 2;
      int dimension = element % Scratch::kHeadDim;
      float2 value = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(source + element));
      if (dimension < Scratch::kRope) {
        rotate_pair(value.x, value.y, rope[dimension], rope[dimension + 1]);
      }
      *reinterpret_cast<__nv_bfloat162*>(destination + element) =
          __floats2bfloat162_rn(value.x, value.y);
    }
    constexpr int kPadElements =
        (Scratch::kPaddedHeads - Scratch::kLocalHeads) * Scratch::kHeadDim;
    for (int element = tid; element < kPadElements; element += kThreads) {
      destination[Scratch::kQ + element] = __float2bfloat16(0.0f);
    }
  }

  // k (rope on the first 64 dims) and v straight into the caches.
  __device__ static void publish_kv(const Args& args, const __nv_bfloat16* fused,
                                    int query, int position) {
    int tid = int(threadIdx.x);
    const __nv_bfloat16* row = fused + std::size_t(query) * Scratch::kFused;
    const float* rope = args.rope_freqs + std::size_t(position) * Scratch::kRope;
    if (tid < Scratch::kHeadDim / 2) {
      int element = tid * 2;
      float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
          row + Scratch::kKBase + element));
      if (element < Scratch::kRope) {
        rotate_pair(value.x, value.y, rope[element], rope[element + 1]);
      }
      *reinterpret_cast<__nv_bfloat162*>(
          args.k + std::size_t(position) * Scratch::kHeadDim + element) =
          __floats2bfloat162_rn(value.x, value.y);
    } else if (tid < Scratch::kHeadDim / 2 + Scratch::kVDim / 2) {
      int element = (tid - Scratch::kHeadDim / 2) * 2;
      *reinterpret_cast<__nv_bfloat162*>(
          args.v + std::size_t(position) * Scratch::kVDim + element) =
          *reinterpret_cast<const __nv_bfloat162*>(row + Scratch::kVBase + element);
    }
  }

  // Index key: LayerNorm(weight, bias) -> bf16 -> rope on the first 64 dims
  // -> ki cache. Same rounding points as the bf16 serving path (LayerNorm
  // output and rotary output are both bf16 tensors there).
  __device__ static void publish_ki(const Args& args, const __nv_bfloat16* fused,
                                    int query, int position, float* reduction,
                                    float* ki_values) {
    int tid = int(threadIdx.x);
    const __nv_bfloat16* source =
        fused + std::size_t(query) * Scratch::kFused + Scratch::kIkBase;
    float value = tid < Scratch::kIndexDim ? bf16(source, tid) : 0.0f;
    float mean = block_sum(value, reduction) / float(Scratch::kIndexDim);
    // block_sum publishes shared[0] before returning. Every warp must read
    // that mean before the next reduction overwrites the same storage.
    __syncthreads();
    float delta = tid < Scratch::kIndexDim ? value - mean : 0.0f;
    float variance = block_sum(delta * delta, reduction) / float(Scratch::kIndexDim);
    float inverse_std = rsqrtf(variance + kNormEps);
    if (tid < Scratch::kIndexDim) {
      float normalized =
          delta * inverse_std * args.ik_norm_weight[tid] + args.ik_norm_bias[tid];
      ki_values[tid] = __bfloat162float(__float2bfloat16(normalized));
    }
    __syncthreads();
    // The roped row, rounded to bf16 as the bf16 cache held it, goes to the cache as
    // e4m3 with one scale per position (row amax / 448); the sweep scales the logit back.
    float real = 0.0f;
    float imaginary = 0.0f;
    if (tid < Scratch::kIndexDim / 2) {
      int element = tid * 2;
      real = ki_values[element];
      imaginary = ki_values[element + 1];
      if (element < Scratch::kRope) {
        const float* rope =
            args.rope_freqs + std::size_t(position) * Scratch::kRope;
        rotate_pair(real, imaginary, rope[element], rope[element + 1]);
      }
      real = __bfloat162float(__float2bfloat16(real));
      imaginary = __bfloat162float(__float2bfloat16(imaginary));
    }
    const float amax = block_max(fmaxf(fabsf(real), fabsf(imaginary)), reduction);
    const float scale = fmaxf(amax, args.wiq_bf16 != nullptr ? 1.0e-4f : 1.0e-30f) / 448.0f;
    if (tid < Scratch::kIndexDim / 2) {
      int element = tid * 2;
      const __nv_fp8x2_e4m3 packed(make_float2(real / scale, imaginary / scale));
      // UMMA K-major SW128 image: the 16-byte chunk index is XORed with (position & 7).
      *reinterpret_cast<unsigned short*>(
          args.ki + std::size_t(position) * Scratch::kIndexDim +
          (((element >> 4) ^ (position & 7)) << 4) + (element & 15)) = packed.__x;
    }
    if (tid == 0) {
      args.ki_scale[position] = scale;
    }
  }

  // Index queries: rope on the first 64 dims of each 128-vector, then the
  // packed layout the shared logits stage reads (packed_iq_index). One
  // 128-vector per 16-lane subgroup, eight contiguous elements per lane.
  __device__ static void rotate_index_queries(const Args& args,
                                              const __nv_bfloat16* fused,
                                              __nv_fp8_e4m3* packed_iq,
                                              const float* index_weights,
                                              float* index_weights_scaled, int base_pos) {
    int warp = int(threadIdx.x) >> 5;
    int lane = int(threadIdx.x) & 31;
    int vector_pair = (int(blockIdx.x) * kWmmaWarps + warp) * 2;
    constexpr int kVectors = Scratch::kQueryRows * Scratch::kIndexHeads;
    if (vector_pair >= kVectors) {
      return;
    }
    int subgroup = lane >> 4;
    int subgroup_lane = lane & 15;
    int vector = vector_pair + subgroup;
    int query = vector / Scratch::kIndexHeads;
    int head = vector - query * Scratch::kIndexHeads;
    const __nv_bfloat16* source = fused + std::size_t(query) * Scratch::kFused +
                                  Scratch::kIqBase + head * Scratch::kIndexDim +
                                  subgroup_lane * 8;
    uint4 raw = *reinterpret_cast<const uint4*>(source);
    unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    float values[8];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      float2 unpacked = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(words + pair));
      values[pair * 2] = unpacked.x;
      values[pair * 2 + 1] = unpacked.y;
    }
    if (subgroup_lane < Scratch::kRope / 8) {
      const float* rope =
          args.rope_freqs + std::size_t(base_pos + query) * Scratch::kRope;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        int dimension = subgroup_lane * 8 + pair * 2;
        rotate_pair(values[pair * 2], values[pair * 2 + 1], rope[dimension],
                    rope[dimension + 1]);
      }
    }
    // Bf16-rounded as before, then e4m3 with one scale per (row, head) that is folded
    // into this head's index weight; packed in the m16n8k32 B-fragment order.
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(__float2bfloat16(values[i]));
      amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float scale = fmaxf(amax, args.wiq_bf16 != nullptr ? 1.0e-4f : 1.0e-30f) / 448.0f;
    const __nv_fp8x4_e4m3 low(make_float4(values[0] / scale, values[1] / scale,
                                          values[2] / scale, values[3] / scale));
    const __nv_fp8x4_e4m3 high(make_float4(values[4] / scale, values[5] / scale,
                                           values[6] / scale, values[7] / scale));
    // The query image in the UMMA K-major SW128 form: column = (row, head), 128 bytes per
    // column, 16-byte chunk index XORed with (column & 7); this lane's eight dims are half a chunk.
    const int column = query * Scratch::kIndexHeads + head;
    const int dimension = subgroup_lane * 8;
    unsigned char* column_base = reinterpret_cast<unsigned char*>(packed_iq) +
                                 std::size_t(column) * Scratch::kIndexDim +
                                 (((dimension >> 4) ^ (column & 7)) << 4) + (dimension & 15);
    *reinterpret_cast<unsigned*>(column_base) = low.__x;
    *reinterpret_cast<unsigned*>(column_base + 4) = high.__x;
    if (subgroup_lane == 0) {
      index_weights_scaled[query * Scratch::kIndexHeads + head] =
          index_weights[query * Scratch::kIndexHeads + head] * scale;
    }
  }

  // ------------------------------------------- 4: split-KV attention

  // Stage this split's selected key / value rows (16-byte cp.async pieces, zero-filled
  // past `count`) and the query row into shared memory in one burst.
  __device__ static void stage_split(const Args& args, const __nv_bfloat16* q_rot,
                                     const int* selections, int query, int count,
                                     int tile_begin, int tokens, GatherStorage& g) {
    constexpr int kKeyPieces = Scratch::kHeadDim / 8;   // 24
    constexpr int kValuePieces = Scratch::kVDim / 8;    // 16
    const int tid = int(threadIdx.x);
    const int* row_selection = selections + query * Scratch::kIndexTopK + tile_begin;
    for (int i = tid; i < tokens * kKeyPieces; i += kThreads) {
      const int row = i / kKeyPieces;
      const int piece = i - row * kKeyPieces;
      const __nv_bfloat16* source = args.k;
      int bytes = 0;
      if (tile_begin + row < count) {
        source = args.k + std::size_t(row_selection[row]) * Scratch::kHeadDim + piece * 8;
        bytes = 16;
      }
      cp_async_16(g.keys + row * kGatherKeyStride + piece * 8, source, bytes);
    }
    for (int i = tid; i < tokens * kValuePieces; i += kThreads) {
      const int row = i / kValuePieces;
      const int piece = i - row * kValuePieces;
      const __nv_bfloat16* source = args.v;
      int bytes = 0;
      if (tile_begin + row < count) {
        source = args.v + std::size_t(row_selection[row]) * Scratch::kVDim + piece * 8;
        bytes = 16;
      }
      cp_async_16(g.values + row * kGatherValueStride + piece * 8, source, bytes);
    }
    const __nv_bfloat16* query_rows =
        q_rot + std::size_t(query) * Scratch::kPaddedHeads * Scratch::kHeadDim;
    for (int i = tid; i < Scratch::kPaddedHeads * kKeyPieces; i += kThreads) {
      const int head = i / kKeyPieces;
      const int piece = i - head * kKeyPieces;
      cp_async_16(g.query + head * kGatherKeyStride + piece * 8,
                  query_rows + head * Scratch::kHeadDim + piece * 8, 16);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();
  }


  // The eight real heads occupy N of m16n8k16; use CuTe's LDSM atoms.
  __device__ static void qk_eight_heads(GatherStorage& g, int tile, int lane) {
    float c[4] = {};
#pragma unroll
    for (int k0 = 0; k0 < Scratch::kHeadDim; k0 += 16) {
      unsigned a[4], b[2];
      const int matrix = lane >> 3;
      const auto* key = g.keys + tile * kTokenTile * kGatherKeyStride +
                        ((lane & 7) + ((matrix & 1) << 3)) * kGatherKeyStride +
                        k0 + ((matrix >> 1) << 3);
      const auto* query = g.query + (lane & 7) * kGatherKeyStride + k0 + (((lane >> 3) & 1) << 3);
      cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(key), a[0], a[1], a[2], a[3]);
      cute::SM75_U32x2_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(query), b[0], b[1]);
      mma_m16n8k16_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    }
    const int row = tile * kTokenTile + (lane >> 2), head = (lane & 3) * 2;
    g.scores[row * Scratch::kPaddedHeads + head] = c[0];
    g.scores[row * Scratch::kPaddedHeads + head + 1] = c[1];
    g.scores[(row + 8) * Scratch::kPaddedHeads + head] = c[2];
    g.scores[(row + 8) * Scratch::kPaddedHeads + head + 1] = c[3];
  }

  __device__ static void pv_eight_heads(GatherStorage& g, float* out, int split,
                                        int tiles, int warp, int lane) {
    float c[4] = {};
    for (int t = 0; t < tiles; ++t) {
      unsigned a[4], b[2];
      const int matrix = lane >> 3;
      // A is V transposed: M = 16 value dimensions, K = 16 tokens.
      const auto* value = g.values + (t * kTokenTile + (lane & 7) + ((matrix >> 1) << 3)) *
                         kGatherValueStride + warp * 16 + ((matrix & 1) << 3);
      // B is P transposed, physically stored [heads][tokens].
      const auto* probability = g.probabilities + t * Scratch::kPaddedHeads * kTokenTile +
                                (lane & 7) * kTokenTile + (((lane >> 3) & 1) << 3);
      cute::SM75_U16x8_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(value), a[0], a[1], a[2], a[3]);
      cute::SM75_U32x2_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(probability), b[0], b[1]);
      mma_m16n8k16_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    }
    const int dimension = warp * 16 + (lane >> 2), head = (lane & 3) * 2;
    out[(split * Scratch::kPaddedHeads + head) * Scratch::kVDim + dimension] = c[0];
    out[(split * Scratch::kPaddedHeads + head + 1) * Scratch::kVDim + dimension] = c[1];
    out[(split * Scratch::kPaddedHeads + head) * Scratch::kVDim + dimension + 8] = c[2];
    out[(split * Scratch::kPaddedHeads + head + 1) * Scratch::kVDim + dimension + 8] = c[3];
  }

  __device__ static void compute_attention(
      const Args& args, const __nv_bfloat16* q_rot, const int* selections,
      __nv_bfloat16* attention_o, int base_pos, float* attention_temp,
      unsigned* selection_ready, unsigned* split_ready, unsigned* output_ready,
      bool wait_for_selection, CombineWorkspace* combine_workspace,
      GatherStorage* gather) {
    int tid = int(threadIdx.x);
    int warp = tid >> 5;
    int lane = tid & 31;
    int split_worker = int(blockIdx.x);
    if (wait_for_selection &&
        split_worker < Scratch::kQueryRows * kAttentionSplits) {
      int query = split_worker / kAttentionSplits;
      if (tid == 0) {
        while (atomicAdd(selection_ready + query, 0u) == 0u) {
          __nanosleep(64);
        }
      }
      __syncthreads();
    }
    if (split_worker < Scratch::kQueryRows * kAttentionSplits) {
      int query = split_worker / kAttentionSplits;
      int split = split_worker - query * kAttentionSplits;
      int visible = base_pos + query + 1;
      int count = visible < Scratch::kIndexTopK ? visible : Scratch::kIndexTopK;
      int token_tiles = (count + kTokenTile - 1) / kTokenTile;
      int desired_splits = count <= 512 ? 8 : kAttentionSplits;
      int active_splits =
          token_tiles < desired_splits ? token_tiles : desired_splits;
      if (split < active_splits) {
        int tile_begin = (token_tiles * split / active_splits) * kTokenTile;
        int tile_end = (token_tiles * (split + 1) / active_splits) * kTokenTile;
        if (tile_end > count) {
          tile_end = (count + kTokenTile - 1) / kTokenTile * kTokenTile;
        }
        float* query_workspace =
            attention_temp + std::size_t(query) * Scratch::kAttentionTempStride;
        float* split_outputs = query_workspace + kSplitOutputOffset;
        float* split_stats = query_workspace + kSplitStatsOffset;
        const int tokens = tile_end - tile_begin;          // whole tiles, <= kSplitTokens
        const int tiles = tokens / kTokenTile;
        GatherStorage& g = *gather;
        // One staging burst, then QK / softmax / PV from shared memory.
        stage_split(args, q_rot, selections, query, count, tile_begin, tokens, g);
        // QK: warp t owns token tile t (all 12 K steps), scores [token][16 heads] fp32.
        if (warp < tiles) {
          if constexpr (Scratch::kQueryRows == 8 && Scratch::kIndexHeads == 16) {
            qk_eight_heads(g, warp, lane);
          } else {
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accumulator;
          nvcuda::wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
          for (int k0 = 0; k0 < Scratch::kHeadDim; k0 += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::row_major> key_fragment;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::col_major> query_fragment;
            nvcuda::wmma::load_matrix_sync(key_fragment,
                                           g.keys + warp * kTokenTile * kGatherKeyStride + k0,
                                           kGatherKeyStride);
            nvcuda::wmma::load_matrix_sync(query_fragment, g.query + k0, kGatherKeyStride);
            nvcuda::wmma::mma_sync(accumulator, key_fragment, query_fragment, accumulator);
          }
          nvcuda::wmma::store_matrix_sync(g.scores + warp * kTokenTile * Scratch::kPaddedHeads,
                                          accumulator, Scratch::kPaddedHeads,
                                          nvcuda::wmma::mem_row_major);
        }
        }
        __syncthreads();
        record_phase(attention_temp, 12);
        float attention_scale = rsqrtf(float(Scratch::kHeadDim));
        if (args.attention_scale_multiplier != 0.0f) {
          attention_scale *= args.attention_scale_multiplier;
        }
        const int local_limit = (tile_end < count ? tile_end : count) - tile_begin;
        // Softmax statistics: warps 0..3 own two heads each (lane parity), lanes 0..15 /
        // 16..31 walk the tokens of a 16-token tile; probabilities land as per-tile
        // [16 heads][16 tokens] bf16 tiles (pad heads zero).
        if (warp < Scratch::kLocalHeads / 2) {
          int head = warp * 2 + (lane & 1);
          int token_sixteenth = lane >> 1;
          float maximum = -3.402823466e+38f;
          for (int t = token_sixteenth; t < local_limit; t += 16) {
            maximum = fmaxf(maximum, g.scores[t * Scratch::kPaddedHeads + head] * attention_scale);
          }
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 16));
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 8));
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 4));
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
          float sum = 0.0f;
          for (int t = token_sixteenth; t < tokens; t += 16) {
            __nv_bfloat16 rounded = __float2bfloat16(0.0f);
            if (t < local_limit) {
              float probability = expf(g.scores[t * Scratch::kPaddedHeads + head] * attention_scale - maximum);
              rounded = __float2bfloat16(probability);
              sum += __bfloat162float(rounded);
            }
            g.probabilities[(t >> 4) * (Scratch::kPaddedHeads * kTokenTile) + head * kTokenTile +
                            (t & 15)] = rounded;
          }
          sum += __shfl_xor_sync(0xffffffffu, sum, 16);
          sum += __shfl_xor_sync(0xffffffffu, sum, 8);
          sum += __shfl_xor_sync(0xffffffffu, sum, 4);
          sum += __shfl_xor_sync(0xffffffffu, sum, 2);
          if (lane < 2) {
            int stat = (split * Scratch::kLocalHeads + head) * 2;
            split_stats[stat + 0] = maximum;
            split_stats[stat + 1] = sum;
          }
        } else if (warp < Scratch::kLocalHeads) {
          // Zero the pad-head rows of every probability tile.
          const int pad_head = Scratch::kLocalHeads + (warp - Scratch::kLocalHeads / 2) * 2 + (lane & 1);
          for (int t = lane >> 1; t < tokens; t += 16) {
            g.probabilities[(t >> 4) * (Scratch::kPaddedHeads * kTokenTile) + pad_head * kTokenTile +
                            (t & 15)] = __float2bfloat16(0.0f);
          }
        }
        __syncthreads();
        record_phase(attention_temp, 13);
        // PV: warp w owns output block w (16 value dims): P tile [heads x tokens] x V tile
        // [tokens x dims] over the split's token tiles.
        constexpr int kOutputBlocks = Scratch::kVDim / 16;
        if (warp < kOutputBlocks) {
          if constexpr (Scratch::kQueryRows == 8 && Scratch::kIndexHeads == 16) {
            pv_eight_heads(g, split_outputs, split, tiles, warp, lane);
          } else {
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> accumulator;
          nvcuda::wmma::fill_fragment(accumulator, 0.0f);
          for (int t = 0; t < tiles; ++t) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::row_major> probability_fragment;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::row_major> value_fragment;
            nvcuda::wmma::load_matrix_sync(probability_fragment,
                                           g.probabilities + t * (Scratch::kPaddedHeads * kTokenTile),
                                           kTokenTile);
            nvcuda::wmma::load_matrix_sync(value_fragment,
                                           g.values + t * kTokenTile * kGatherValueStride + warp * 16,
                                           kGatherValueStride);
            nvcuda::wmma::mma_sync(accumulator, probability_fragment, value_fragment, accumulator);
          }
          nvcuda::wmma::store_matrix_sync(
              split_outputs + std::size_t(split) * Scratch::kPaddedHeads * Scratch::kVDim + warp * 16,
              accumulator, Scratch::kVDim, nvcuda::wmma::mem_row_major);
        }
        }
        record_phase(attention_temp, 14);
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          atomicAdd(split_ready + query, 1u);
        }
      }
    }
    record_phase(attention_temp, 15);
    // Combine: one CTA per (query, head, value-dim half).
    int combine_worker = int(blockIdx.x);
    int combine_workers = Scratch::kQueryRows * Scratch::kLocalHeads * 2;
    if (combine_worker < combine_workers) {
      int query_head = combine_worker >> 1;
      int dimension_half = combine_worker & 1;
      int query = query_head / Scratch::kLocalHeads;
      int head = query_head - query * Scratch::kLocalHeads;
      int visible = base_pos + query + 1;
      int count = visible < Scratch::kIndexTopK ? visible : Scratch::kIndexTopK;
      int token_tiles = (count + kTokenTile - 1) / kTokenTile;
      int desired_splits = count <= 512 ? 8 : kAttentionSplits;
      int active_splits =
          token_tiles < desired_splits ? token_tiles : desired_splits;
      if (tid == 0) {
        while (atomicAdd(split_ready + query, 0u) < unsigned(active_splits)) {
          __nanosleep(64);
        }
      }
      __syncthreads();
      float* query_workspace =
          attention_temp + std::size_t(query) * Scratch::kAttentionTempStride;
      float* split_outputs = query_workspace + kSplitOutputOffset;
      float* split_stats = query_workspace + kSplitStatsOffset;
      constexpr int kDimensionsPerHalf = Scratch::kVDim / 2;
      constexpr int kVectorsPerSplit = kDimensionsPerHalf / 4;
      int dimension_begin = dimension_half * kDimensionsPerHalf;
      int dimension_end = dimension_begin + kDimensionsPerHalf;
      int vector_count = active_splits * kVectorsPerSplit;
      for (int vector = tid; vector < vector_count; vector += kThreads) {
        int split = vector / kVectorsPerSplit;
        int split_vector = vector - split * kVectorsPerSplit;
        int dimension = dimension_begin + split_vector * 4;
        cp_async_16(
            combine_workspace->values + split * kDimensionsPerHalf + split_vector * 4,
            split_outputs +
                (std::size_t(split) * Scratch::kPaddedHeads + head) * Scratch::kVDim +
                dimension,
            16);
      }
      cp_async_wait_all();
      float* corrections = combine_workspace->corrections;
      if (warp == 0) {
        float global_max =
            lane < active_splits
                ? split_stats[(lane * Scratch::kLocalHeads + head) * 2]
                : -3.402823466e+38f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          global_max = fmaxf(global_max,
                             __shfl_down_sync(0xffffffffu, global_max, offset));
        }
        global_max = __shfl_sync(0xffffffffu, global_max, 0);
        if (lane < active_splits) {
          corrections[lane] =
              expf(split_stats[(lane * Scratch::kLocalHeads + head) * 2] - global_max);
        }
        __syncwarp();
        if (lane == 0) {
          float denominator = 0.0f;
          for (int split = 0; split < active_splits; ++split) {
            int stat = (split * Scratch::kLocalHeads + head) * 2;
            denominator = fmaf(split_stats[stat + 1], corrections[split], denominator);
          }
          corrections[kAttentionSplits] = denominator;
        }
      }
      __syncthreads();
      float denominator = corrections[kAttentionSplits];
      for (int dimension0 = dimension_begin + warp * 32; dimension0 < dimension_end;
           dimension0 += kWmmaWarps * 32) {
        int dimension = dimension0 + lane;
        if (dimension < dimension_end) {
          float value = 0.0f;
#pragma unroll
          for (int split = 0; split < active_splits; ++split) {
            value = fmaf(combine_workspace->values[split * kDimensionsPerHalf +
                                                   dimension - dimension_begin],
                         corrections[split], value);
          }
          attention_o[(std::size_t(query) * Scratch::kLocalHeads + head) *
                          Scratch::kVDim +
                      dimension] =
              __float2bfloat16(denominator > 0.0f ? value / denominator : 0.0f);
        }
      }
      record_phase(attention_temp, 16);
      __syncthreads();
      if (tid == 0) {
        __threadfence();
        atomicAdd(output_ready + head, 1u);
      }
    }
  }

  // ------------------------------------------------------------ driver

  __device__ static bool arguments_valid(const Args& args, int base_pos) {
    return args.workspace != nullptr && args.residual != nullptr &&
           args.input_norm != nullptr && args.wq != nullptr && args.wk != nullptr &&
           args.wv != nullptr && (args.wiq != nullptr || args.wiq_bf16 != nullptr) && args.wiq_scale != nullptr &&
           args.wik != nullptr &&
           args.ik_norm_weight != nullptr && args.ik_norm_bias != nullptr &&
           args.wis != nullptr && args.rope_freqs != nullptr && args.k != nullptr &&
           args.v != nullptr && args.ki != nullptr && args.ki_scale != nullptr &&
           args.proj_o != nullptr &&
           base_pos >= 0 && args.max_seq_len <= Scratch::kMaxSeqLen &&
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
    __shared__ SharedStorage shared_storage;
    __shared__ alignas(16) __nv_bfloat16 x_norm[Scratch::kQueryRows * Scratch::kDim];
    auto* reduction = shared_storage.workspace.reduction;
    auto* ki_values = shared_storage.workspace.ki_values;
    auto* attention_scores = shared_storage.workspace.attention_scores;
    auto* projection_tile = &shared_storage.workspace.projection_tile;
    auto* topk_sort_storage = &shared_storage.topk_sort;
    auto* topk_scratch = reinterpret_cast<unsigned*>(&shared_storage);
    auto* combine_workspace = &shared_storage.combine;

    auto* fused = reinterpret_cast<__nv_bfloat16*>(args.workspace + Scratch::kFusedOffset);
    auto* q_rot = reinterpret_cast<__nv_bfloat16*>(args.workspace + Scratch::kQRotOffset);
    auto* packed_iq =
        reinterpret_cast<__nv_fp8_e4m3*>(args.workspace + Scratch::kIqPackedOffset);
    auto* index_weights =
        reinterpret_cast<float*>(args.workspace + Scratch::kIndexWeightsOffset);
    auto* index_weights_scaled =
        reinterpret_cast<float*>(args.workspace + Scratch::kIndexWeightsScaledOffset);
    auto* logits = reinterpret_cast<float*>(args.workspace + Scratch::kIndexLogitsOffset);
    auto* selections = reinterpret_cast<int*>(args.workspace + Scratch::kIndexSelectsOffset);
    auto* radix = reinterpret_cast<RadixControl*>(args.workspace + Scratch::kRadixControlOffset);
    auto* attention_temp =
        reinterpret_cast<float*>(args.workspace + Scratch::kAttentionTempOffset);
    auto* attention_output_ready =
        reinterpret_cast<unsigned*>(attention_temp + kPhaseTimerFloatOffset + 64);
    auto* attention_split_ready = attention_output_ready + Scratch::kLocalHeads;
    auto* topk_grid = reinterpret_cast<unsigned*>(args.workspace + Scratch::kTopKGridOffset);

    bool project_index = base_pos + Scratch::kQueryRows > Scratch::kIndexTopK;
    // The near-top-k regime (2049..2056 visible) also goes through the grid select, whose
    // per-query local path handles it in one CTA; the shared-stage sort path stays for other geometries.
    constexpr bool kUseNearTopKStage = false;
    bool near_topk = kUseNearTopKStage && project_index &&
                     base_pos + Scratch::kQueryRows <=
                         Scratch::kIndexTopK + Scratch::kQueryRows;

    auto grid = cg::this_grid();
    if (blockIdx.x == 0) {
      if (threadIdx.x < Scratch::kQueryRows) {
        radix->prefix[threadIdx.x] = 0u;
        radix->remaining[threadIdx.x] = 0u;
        radix->threshold[threadIdx.x] = 0u;
        attention_split_ready[threadIdx.x] = 0u;
      }
      if (threadIdx.x < kFusedReadyShards) {
        radix->fused_ready[threadIdx.x] = 0u;
      }
      if (threadIdx.x < Scratch::kLocalHeads) {
        attention_output_ready[threadIdx.x] = 0u;
      }
      if (threadIdx.x == 0) {
        radix->qnorm_ready = 0u;
      }
    }
    if (project_index && !near_topk) {
      // The digit histograms of the grid-parallel top-k must be zero before the
      // selection passes; every CTA zeroes a slice, ordered by the projection barrier.
      zero_topk_grid(topk_grid);
    }
    record_phase(attention_temp, 0);
    // The flag reset above is first read after the barrier that follows the projection
    // (publish / rotate, then the attention phase), and every CTA normalises into its own
    // shared x_norm: no grid barrier between RMSNorm and the projection.
    // normalize_x_local writes only x_norm, not its unused reduction argument:
    // reduction aliases the in-flight weight image here. Its final CTA join
    // publishes the transaction-barrier initialization before project_fused waits.
    if constexpr (Scratch::kUseProjectionTma)
      issue_projection(args, shared_storage.projection_stage);
    normalize_x_local(args, x_norm, reduction);
    record_phase(attention_temp, 1);

    // The projection tile's accumulator area is free until the attention phase.
    project_fused(args, x_norm, fused, index_weights, project_index,
                  &projection_tile->c[0][0], shared_storage.workspace.x8, shared_storage.projection_stage);
    grid.sync();
    record_phase(attention_temp, 2);

    // Publishers: one CTA per (row, stage). Meanwhile the grid either writes
    // the identity selection (dense regime) or rotates the index queries.
    {
      int worker = int(blockIdx.x);
      constexpr int kPublishStages = 3;
      if (worker < Scratch::kQueryRows * kPublishStages) {
        int query = worker / kPublishStages;
        int stage = worker - query * kPublishStages;
        int position = base_pos + query;
        if (stage == 0) {
          publish_query(args, fused, q_rot, query, position);
        } else if (stage == 1) {
          publish_kv(args, fused, query, position);
        } else {
          publish_ki(args, fused, query, position, reduction, ki_values);
        }
      }
      if (project_index) {
        rotate_index_queries(args, fused, packed_iq, index_weights, index_weights_scaled, base_pos);
      } else {
        publish_identity_selections(selections, base_pos);
      }
    }
    grid.sync();
    record_phase(attention_temp, 3);

    if (project_index) {
      // The sweep on tcgen05; keys and queries are UMMA K-major SW128 images (e4m3).
      compute_index_logits_tcgen05(args, reinterpret_cast<const unsigned char*>(packed_iq),
                                   index_weights_scaled, logits, base_pos, shared_storage.sweep);
      grid.sync();
      record_phase(attention_temp, 4);   // sweep done (the grid path overwrites 4 after its pass 0)
      if (near_topk) {
        select_per_query_topk(logits, selections, radix, base_pos, attention_scores,
                              attention_temp, topk_sort_storage, topk_scratch,
                              projection_tile);
      } else {
        // Exact top-k by grid-wide radix select (two digit passes, compaction,
        // one CTA per query finishes); selections in ascending token order for any number
        // of visible tokens.
        select_per_query_topk_grid(logits, selections, topk_grid, base_pos, attention_temp,
                                   topk_scratch, topk_sort_storage);
      }
    }
    record_phase(attention_temp, 8);

    compute_attention(args, q_rot, selections, args.proj_o, base_pos, attention_temp,
                      radix->threshold, attention_split_ready, attention_output_ready,
                      near_topk, combine_workspace, &shared_storage.gather);
    record_phase(attention_temp, 9);
    if (args.unproj.o_weight != nullptr) {
      // O-projection + TP8 all-reduce as the kernel's last phase (unproj_tail.cuh);
      // the attention output is complete after the barrier, x_norm's shared memory is free.
      grid.sync();
      using Tail = naivert::attn_unproj::UnprojTail<Scratch::kQueryRows, Scratch::kDim,
                                                    Scratch::kLocalHeads * Scratch::kVDim,
                                (Scratch::kQueryRows == 8), (Scratch::kQueryRows == 8 ? kThreads : 0)>;   // As the SWA tail (R54)
      static_assert(Tail::kSharedBytes <= int(sizeof(x_norm)),
                    "the fused o-projection reuses x_norm's shared memory");
      Tail::run(args.unproj, args.proj_o, args.residual,
                reinterpret_cast<unsigned char*>(x_norm));
    }
  }
};

using AttnDsaBody = AttnDsaBodyImpl<false>;

}  // namespace naivert::attn_dsa
