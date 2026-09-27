#pragma once
#include "naivert/body/dspark_context/fc_body.cuh"
#include "naivert/body/dspark_context/kv_body.cuh"
#include "naivert/pdl/grid_dependency.cuh"

namespace naivert_dspark_ops {
constexpr int kD12Splits = 8;
constexpr int kD12CtasPerSplit = kCtas / kD12Splits;
static_assert(kCtas % kD12Splits == 0);
static_assert((kTotalCols / 16) % kD12CtasPerSplit == 0);
constexpr int kD12Stride = kHidden + 32;
constexpr int kD12Scratch = kWindow * kD12Stride * 2 +
    (2 * kWindow * kColsPerCta + kWindow + kShellWarps * 128) * sizeof(float);
constexpr int kD12WeightOffset = (kD12Scratch + 127) & ~127;
constexpr int kD12WeightStride = kHidden / kD12Splits + 32;
constexpr int kD12WeightRows = kTotalCols / kD12CtasPerSplit;
constexpr int kD12WeightBytes = kD12WeightRows * kD12WeightStride * 2;
struct D12FinishCoefficients {
  float gamma[kKvRowsR], cosine[kHeadDim], sine[kHeadDim];
  std::int64_t destination;
};
constexpr int kD12FinishOffset = kWindow * kD12WeightStride * 2 + kShellWarps * 128 * sizeof(float);
static_assert(kD12FinishOffset + sizeof(D12FinishCoefficients) <= kD12WeightOffset);
static_assert(kThreads - kWindow * 32 == kHeadDim && kHeadDim == kKvRowsR);
struct FcCtxKvBody {
  __device__ static void run(unsigned char* smem_raw, const FcCtxKvArgs& args) {
    static_assert(kCtas == kCtasD2);
    // D12 has no programmatic predecessor (the tail launches it). Trigger its dependent at once
    // so the first D3's CTAs launch into the spare SMs and run their prologue during this
    // kernel; D3 waits for D12's full completion before touching its output. No-op without
    // a programmatic edge out of this kernel.
    naivert::pdl::trigger_dependents<true>();
    // Each producer publishes BF16 reduced values plus its SSQ tile. The
    // original full-grid barrier inside D1 makes ALL 4096 columns ready.
    constexpr int split_k = kHidden / kD12Splits;
    constexpr int query_stride = kD12WeightStride;
    constexpr int tiles_per_cta = (kTotalCols / 16) / kD12CtasPerSplit;
    constexpr int partial_elements = kLayers * 2 * kWindow * kKvRowsR;
    const int split = int(blockIdx.x) / kD12CtasPerSplit;
    const int bucket = int(blockIdx.x) % kD12CtasPerSplit;
    const int k_begin = split * split_k;
    const int tid = int(threadIdx.x), row = tid / 32, lane = tid % 32;
    auto* weights_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + kD12WeightOffset);
    FcCtxBody::run<true>(smem_raw, args.fc, [&] {
      // Stage KV weights after FC MMA; transport may progress during TP8/RMSNorm.
      // Each copy reads one aligned 16-byte K segment of an original weight row.
      int pending = 0;
      for (int i = tid; i < kD12WeightRows * (split_k / 8); i += kThreads) {
        const int local_row = i / (split_k / 8), k = (i % (split_k / 8)) * 8;
        const int column = bucket * kD12WeightRows + local_row;
        const int layer = column / kColsPerLayer, kv = column % kColsPerLayer;
        const auto* source = (kv >= kKvRowsR ? args.kv.v_w[layer] : args.kv.k_w[layer]) +
            std::size_t(kv % kKvRowsR) * kHidden + k_begin + k;
        auto* destination = weights_s + local_row * kD12WeightStride + k;
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(destination, source);
        if (++pending == 8) { cutlass::arch::cp_async_fence(); pending = 0; }
      }
      cutlass::arch::cp_async_fence();
    });
    auto* ctx_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* finish = reinterpret_cast<D12FinishCoefficients*>(smem_raw + kD12FinishOffset);
    // These four warps do not normalize query rows. Immutable finish inputs
    // are staged while the other eight warps perform the complete SSQ fold.
    if (row >= kWindow && int(blockIdx.x) < kFinishCtas) {
      const int component = tid - kWindow * 32;
      const int finish_layer = int(blockIdx.x) / kWindow, finish_query = int(blockIdx.x) % kWindow;
      const auto position = args.kv.ctx_pos[finish_query];
      const std::size_t rp = std::size_t(position) * kHeadDim + component;
      finish->gamma[component] = args.kv.k_norm[finish_layer][component];
      finish->cosine[component] = args.kv.rope_cos[rp];
      finish->sine[component] = args.kv.rope_sin[rp];
      if (component == 0) finish->destination = args.kv.kv_dst[finish_query];
    }
    if (row < kWindow) {
      // Every lane sees the same 128 SSQ values and retains their original order.
      // Replicated loads trade register/request pressure for 32 dependent shuffles.
      float4 pieces[32];
#pragma unroll
      for (int owner = 0; owner < 32; ++owner)
        pieces[owner] = reinterpret_cast<const float4*>(args.fc.ssq + row * kCtas)[owner];
      float sum = 0.0f;
#pragma unroll
      for (int owner = 0; owner < 32; ++owner) {
        sum = __fadd_rn(sum, pieces[owner].x); sum = __fadd_rn(sum, pieces[owner].y);
        sum = __fadd_rn(sum, pieces[owner].z); sum = __fadd_rn(sum, pieces[owner].w);
      }
      const float inverse_rms = rsqrtf(sum / float(kHidden) + args.fc.rms_epsilon);
      const bool valid = __bfloat162float(args.fc.ctx_valid[row]) != 0.0f;
#pragma unroll
      for (int piece = 0; piece < split_k / 256; ++piece) {
        const int n = (piece * 32 + lane) * 8;
        const int column = k_begin + n;
        const uint4 in = *reinterpret_cast<const uint4*>(args.fc.ctx_sum + row * kHidden + column);
        const float4 lo = *reinterpret_cast<const float4*>(args.fc.hidden_norm + column);
        const float4 hi = *reinterpret_cast<const float4*>(args.fc.hidden_norm + column + 4);
        const float gamma[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const unsigned words[4] = {in.x, in.y, in.z, in.w};
        uint4 out; auto* packed = reinterpret_cast<unsigned*>(&out);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
          const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + word));
          const float a = (value.x * inverse_rms) * gamma[2 * word];
          const float b = (value.y * inverse_rms) * gamma[2 * word + 1];
          const auto normalized = __floats2bfloat162_rn(valid ? a : 0.0f, valid ? b : 0.0f);
          packed[word] = *reinterpret_cast<const unsigned*>(&normalized);
        }
        *reinterpret_cast<uint4*>(ctx_s + row * query_stride + n) = out;
        if constexpr (kExportDiagnostics) {
          if (bucket == 0) *reinterpret_cast<uint4*>(args.fc.ctx + row * kHidden + column) = out;
        }
      }
    }
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    auto* scratch = reinterpret_cast<float*>(smem_raw + kWindow * query_stride * 2);
    // One warp owns one of five 16-column tiles for this 512-K partition.
    // The eight partitions are exactly the original eight MMA K splits.
    naivert::attn_rows::project_strided_rows_mma<kWindow, split_k, 1, tiles_per_cta,
        kShellWarps, query_stride, true, true>(bucket * tiles_per_cta * 16, tiles_per_cta,
        [&](int column) {
          return weights_s + (column - bucket * kD12WeightRows) * kD12WeightStride;
        }, ctx_s, scratch,
        [&](int column, int query, float value) {
          const int layer = column / kColsPerLayer, kv = column % kColsPerLayer;
          const int offset = ((layer * 2 + kv / kKvRowsR) * kWindow + query) * kKvRowsR + kv % kKvRowsR;
          args.kv_partials[split * partial_elements + offset] = value;
        }, 0);
    grid_barrier(args.fc.barrier);
    if (int(blockIdx.x) >= kFinishCtas) return;
    const int layer = int(blockIdx.x) / kWindow;
    const int query = int(blockIdx.x) % kWindow;
    const bool is_k = tid < kKvRowsR;
    const bool is_v = tid >= kKvRowsR && tid < 2 * kKvRowsR;
    __nv_bfloat16 raw = __float2bfloat16_rn(0.0f);
    if (is_k || is_v) {
      const int offset = ((layer * 2 + int(is_v)) * kWindow + query) * kKvRowsR + tid % kKvRowsR;
      float value = 0.0f;
#pragma unroll
      for (int s = 0; s < kD12Splits; ++s) value = __fadd_rn(value, args.kv_partials[s * partial_elements + offset]);
      raw = __float2bfloat16_rn(value);
      if constexpr (kExportDiagnostics) args.kv.kv_raw[offset] = raw;
    }
    const std::size_t out = (std::size_t(layer) * args.kv.cache_rows + std::size_t(finish->destination)) * kKvRowsR;
    if (is_v) args.kv.kv_v[out + tid - kKvRowsR] = raw;
    const float x = is_k ? __bfloat162float(raw) : 0.0f;
    float square = x * x;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) square += __shfl_xor_sync(0xffffffffu, square, off);
    if (lane == 0) scratch[tid / 32] = square;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int warp = 0; warp < kKvRowsR / 32; ++warp) total += scratch[warp];
    const float scale = rsqrtf(total / float(kHeadDim) + args.kv.rms_epsilon);
    auto* head = reinterpret_cast<__nv_bfloat16*>(scratch + kShellWarps);
    if (is_k) head[tid] = __float2bfloat16_rn((x * scale) * finish->gamma[tid]);
    __syncthreads();
    if (is_k) {
      const float value = __bfloat162float(head[tid]);
      const float rotated = tid < kHeadDim / 2 ? -__bfloat162float(head[tid + kHeadDim / 2]) : __bfloat162float(head[tid - kHeadDim / 2]);
      args.kv.kv_k[out + tid] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(value, finish->cosine[tid]), __fmul_rn(rotated, finish->sine[tid])));
    }
  }
};
}  // namespace naivert_dspark_ops
