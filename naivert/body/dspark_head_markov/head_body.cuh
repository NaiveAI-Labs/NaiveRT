#pragma once
#include "naivert/body/dspark_head_markov/args.hpp"
// D5: existing rows normalization, target-head TMA transport and FP32 MMA
// with direct body dispatch. Weights are packed as [tile16, Kpage1024, K32page, row16, k32].
// Eight consumer warps and four helper warps share four weight panes.
// Shared query rows have 32 BF16 padding elements; exported norm stays contiguous.
// The FP32 projection grouping differs from the four-split rows primitive;
// independent FP64 checks and the frozen draft token columns define accuracy.
// Pipeline failures use the same per-runtime watchdog as the other draft ops.
#include "naivert/body/attn_common/staged_rows_norm.cuh"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_rows_body.cuh"
#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/body/dspark_common/diagnostics.hpp"

namespace naivert_dspark_ops {
constexpr int kHeadWindow = 8;
constexpr int kHeadHidden = 4096;
constexpr int kQueryStride = kHeadHidden + 32;
constexpr int kHeadVocab = 19072;
constexpr int kHeadCtas = NAIVERT_SM_COUNT;
constexpr int kHeadThreads = 384;
constexpr int kTiles = kHeadVocab / 16;
constexpr int kPageK = 1024;
constexpr int kPageBytes = 16 * kPageK * 2;
constexpr int kBarrierOffset = kHeadWindow * kQueryStride * 2;
constexpr int kPartialOffset = kBarrierOffset + 8 * sizeof(std::uint64_t);
constexpr int kPaneOffset = (kPartialOffset + 8 * 128 * 4 + 127) & ~127;
constexpr int kGammaOffset = kPaneOffset + 4 * kPageBytes;
constexpr int kGammaBytes = kHeadHidden * sizeof(float);
constexpr int kGammaReadyOffset = kGammaOffset + kGammaBytes;
constexpr int kHeadSmem = (kGammaReadyOffset + sizeof(std::uint64_t) + 127) & ~127;
static_assert(kHeadThreads == 384);
static_assert(kHeadHidden == 4 * kPageK);
static_assert(kHeadSmem <= 232448);


static_assert(sizeof(HeadArgs) == 64, "HeadArgs must match the shared parameter contract");
using Transport =
    naivert_replica_rmsnorm_head_proj::RMSNormHeadProjRows8Body;
// This diagnostic id is not a mailbox publication tag: D5 still has no epoch.
constexpr unsigned kHeadPipelineWatchdogSite = 0xD5u;
__device__ bool wait_pane(std::uint64_t* barrier, unsigned phase,
                         const naivert::MailboxWatchdog& watchdog) {
  if (naivert::Barrier::try_wait(barrier, phase)) return true;
  const bool bounded = watchdog.limit_ns != 0ull;
  const auto start = bounded ? naivert::mailbox_globaltimer_ns() : 0ull;
  unsigned spins = 0;
  for (;;) {
    if (naivert::Barrier::try_wait(barrier, phase)) return true;
    if (bounded && ((++spins & naivert::kMailboxWatchdogSpinMask) == 0u) &&
        naivert::mailbox_globaltimer_ns() - start > watchdog.limit_ns) {
      naivert::mailbox_watchdog_trip(watchdog, kHeadPipelineWatchdogSite);
      // The caller must stop advancing this pane. Arriving on an incomplete
      // phase again can underflow its arrival count and raise a GPU exception.
      return false;
    }
  }
}
struct HeadBody {
  template<bool Fused, int LogitStride = 144>
  __device__ static void run(unsigned char* memory, const HeadArgs& args) {
    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (tid == 0) {
      naivert::Barrier::init(reinterpret_cast<std::uint64_t*>(memory + kGammaReadyOffset), 1u);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();  // Gamma-barrier initialization join.
    auto* x = reinterpret_cast<__nv_bfloat16*>(memory);
    auto* ready = reinterpret_cast<std::uint64_t*>(memory + kBarrierOffset);
    auto* released = ready + 4;
    auto* partials = reinterpret_cast<float*>(memory + kPartialOffset);
    if (tid < 4) {
      naivert::Barrier::init(ready + tid, 1u);
      naivert::Barrier::init(released + tid, 8u);
      naivert::barrier_arrive_count(released + tid, 8u);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    auto* gamma = reinterpret_cast<float*>(memory + kGammaOffset);
    auto* gamma_ready = reinterpret_cast<std::uint64_t*>(memory + kGammaReadyOffset);
    if (tid == 256) {
      naivert::Barrier::expect_transaction(gamma_ready, kGammaBytes);
      Transport::cp_async_bulk_evict_first(gamma, args.norm_gamma, kGammaBytes, gamma_ready);
      naivert::Barrier::arrive(gamma_ready);
    }
    // Programmatic dependent launch (D4 -> D56 edge in the round graph): everything above and the
    // first head pane per page depend only on weights, so they are issued before the wait; the
    // normalization below is the first read of the predecessor's output. `griddepcontrol.wait` is a
    // no-op when the kernel is launched without a programmatic edge. Bytes and order are unchanged.
    if (tid >= 256) {
      const int page = warp - 8;
      if (int(blockIdx.x) < kTiles && lane == 0) {
        naivert::Barrier::expect_transaction(ready + page, kPageBytes);
        Transport::cp_async_bulk_evict_first(
            memory + kPaneOffset + page * kPageBytes,
            args.lm_head + (std::size_t(blockIdx.x) * 4 + page) * 16 * kPageK,
            kPageBytes, ready + page);
        naivert::Barrier::arrive(ready + page);
      }
      __syncwarp();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    naivert::attn_rows::normalize_staged_rows<8, 4096, kQueryStride, true>(
        args.resid_in, gamma, x, args.rms_epsilon, [&] {
          return wait_pane(gamma_ready, 0u,
              naivert::MailboxWatchdog{args.watchdog_limit_ns, args.watchdog_flag});
        });
    if (kExportDiagnostics && blockIdx.x == 0) {
      for (int i = tid; i < 8 * 4096; i += kHeadThreads)
        args.norm_out[i] = x[(i / kHeadHidden) * kQueryStride + i % kHeadHidden];
    }
    const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns, args.watchdog_flag};
    // normalize_rows ends with a full CTA barrier, also publishing barrier init.
    if (tid >= 256) {
      const int page = warp - 8;
      auto* destination = memory + kPaneOffset + page * kPageBytes;
      // wave 0 was issued before the programmatic wait above
      for (int wave = 1, tile = int(blockIdx.x) + kHeadCtas; tile < kTiles;
           ++wave, tile += kHeadCtas) {
        const bool available = wait_pane(released + page, unsigned(wave & 1), watchdog);
        if (!__all_sync(0xffffffffu, available)) break;
        if (lane == 0) {
          naivert::Barrier::expect_transaction(ready + page, kPageBytes);
          Transport::cp_async_bulk_evict_first(
              destination,
              args.lm_head + (std::size_t(tile) * 4 + page) * 16 * kPageK,
              kPageBytes, ready + page);
          naivert::Barrier::arrive(ready + page);
        }
        __syncwarp();
      }
    } else {
      const int g = lane >> 2;
      const int q = lane & 3;
      // The normalized queries are identical for every vocabulary tile.
      // Retain the existing MMA query fragments across waves.
      uint4 query[4][4];
#pragma unroll
      for (int page = 0; page < 4; ++page) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          const int k = warp * (kPageK / 8) + step * 32 + q * 8;
          query[page][step] = *reinterpret_cast<const uint4*>(
              x + g * kQueryStride + page * kPageK + k);
        }
      }
      unsigned failed_panes = 0;
      for (int wave = 0, tile = int(blockIdx.x); tile < kTiles;
           ++wave, tile += kHeadCtas) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int page = 0; page < 4; ++page) {
          if (failed_panes & (1u << page)) continue;
          const bool available = wait_pane(ready + page, unsigned(wave & 1), watchdog);
          if (!__all_sync(0xffffffffu, available)) {
            // All consumer warps still reach the named barriers below. Other
            // panes drain normally; this pane is never read or released again.
            failed_panes |= 1u << page;
            continue;
          }
          const auto* weight = reinterpret_cast<const __nv_bfloat16*>(
              memory + kPaneOffset + page * kPageBytes);
#pragma unroll
          for (int k0 = 0; k0 < kPageK / 8; k0 += 32) {
            const int k = warp * (kPageK / 8) + k0 + q * 8;
            using Row = naivert::attn_rows::K32PackedRow;
            const Row row0{weight + g * Row::Page};
            const Row row1{weight + (g + 8) * Row::Page};
            const uint4 a0 = *reinterpret_cast<const uint4*>(row0 + k);
            const uint4 a1 = *reinterpret_cast<const uint4*>(row1 + k);
            const uint4 b = query[page][k0 / 32];
            naivert::attn_rows::mma_bf16_16816(
                c, a0.x, a1.x, a0.y, a1.y, b.x, b.y);
            naivert::attn_rows::mma_bf16_16816(
                c, a0.z, a1.z, a0.w, a1.w, b.z, b.w);
          }
          if (lane == 0) naivert::Barrier::arrive(released + page);
        }
        float* mine = partials + warp * 128;
        mine[g * 8 + 2 * q] = c[0];
        mine[g * 8 + 2 * q + 1] = c[1];
        mine[(g + 8) * 8 + 2 * q] = c[2];
        mine[(g + 8) * 8 + 2 * q + 1] = c[3];
        naivert::consumer_group_named_sync();
        if (tid < 128) {
          const int column = tid / 8;
          const int row = tid % 8;
          float sum = 0.f;
#pragma unroll
          for (int split = 0; split < 8; ++split)
            sum += partials[split * 128 + column * 8 + row];
          if (row >= args.first_logit_row && row < args.first_logit_row + 7) {
            if constexpr (Fused)
              reinterpret_cast<float*>(memory + kHeadSmem)[(row - args.first_logit_row) * LogitStride + wave * 16 + column] = sum;
            if constexpr (!Fused || kExportDiagnostics)
              args.logits[std::size_t(row - args.first_logit_row) * kHeadVocab + tile * 16 + column] = sum;
          }
        }
        naivert::consumer_group_named_sync();
      }
    }
    __syncthreads();
  }
};
}  // namespace naivert_dspark_ops
