#pragma once
#include <cutlass/arch/memory_sm80.h>
#include "naivert/body/dspark_context/args.hpp"
#include "naivert/body/dspark_context/geometry.cuh"
#include "naivert/common/resident_grid_barrier.cuh"
#include "naivert/body/dspark_common/diagnostics.hpp"
// D1: existing BF16 rows MMA, followed by the original TP8
// rank-order reduction, masked RMSNorm and replay-generation rendezvous.
// Each CTA owns 32 output columns (two MMA tiles). D12 uses all 384 threads
// for peer exchange; 64 reducers preserve rank order. All threads join barriers.
// Projection, transport and RMSNorm reduction orders are preserved.
#include "naivert/body/attn_common/strided_rows_mma.cuh"

#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/common/prefetched_rank_sum.cuh"
#include "naivert/body/dspark_common/rendezvous.cuh"

#include <cuda_bf16.h>
#include <cuda/ptx>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace naivert_dspark_ops {

constexpr int kRanks = naivert::geometry::kTpRanks;
constexpr int kCtas = 128;                    // < SM count, so every CTA is resident: both the
                                              // mailbox spin and the grid barrier require it
constexpr int kColsPerCta = kHidden / kCtas;  // 32
constexpr int kShellWarps = kThreads / 32;
constexpr int kQwords = kColsPerCta / 2;      // 16 mailbox qwords per (row, CTA)
constexpr int kPairs = kQwords / 2;           // 8 qword pairs, one thread each

// All-reduce sites alternate mailbox phases in execution order:
//   D1: 2; D3: 3,5,7,9,11; D4: 4,6,8,10,12;
//   D6 candidate: 20..26. Former latent tags 13..19 are reserved.
// D1/D3/D4 use row slabs 0..7; D6 uses slab 9 (8 is reserved). D2/D5 do not reduce.
// Within a block, alternating phases tolerate one site of rank skew. At a block
// boundary, D4(12)->D1(2) and D6(26)->D6(20) reuse a phase.
// D1's monotone rendezvous waits until every rank has finished the previous block
// before any new mailbox publication, covering both reuse hazards.
//
// Every CTA reads replay_generation at entry. One thread increments it after the
// grid barrier; D3/D4/D6 only read it. D1 must execute equally often on all ranks.
// Generation tags prevent stale matches at a repeated site; they do not protect
// a peer that still needs the previous contents of a reused mailbox address.
// All waits therefore retain their watchdog deadline and device error flag, which
// the runtime checks after draining. watchdog_limit_ns==0 explicitly disables it.
constexpr std::uint32_t kSaltD1 = 2;

static_assert(kColsPerCta == 32, "each mailbox tile covers 32 columns");
static_assert(std::size_t(kCtas) * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride,
              "one row's CTA records must fit one phase of one peer lane");

// A row of the block gets its own mailbox slab; inside it a CTA's record is one warp stride.
struct FcCtxLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};

// One physical row view in [tile16][K/32][row16][K32] storage.
using PackedFcRow = naivert::attn_rows::K32PackedRow;



// FcCtxArgs is shared with bindings through args.hpp. Pin its
// original ABI so the frozen standalone D1 launcher remains compatible.
// 9 pointers (72) + mailbox_peers[8] (64) + replay_generation (8) + salt/rank (8) + the two
// watchdog fields (16).
static_assert(sizeof(FcCtxArgs) == 176,
              "FcCtxArgs changed size -- check the shared D12 parameter contract");

// ------------------------------------------------------------------ grid barrier
// Sense-reversing arrive-and-wait across the whole grid. Every CTA is resident (kCtas < SM count),
// so the spin cannot deadlock. `generation` is never reset, which is what makes the barrier safe
// across graph replays: a replay just continues counting.

// The barrier and the mailbox spin BOTH assume every CTA is resident. If the grid ever exceeds
// what the device can hold, a CTA that has not launched cannot arrive and the ones that did spin
// forever -- inside a captured graph that is an unkillable hang, not an error.
static_assert(kCtas < NAIVERT_SM_COUNT,
              "dspark_fc_ctx needs the whole grid resident; kCtas must stay below the SM count");

struct FcBeforeReduce { __device__ void operator()() const {} };
struct FcCtxBody {
  template <bool kConsumerNorm = false, class BeforeReduce = FcBeforeReduce>
  __device__ static void run(unsigned char* smem_raw, const FcCtxArgs& args, BeforeReduce before_reduce = {}) {
    constexpr int stride = kHidden + (kConsumerNorm ? 32 : 0);
    constexpr int packed_offset=4096;
    static_assert((kRanks-1)*kWindow*kPairs*sizeof(naivert::PayloadPair)<=packed_offset);
    static_assert(packed_offset+kWindow*kColsPerCta*sizeof(__nv_bfloat16)<=kWindow*kHidden*2);
    [[maybe_unused]] auto* packed_fc=reinterpret_cast<__nv_bfloat16*>(smem_raw+packed_offset);
    auto* sa = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [8][4096] staged hidden, 64 KiB
    auto* stage = reinterpret_cast<float*>(smem_raw + kWindow * stride * 2);  // [8][32] partials
    auto* sctx = stage + kWindow * kColsPerCta;                     // [8][32] reduced, masked
    auto* sred = sctx + kWindow * kColsPerCta;                      // [8] rsqrt scale

    auto* mma_partials = sred + kWindow;
    const int tid = int(threadIdx.x);
    const int cta = int(blockIdx.x);
    const int col0 = cta * kColsPerCta;

// --- block-boundary rendezvous, announce side (dspark_common/rendezvous.cuh)
    // Announced HERE and waited on below, so the NVLink round trip overlaps the whole GEMV instead
    // of being paid in series. `k` is the same single read of the counter every mailbox site below
    // uses -- see the salt table's "D1 READS the counter ONCE".
    const unsigned k = static_cast<unsigned>(*args.replay_generation);
    if (cta == 0 && tid == 0) {
      rendezvous_announce(args.mailbox_peers, args.rank, k);
    }

    // --- stage this rank's eight hidden rows (every warp reads all of them, 8x reuse)
    {
      const auto* src = reinterpret_cast<const uint4*>(args.target_hidden);
      auto* dst = reinterpret_cast<uint4*>(sa);
      const int total = kWindow * kHidden / 8;      // 16 B per element
      if constexpr (kConsumerNorm) {
        int pending = 0;
        for (int i = tid; i < total; i += kThreads) {
          const int destination = (i / (kHidden / 8)) * (stride / 8) + i % (kHidden / 8);
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(dst + destination, src + i);
          if (++pending == 8) {
            cutlass::arch::cp_async_fence();
            pending = 0;
          }
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
      } else {
        for (int i = tid; i < total; i += kThreads) {
          dst[(i / (kHidden / 8)) * (stride / 8) + i % (kHidden / 8)] = src[i];
        }
      }
    }
    __syncthreads();

    naivert::attn_rows::project_strided_rows_mma<kWindow, kHidden, 4, 2, kShellWarps, stride>(
        0, kHidden / 16,
        [&](int column) {
          return PackedFcRow{args.fc + std::size_t(column / PackedFcRow::Rows) * (PackedFcRow::Rows * kHidden) + (column % PackedFcRow::Rows) * PackedFcRow::Page};
        },
        sa, mma_partials,
        [&](int column, int row, float value) {
          stage[row * kColsPerCta + column - col0] = value;
        });

    // --- publish this rank's partial (bf16, exactly what the stand-in's `fc_partial` holds)
    if (tid < kWindow * kColsPerCta) {
      const int m = tid >> 5, col = tid & 31;
      const __nv_bfloat16 v = __float2bfloat16_rn(stage[m * kColsPerCta + col]);
      if constexpr (kExportDiagnostics)
        args.fc_partial[std::size_t(m) * kHidden + col0 + col] = v;
      if constexpr (kConsumerNorm) packed_fc[tid]=v;
      else stage[m * kColsPerCta + col] = __bfloat162float(v);   // the reduce sums the ROUNDED partials
    }
    __syncthreads();

    before_reduce();

    const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns, args.watchdog_flag};

    // --- block-boundary rendezvous, wait side. Every CTA, one thread, then `__syncthreads()`: the
    // condition is monotone, so no grid barrier is needed to let the rest of the grid through.
    // This is the LAST point before anything touches the mailbox.
    if (tid == 0) {
      rendezvous_wait(args.mailbox_peers[args.rank], args.rank, k, watchdog);
    }
    __syncthreads();

    const unsigned gen = naivert::mailbox_generation(k, args.generation_salt);

    constexpr int exchange_items = kWindow * kPairs;
    [[maybe_unused]] auto* gathered = reinterpret_cast<naivert::PayloadPair*>(smem_raw);
    static_assert((kRanks - 1) * exchange_items * sizeof(naivert::PayloadPair) <= kWindow * kHidden * 2);
    if constexpr (kConsumerNorm) {
      // The original FC input is dead; the packed BF16 tile remains live.
      // First issue EVERY assigned publication, before any helper may poll.
      for (int job = tid; job < (kRanks - 1) * exchange_items; job += kThreads) {
        const int peer = job / exchange_items, item = job % exchange_items;
        const int row = item / kPairs, pair = item % kPairs;
        const uint2 payload=*reinterpret_cast<const uint2*>(packed_fc+item*4);
        const auto lo=payload.x, hi=payload.y;
        const auto mailbox = naivert::TileMailbox<FcCtxLayout>::make(
            args.mailbox_peers, args.rank, gen, std::size_t(row) * naivert::kMailboxSlabStride);
        auto* destination = mailbox.remote_word(naivert::mailbox_peer_rank(peer,args.rank),cta,pair*2);
        naivert::store_tagged_pair(destination, naivert::tagged_word(gen, lo),
                                  naivert::tagged_word(gen, hi));
      }
      for (int job = tid; job < (kRanks - 1) * exchange_items; job += kThreads) {
        const int slot = job / exchange_items, item = job % exchange_items;
        const int row = item / kPairs, pair = item % kPairs;
        const auto mailbox = naivert::TileMailbox<FcCtxLayout>::make(
            args.mailbox_peers, args.rank, gen, std::size_t(row) * naivert::kMailboxSlabStride);
        gathered[job] = mailbox.wait_pair_bounded<true>(slot,cta,pair*2,watchdog,args.generation_salt);
      }
      // All gathered payloads precede the original rank-ordered arithmetic.
      __syncthreads();
    }

    // --- the folded all-reduce: 64 threads, one (row, qword pair) each
    if (tid < kWindow * kPairs) {
      const int m = tid / kPairs;
      const int pair = tid % kPairs;
      const int qword = pair * 2;
      const auto mailbox = naivert::TileMailbox<FcCtxLayout>::make(
          args.mailbox_peers, args.rank, gen, std::size_t(m) * naivert::kMailboxSlabStride);
      std::uint32_t p01,p23;
      if constexpr (kConsumerNorm) {
        const uint2 payload=*reinterpret_cast<const uint2*>(packed_fc+tid*4);
        p01=payload.x;p23=payload.y;
      } else {
        const float* v=stage+m*kColsPerCta+pair*4;
        p01=naivert::pack_bf16x2(v[0],v[1]);
        p23=naivert::pack_bf16x2(v[2],v[3]);
      }
      if constexpr (!kConsumerNorm) mailbox.publish_pair(cta, qword, p01, p23);

      naivert::Bf16x4Acc red = naivert::Bf16x4Acc::zero();
      if constexpr (kConsumerNorm) {
        naivert::accumulate_rank_order(red,args.rank,p01,p23,[&](int slot) {
          return gathered[slot * exchange_items + tid];
        });
      } else {
      naivert::accumulate_prefetched_rank_order(red, mailbox, p01, p23, cta, qword, [&](int slot) {
        return mailbox.wait_pair_bounded<true>(slot, cta, qword, watchdog,
                                               args.generation_salt);
      });
      }

      // Round once, then mask. The order matters: the stand-in stores the reduced bf16 into
      // `ctx_sum` and the norm reads THAT, so the square below has to see the rounded value.
      const bool valid = __bfloat162float(args.ctx_valid[m]) != 0.0f;
      float r[4] = {red.v0, red.v1, red.v2, red.v3};
      float ssq = 0.0f;
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat16 b = __float2bfloat16_rn(r[i]);
        if (!valid) {
          b = __float2bfloat16_rn(0.0f);
        }
        if constexpr (kExportDiagnostics || kConsumerNorm)
          args.ctx_sum[std::size_t(m) * kHidden + col0 + pair * 4 + i] = b;
        const float f = __bfloat162float(b);
        sctx[m * kColsPerCta + pair * 4 + i] = f;
        ssq += f * f;
      }
      // The eight threads of a row are eight consecutive lanes, so this stays inside one warp.
  #pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
      }
      if (pair == 0) {
        args.ssq[m * kCtas + cta] = ssq;
      }
    }

    if constexpr (kConsumerNorm) asm volatile("fence.proxy.async;" ::: "memory");
    grid_barrier(args.barrier);

    if constexpr (!kConsumerNorm) {
    // One output warp owns one row's ordered SSQ fold and normalization.
    // Lane l loads columns 4*l..4*l+3. The running sum visits lanes 0..31,
    // retaining exactly the original scalar c=0..127 addition order.
    static_assert(kCtas == 32 * 4 && kColsPerCta == 32);
    const int row = tid >> 5, lane = tid & 31;
    if (row < kWindow) {
      const float4 pieces = reinterpret_cast<const float4*>(args.ssq + row * kCtas)[lane];
      float sum = 0.0f;
#pragma unroll 1
      for (int owner = 0; owner < 32; ++owner) {
        if (lane == owner) {
          sum = __fadd_rn(sum, pieces.x);
          sum = __fadd_rn(sum, pieces.y);
          sum = __fadd_rn(sum, pieces.z);
          sum = __fadd_rn(sum, pieces.w);
        }
        sum = __shfl_sync(0xffffffffu, sum, owner);
      }
      const float inverse_rms = rsqrtf(sum / float(kHidden) + args.rms_epsilon);
      const int n = col0 + lane;
      const bool valid = __bfloat162float(args.ctx_valid[row]) != 0.0f;
      const float f = (sctx[row * kColsPerCta + lane] * inverse_rms) * args.hidden_norm[n];
      args.ctx[std::size_t(row) * kHidden + n] = __float2bfloat16_rn(valid ? f : 0.0f);
    }

    }

    // Every CTA has read the counter (that happened before the barrier), so one thread may advance
    // it for the next execution.
    if (cta == 0 && tid == 0) {
      atomicAdd(reinterpret_cast<unsigned*>(args.replay_generation), 1u);
    }
  }

};

}  // namespace naivert_dspark_ops
