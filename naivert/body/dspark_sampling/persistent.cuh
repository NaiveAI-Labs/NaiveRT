#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "random.cuh"
#include "naivert/body/dspark_common/temperature.cuh"
#include "sync.cuh"

// TP8-sharded DSpark T=1, top-p=.95 sampler as ONE cooperative kernel per rank,
// with the naivert mailbox protocol for every internal exchange.
//
// Arithmetic, tile layout and reduction structure (CUB block reductions and scans
// over the same operands in the same order) are those of sharded.cuh, so p / q /
// tokens are bitwise those of the 33-kernel chain. What changes is only how the
// ranks synchronize:
//
//  * every published value (row maxima, sums, per-pass histograms, proposal p/q,
//    CDF masses, winners) is PUSHED into all eight ranks' inboxes as 64-bit
//    tagged words = (generation << 32) | payload32 (doubles as two halves);
//    the consumer polls its own local memory until the tags match. The data is
//    self-validating, so no fence, flag or rendezvous kernel is needed
//    (fence-based rendezvous cost ~6 us per round on this fabric; a tagged
//    exchange ~2 us).
//  * rows are independent pipelines synchronized through row-local arrival
//    counters; the only grid-wide barrier is one at kernel start. Each CTA of a
//    row merges and pushes a slice of the row histogram and recomputes the
//    row's selection state from its local inbox (identical in all ten CTAs).
//  * the external ready / consumed rendezvous on the runtime's flags array keep
//    sharded.cuh's pull protocol unchanged (the verify epilogue depends on it).
//
// Grid: rows x tiles CTAs (80 for verify, 10 for the anchor). CTA (row, part)
// owns the target tile (row, part) and, for row < 7 in verify, the corrected
// draft tile (row, part). The row's part-0 CTA does the row's reductions,
// publishes, selects, and picks the token. Cross-CTA scratch is read with
// __ldcg; inbox words with relaxed system-scope atomics.
namespace naivert_dspark_persistent {
constexpr int full_vocabulary = 152576;
constexpr int vocabulary = full_vocabulary / 8;
constexpr int threads = 256;
constexpr int items = 8;
constexpr int tile = threads * items;
constexpr int tiles = (vocabulary + tile - 1) / tile;
constexpr int bins = 256;            // sharded.cuh prefix compatibility only
constexpr int max_rows = 8;
constexpr int distributions = 15;
#ifndef PERSISTENT_PASSES
#define PERSISTENT_PASSES 4          // 4: sharded.cuh's 8 / 8 / 8 / 8 (3: 11 / 11 / 10 bits: correct but the wider polls/pushes lose)
#endif
constexpr int passes = PERSISTENT_PASSES;
#ifndef PERSISTENT_HIST_ATOMIC
#define PERSISTENT_HIST_ATOMIC 0   // 1: plain shared atomics per element instead of warp label-partition + reduce
#endif
__host__ __device__ constexpr int pass_shift(int pass) { return passes == 3 ? (pass == 0 ? 21 : pass == 1 ? 10 : 0) : 24 - 8 * pass; }
__host__ __device__ constexpr int pass_bits(int pass) { return passes == 3 ? (pass == 2 ? 10 : 11) : 8; }
constexpr int max_bins = passes == 3 ? 2048 : 256;
constexpr int scan_items = max_bins / threads;   // digits per thread in the selection scan
#ifndef PERSISTENT_GATHER
#define PERSISTENT_GATHER 1          // 1: after pass 0, gather the boundary bucket's survivors and finish passes 1..3 locally
#endif
constexpr int cand_cap = 4096;       // survivors per rank per row the inbox can hold; more -> global passes (fallback)
constexpr int cand_shared = 8192;    // gathered survivors (all ranks) held in shared memory for the local passes; more -> fallback
#ifndef PERSISTENT_ONLINE_SOFTMAX
#define PERSISTENT_ONLINE_SOFTMAX 0   // 1: exponentiate with the rank-local maximum and exchange (max, sum) once; p differs from the chain by ~1 ulp
#endif
#ifndef PERSISTENT_GUMBEL
#define PERSISTENT_GUMBEL 1          // 1 (default): Gumbel-max draw, one exchange; same distribution as the CDF draw, different tokens per seed.
                                     // 0: sharded.cuh's vocabulary-order CDF draw (mass + winner exchanges): token-exact vs the chain.
#endif


#ifndef PERSISTENT_EXTERNAL
#define PERSISTENT_EXTERNAL 1   // 0: skip BOTH external rendezvous (experiment; breaks the tail contract)
#endif
#ifndef PERSISTENT_READY
#define PERSISTENT_READY 0      // default 0: no external READY rendezvous (the tagged inboxes make it redundant); the external
                                // CONSUMED rendezvous is kept and advances the generation itself. 1: sharded.cuh's ready + consumed.
                                // Integration note: with a DEFERRED consumed (verify epilogue) the epilogue must increment the
                                // generation (own[2]) when PERSISTENT_READY == 0, or its rendezvous degenerates to a no-op.
                                // Token-exact configuration: PERSISTENT_GUMBEL=0 with PERSISTENT_READY=0 (71.9 / 63.7 us, bitwise).
#endif
#ifdef PERSISTENT_STAMPS
constexpr int stamp_points = 32, stamp_ctas = 160;
__device__ unsigned long long g_stamps[stamp_points * stamp_ctas];   // per CTA, thread 0, %globaltimer
__device__ __forceinline__ void stamp(int i) {
  if (threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    g_stamps[i * stamp_ctas + blockIdx.x] = t;
    // Points 15 / 16: SM clock64 at points 0 / 14, so (clk16 - clk15) / (t14 - t0) is the SM clock the
    // kernel actually ran at (in-round after 2.7 ms of power-bound target kernels vs standalone).
    if (i == 0 || i == 14) g_stamps[(i == 0 ? 15 : 16) * stamp_ctas + blockIdx.x] = clock64();
  }
}
// Survivor history (stamps build only): one word per (invocation, row, stage) written by the row leader after
// the count exchange: epoch << 40 | row << 36 | stage << 32 | overflow << 31 | gathered total (capped counts).
constexpr int cand_hist_words = 4096;
__device__ unsigned long long g_cand_hist[cand_hist_words];
__device__ unsigned g_cand_hist_next;
__device__ __forceinline__ void cand_hist(unsigned epoch, int row, int stage, unsigned overflow, unsigned total) {
  const unsigned i = atomicAdd(&g_cand_hist_next, 1u) % unsigned(cand_hist_words);
  using U = unsigned long long;
  g_cand_hist[i] = (U(epoch) << 40) | (U(row) << 36) | (U(stage) << 32) | (U(overflow) << 31) | U(total);
}
#else
__device__ __forceinline__ void stamp(int) {}
__device__ __forceinline__ void cand_hist(unsigned, int, int, unsigned, unsigned) {}
#endif

// Tagged inbox: every word is written by exactly one (source rank, phase) and
// read by this rank only. Slots are reused once per invocation; a writer cannot
// reach invocation n+1 before every reader finished invocation n's last exchange.
struct Inbox {
  unsigned long long maximum[distributions][8];   // [distribution][source]  float payload
  unsigned long long sum[distributions][8][2];    // [distribution][source][half of the double]
  unsigned long long histogram[passes][max_rows][8][max_bins][2];   // [pass][row][source][digit][half]
  unsigned long long proposal_p[7], proposal_q[7];        // written by the proposal's owner rank
  unsigned long long mass[max_rows][8][2];
  unsigned long long winner[max_rows][8];
  unsigned long long epoch;   // this rank's invocation counter = tag generation (never reset)
  unsigned long long gumbel[max_rows][8][2];         // [row][source]: (key bits, global index) of the rank's Gumbel-max candidate
  unsigned long long cand_count[max_rows][8];        // [row][source]: survivors of the boundary bucket (cand_cap + 1 = overflow)
  unsigned long long cand[max_rows][8][cand_cap];    // [row][source][slot]: FP32 bits of the survivors
};

// naivert_dspark_sharded::Mailbox as a prefix (peer-compatible), plus the inbox.
struct Mailbox {
  unsigned long long flags[8][3];
  float maximum[15];
  double sum[15];
  double histogram[4][8][bins];
  float proposal_p[7], proposal_q[7];
  double mass[8];
  int winner[8];
  Inbox inbox;
};

// Identical layout to naivert_dspark_sharded::Args.
struct Args {
  const float* logits;
  const float* draft_logits;
  Mailbox* peers[8];
  unsigned long long* flags[8];
  int rank;
  float* p;
  float* q;
  int32_t* tokens;
  const int64_t* proposals;
  const int64_t* rng;
  unsigned* watchdog;
  bool anchor;
  // Per-request sampling parameters, host-written (DraftBuffers.sampling_params, float[8]):
  // [0..3] repeat penalty {repetition theta, frequency alpha, presence beta, active}, [4] 1/T,
  // [6..7] top-p as one float64. rep_count[v]: occurrences of token v in the committed context over
  // the full vocabulary (this rank reads its shard; the tail keeps it current); row d of a block
  // also counts proposals[0..d-1]. Penalty inactive, 1/T == 1 and top-p 0.95 are bitwise the
  // compile-time defaults.
  const int32_t* rep_count;
  const float* sampling_params;
};
__device__ __forceinline__ double top_p_of(const Args& a) {
  return a.sampling_params != nullptr ? *reinterpret_cast<const double*>(a.sampling_params + 6) : 0.95;
}
__device__ __forceinline__ float inv_temperature_of(const Args& a) {
  return a.sampling_params != nullptr ? a.sampling_params[4] : naivert_dspark::kInvTemperature;
}

// naivert_dspark_sharded::Scratch as a prefix, plus the persistent kernel's own state.
struct Scratch {
  float maxima[15 * tiles];
  double sums[15 * tiles];
  float peak[15], inverse[15];
  double histogram[8 * tiles * max_bins];
  unsigned prefix[8];
  double remaining[8], above[8], kept[8], kept_inverse[8], mass[8],
      rank_base[8], chunk_prefix[8 * tiles];
  int reject[8], selected_tile[8];
  unsigned counter[8];               // grid-level arrivals (last arriver resets)
  unsigned epoch;                    // invocation counter = the tag generation
  unsigned ready_epoch;              // set by CTA 0 after the external ready rendezvous
  unsigned row_count[max_rows][8];   // row-local arrivals [row][phase] (the last arriver resets)
  unsigned row_gen[max_rows][8];     // row-local generation published by the last arriver of [row][phase]
  unsigned cand_offset[max_rows];    // survivor slot allocator per row (the leader resets)
  float gumbel_key[max_rows * tiles]; // per-tile Gumbel-max key
  float local_max[distributions];     // (online softmax) the rank's maximum per distribution, published by the row leader
  unsigned lmax_epoch[max_rows];      // (online softmax) flag: local maxima of the row published for this tag
  int gumbel_index[max_rows * tiles]; // per-tile Gumbel-max local index
  unsigned fallback[max_rows];       // 1: some rank overflowed cand_cap, finish with the global passes
  unsigned state_epoch[max_rows];    // (v4) row state published for (epoch, pass)
  unsigned fallback2[max_rows];      // (v18) 1: some rank overflowed cand_cap in the second gather (after global pass 1)
  unsigned decision[max_rows];       // (v18) the leader's verdict after a gather stage: 1 = continue with global passes.
                                     // Separate from fallback/fallback2, which the gathering CTAs set and the leader
                                     // resets after reading; storing the verdict there made a fallback sticky across
                                     // invocations (the next leader read its own old verdict as a fresh overflow).
};

struct Maximum {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Minimum {
  __device__ int operator()(int a, int b) const { return min(a, b); }
};

using ReduceF = cub::BlockReduce<float, threads>;
using ReduceD = cub::BlockReduce<double, threads>;
using ReduceI = cub::BlockReduce<int, threads>;
using ScanD = cub::BlockScan<double, threads>;
using ReduceD2 = cub::BlockReduce<double2, threads>;
struct KeyIndex {
  float key;
  int index;
};
struct MaxKeyIndex {
  __device__ KeyIndex operator()(KeyIndex a, KeyIndex b) const {
    if (a.key > b.key) return a;
    if (b.key > a.key) return b;
    return a.index <= b.index ? a : b;
  }
};
using ReduceKI = cub::BlockReduce<KeyIndex, threads>;
struct SumD2 {
  __device__ double2 operator()(double2 a, double2 b) const { return make_double2(a.x + b.x, a.y + b.y); }
};
union Temp {
  ReduceF::TempStorage rf;
  ReduceD::TempStorage rd;
  ReduceD2::TempStorage rd2;
  cub::BlockReduce<KeyIndex, threads>::TempStorage ki;
  ReduceI::TempStorage ri;
  ScanD::TempStorage sd;
};

using SysAtomic = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
using DevAtomicU = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;

__device__ __forceinline__ unsigned long long tagged(unsigned tag, unsigned payload) {
  return (static_cast<unsigned long long>(tag) << 32) | payload;
}
__device__ __forceinline__ unsigned payload(unsigned long long word) { return static_cast<unsigned>(word); }
__device__ __forceinline__ void push(unsigned long long* remote, unsigned long long word) {
  SysAtomic(*remote).store(word, cuda::memory_order_relaxed);
}
__device__ __forceinline__ unsigned long long timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Poll a local tagged word until its tag matches; returns the payload.
__device__ __forceinline__ unsigned poll(unsigned long long* local, unsigned tag, unsigned* watchdog) {
  const unsigned long long start = timer();
  for (;;) {
    const unsigned long long word = SysAtomic(*local).load(cuda::memory_order_relaxed);
    if (static_cast<unsigned>(word >> 32) == tag) return payload(word);
    if (timer() - start > 30000000000ull) {
      atomicCAS(watchdog, 0u, 0xD6FAu);
      return payload(word);
    }
  }
}
__device__ __forceinline__ double poll_double(unsigned long long* local2, unsigned tag, unsigned* watchdog) {
  const unsigned lo = poll(local2, tag, watchdog), hi = poll(local2 + 1, tag, watchdog);
  return __longlong_as_double(static_cast<long long>((static_cast<unsigned long long>(hi) << 32) | lo));
}
__device__ __forceinline__ void push_double(unsigned long long* remote2, unsigned tag, double v) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));
  push(remote2, tagged(tag, static_cast<unsigned>(bits)));
  push(remote2 + 1, tagged(tag, static_cast<unsigned>(bits >> 32)));
}
// Poll a local device-scope counter/flag until it reaches `value`.
__device__ __forceinline__ void wait_flag(unsigned* flag, unsigned value, unsigned* watchdog) {
  const unsigned long long start = timer();
  while (DevAtomicU(*flag).load(cuda::memory_order_acquire) < value) {
    if (timer() - start > 30000000000ull) {
      atomicCAS(watchdog, 0u, 0xD6FCu);
      return;
    }
  }
}

// Warp helpers: lanes 0..7 poll one source each in parallel, then combine in rank order 0..7 (the same
// operand order as the serial loops of sharded.cuh, so results are bitwise identical). All 32 lanes call.
__device__ __forceinline__ double warp_poll_sum8(unsigned long long (*words)[2], unsigned tag, unsigned* watchdog,
                                                 int rank, double* preceding_out = nullptr) {
  const int lane = threadIdx.x & 31;
  double v = 0;
  if (lane < 8) v = poll_double(words[lane], tag, watchdog);
  double sum = 0, preceding = 0;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const double x = __shfl_sync(0xffffffffu, v, r);
    if (r < rank) preceding += x;
    sum += x;
  }
  if (preceding_out) *preceding_out = preceding;
  return sum;
}
__device__ __forceinline__ float warp_poll_fmax8(unsigned long long* words, unsigned tag, unsigned* watchdog) {
  const int lane = threadIdx.x & 31;
  float v = -INFINITY;
  if (lane < 8) v = __uint_as_float(poll(&words[lane], tag, watchdog));
  float m = -INFINITY;
#pragma unroll
  for (int r = 0; r < 8; ++r) m = fmaxf(m, __shfl_sync(0xffffffffu, v, r));
  return m;
}
__device__ __forceinline__ int warp_poll_min8(unsigned long long* words, unsigned tag, unsigned* watchdog) {
  const int lane = threadIdx.x & 31;
  int v = full_vocabulary;
  if (lane < 8) v = int(poll(&words[lane], tag, watchdog));
  int m = full_vocabulary;
#pragma unroll
  for (int r = 0; r < 8; ++r) m = min(m, __shfl_sync(0xffffffffu, v, r));
  return m;
}

__device__ __forceinline__ float* distribution(const Args& a, int d) {
  return d < 8 ? a.p + d * vocabulary : a.q + (d - 8) * vocabulary;
}

// External ready / consumed rendezvous: sharded.cuh's protocol on the runtime flags array (one warp).
__device__ __forceinline__ void external_rendezvous(const Args& a, bool consumed) {
  using Atomic = naivert_dspark_ops::Atomic;
  const int lane = threadIdx.x & 31;
  auto* own = a.flags[a.rank];
  unsigned long long generation = 0;
  if (lane == 0) {
    generation = (consumed && PERSISTENT_READY) ? own[2] : ++own[2];   // without ready, consumed advances the generation
    __threadfence_system();
    Atomic(own[consumed ? 1 : 0]).store(generation, cuda::memory_order_release);
  }
  generation = __shfl_sync(0xffffffffu, generation, 0);
  if (lane < 8) {
    auto* peer = a.flags[lane];
    const unsigned long long start = timer();
    while (Atomic(peer[consumed ? 1 : 0]).load(cuda::memory_order_acquire) < generation) {
      if (timer() - start > 30000000000ull) {
        atomicCAS(a.watchdog, 0u, 0xD6FAu);
        break;
      }
      __nanosleep(64);
    }
  }
  __syncwarp();
}

// Row-local arrival: every CTA of the row signals after its global writes; the
// last arriver resets the counter and publishes the row generation `tag` for the phase.
__device__ __forceinline__ void row_arrive(Scratch* s, int row, int phase, unsigned tag = 0) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned previous = atomicAdd(&s->row_count[row][phase], 1u);
    if (tag != 0 && previous + 1 == unsigned(tiles)) {
      atomicExch(&s->row_count[row][phase], 0u);
      __threadfence();
      DevAtomicU(s->row_gen[row][phase]).store(tag, cuda::memory_order_release);
    }
  }
}
// All CTAs of the row wait for the phase generation (published by the last arriver).
__device__ __forceinline__ void row_wait_all(Scratch* s, int row, int phase, unsigned tag, unsigned* watchdog) {
  if (threadIdx.x == 0) wait_flag(&s->row_gen[row][phase], tag, watchdog);
  __syncthreads();
}
__device__ __forceinline__ void row_wait(Scratch* s, int row, int phase, unsigned* watchdog) {
  if (threadIdx.x == 0) {
    wait_flag(&s->row_count[row][phase], tiles, watchdog);
    atomicExch(&s->row_count[row][phase], 0u);
    __threadfence();
  }
  __syncthreads();
}

// ---- per-tile work (bodies identical to sharded.cuh, loads hoisted before stores) ----

__device__ __forceinline__ void partial_maximum(const Args& a, Scratch* s, Temp& temp, int d, int part) {
  float* x = distribution(a, d);
  const float* input = d < 8 ? a.logits + d * vocabulary : a.draft_logits + (d - 8) * vocabulary;
  float w[items];
  const float inv = inv_temperature_of(a);
  const bool penalize = d < 8 && a.sampling_params != nullptr && a.sampling_params[3] != 0.0f;
  if (!penalize) {
#pragma unroll
    for (int k = 0; k < items; ++k) {
      const int i = part * tile + threadIdx.x + k * threads;
      w[k] = i < vocabulary ? naivert_dspark::tempered(input[i], inv) : -INFINITY;
    }
  } else {
    const float theta = a.sampling_params[0], alpha = a.sampling_params[1], beta = a.sampling_params[2];
    const int64_t base = int64_t(a.rank) * vocabulary;
    const int32_t* counts = a.rep_count + base;
    int64_t block[7];
#pragma unroll
    for (int j = 0; j < 7; ++j) block[j] = j < d ? a.proposals[j] : int64_t(-1);
#pragma unroll
    for (int k = 0; k < items; ++k) {
      const int i = part * tile + threadIdx.x + k * threads;
      float v = -INFINITY;
      if (i < vocabulary) {
        v = input[i];
        int c = counts[i];
#pragma unroll
        for (int j = 0; j < 7; ++j) c += block[j] == base + i ? 1 : 0;
        if (c > 0) {
          v = v > 0.0f ? v / theta : v * theta;
          v -= alpha * float(c) + beta;
        }
        v = naivert_dspark::tempered(v, inv);
      }
      w[k] = v;
    }
  }
  float v = -INFINITY;
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      x[i] = w[k];
      v = fmaxf(v, w[k]);
    }
  }
  v = ReduceF(temp.rf).Reduce(v, Maximum());
  if (threadIdx.x == 0) s->maxima[d * tiles + part] = v;
  __syncthreads();
}

__device__ __forceinline__ void exponentiate(const Args& a, Scratch* s, Temp& temp, int d, int part, float peak) {
  float* x = distribution(a, d);
  float w[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    w[k] = i < vocabulary ? x[i] : 0.0f;
  }
  double sum = 0;
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      const float v = expf(w[k] - peak);
      x[i] = v;
      sum += v;
    }
  }
  sum = ReduceD(temp.rd).Sum(sum);
  if (threadIdx.x == 0) s->sums[d * tiles + part] = sum;
  __syncthreads();
}

// p and q tiles of the same (row, part) together: all sixteen loads in flight before the first store.
__device__ __forceinline__ void exponentiate2(const Args& a, Scratch* s, Temp& temp, int d0, int d1, int part,
                                              float peak0, float peak1) {
  float* x0 = distribution(a, d0);
  float* x1 = distribution(a, d1);
  float w0[items], w1[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    w0[k] = i < vocabulary ? x0[i] : 0.0f;
    w1[k] = i < vocabulary ? x1[i] : 0.0f;
  }
  double sum0 = 0, sum1 = 0;
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      const float v0 = expf(w0[k] - peak0), v1 = expf(w1[k] - peak1);
      x0[i] = v0;
      x1[i] = v1;
      sum0 += v0;
      sum1 += v1;
    }
  }
  const double2 both = ReduceD2(temp.rd2).Reduce(make_double2(sum0, sum1), SumD2());
  if (threadIdx.x == 0) {
    s->sums[d0 * tiles + part] = both.x;
    s->sums[d1 * tiles + part] = both.y;
  }
  __syncthreads();
}

__device__ __forceinline__ void normalize2(const Args& a, int d0, int d1, int part, float inv0, float inv1, unsigned tag) {
  float* x0 = distribution(a, d0);
  float* x1 = distribution(a, d1);
  const int proposal_row = d0;   // d0 < 8 is the target row, d1 = 8 + d0 the draft row of the same proposal
  const long long proposal = !a.anchor && proposal_row < 7 ? (long long)a.proposals[proposal_row] : -1ll;
  float w0[items], w1[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    w0[k] = i < vocabulary ? x0[i] : 0.0f;
    w1[k] = i < vocabulary ? x1[i] : 0.0f;
  }
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      const float v0 = w0[k] * inv0, v1 = w1[k] * inv1;
      x0[i] = v0;
      x1[i] = v1;
      if (proposal == (long long)(a.rank * vocabulary + i)) {
        for (int r = 0; r < 8; ++r) {
          push(&a.peers[r]->inbox.proposal_p[proposal_row], tagged(tag, __float_as_uint(v0)));
          push(&a.peers[r]->inbox.proposal_q[proposal_row], tagged(tag, __float_as_uint(v1)));
        }
      }
    }
  }
}

// Normalizes the tile; the proposal's owner pushes the proposal's p or q to every rank.
__device__ __forceinline__ void normalize(const Args& a, int d, int part, float inv, unsigned tag) {
  float* x = distribution(a, d);
  const int proposal_row = d < 8 ? d : d - 8;
  const long long proposal = !a.anchor && proposal_row < 7 ? (long long)a.proposals[proposal_row] : -1ll;
  float w[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    w[k] = i < vocabulary ? x[i] : 0.0f;
  }
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      const float v = w[k] * inv;
      x[i] = v;
      if (proposal == (long long)(a.rank * vocabulary + i)) {
        const unsigned long long word = tagged(tag, __float_as_uint(v));
        for (int r = 0; r < 8; ++r)
          push(d < 8 ? &a.peers[r]->inbox.proposal_p[proposal_row] : &a.peers[r]->inbox.proposal_q[proposal_row], word);
      }
    }
  }
}

template <int pass>
__device__ __forceinline__ void radix_histogram(const Args& a, Scratch* s, unsigned long long* hist_u,
                                                double* hist_d, int row, int part, unsigned prefix) {
  namespace cg = cooperative_groups;
  constexpr int shift = pass_shift(pass), width = pass_bits(pass), nbins = 1 << width;
  const int t = threadIdx.x;
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  float w[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + t + k * threads;
    w[k] = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
  }
  if constexpr (pass == 0) {
    // Exact integer units: every value of a bucket shares one exponent (width >= 9) or at most two adjacent
    // exponents (width 8, sharded's trick); units < 2^25, a warp sum < 2^30, a tile sum exact in uint64.
    for (int b = t; b < nbins; b += threads) hist_u[b] = 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < items; ++k) {
      const float v = w[k];
      const unsigned bits = __float_as_uint(v), digit = bits >> shift;
      const unsigned exponent = (bits >> 23) & 255;
      unsigned units = (bits & 0x7fffff) | (exponent ? 0x800000 : 0);
      if constexpr (width == 8) { if (digit) units <<= exponent & 1; }
      if constexpr (PERSISTENT_HIST_ATOMIC) {
        if (units) atomicAdd(&hist_u[digit], static_cast<unsigned long long>(units));   // exact integers: any order
      } else {
        const auto group = cg::labeled_partition(warp, digit);
        const unsigned mass = cg::reduce(group, units, cg::plus<unsigned>());
        if (group.thread_rank() == 0 && mass) atomicAdd(&hist_u[digit], static_cast<unsigned long long>(mass));
      }
    }
    __syncthreads();
    for (int b = t; b < nbins; b += threads) {
      int power;
      if constexpr (width == 8) power = b ? 2 * b - 150 : -149;
      else {
        const int exponent = (b >> (width - 9)) & 255;   // digit = sign | exponent | (width-9) mantissa bits
        power = exponent ? exponent - 150 : -149;
      }
      const double unit = __longlong_as_double(static_cast<long long>(power + 1023) << 52);
      s->histogram[(row * tiles + part) * max_bins + b] = double(hist_u[b]) * unit;
    }
  } else {
    for (int b = t; b < nbins; b += threads) hist_d[b] = 0;
    __syncthreads();
    constexpr unsigned mask = 0xffffffffu << (shift + width);
#pragma unroll
    for (int k = 0; k < items; ++k) {
      const int i = part * tile + t + k * threads;
      const float v = w[k];
      const unsigned bits = __float_as_uint(v);
      const bool active = i < vocabulary && (bits & mask) == prefix;
      if (!__any_sync(0xffffffffu, active && v > 0)) continue;
      const unsigned digit = (bits >> shift) & (nbins - 1);
      if constexpr (PERSISTENT_HIST_ATOMIC) {
        // values of one bucket share an exponent, so their FP64 sum is exact in any order
        if (active && v > 0) atomicAdd(&hist_d[digit], double(v));
      } else {
        const auto group = cg::labeled_partition(warp, digit);
        const double mass = cg::reduce(group, active ? double(v) : 0.0, cg::plus<double>());
        if (group.thread_rank() == 0 && mass != 0) atomicAdd(&hist_d[digit], mass);
      }
    }
    __syncthreads();
    for (int b = t; b < nbins; b += threads) s->histogram[(row * tiles + part) * max_bins + b] = hist_d[b];
  }
}

// ---- part-0 CTA work -------------------------------------------------------------

// Per-row selection state, identical in every CTA of the row (each CTA recomputes it from the same inbox).
struct RowState {
  unsigned prefix;
  double remaining, above, kept, kept_inverse;
  int reject;
  float u_accept, u_draw;   // sampling_uniform(rng, 1, row) and sampling_uniform(rng, anchor ? 3 : 2, row)
};

// Every CTA of the row: merge its digit slice (same tile order as sharded's merge_histogram) and push it to
// all ranks; then poll the whole row histogram of all eight sources from the local inbox and run the same
// scan / select as sharded's select_digit. Ten CTAs do the poll + scan redundantly (local memory, cheap);
// this replaces the leader's serial merge and the state publish / wait.
template <int pass>
__device__ __forceinline__ void merge_exchange_select(const Args& a, const Scratch* s, Temp& temp, RowState& st,
                                                      int row, int part, unsigned tag) {
  constexpr int shift = pass_shift(pass), width = pass_bits(pass), nbins = 1 << width, per = nbins / threads;
  const int t = threadIdx.x;
  // merge this CTA's digit slice over the ten tiles (tile order) and push it to every rank
  for (int b = t; b < nbins; b += threads) {
    if (b % tiles != part) continue;
    double merged = 0;
    for (int p = 0; p < tiles; ++p) merged += __ldcg(&s->histogram[(row * tiles + p) * max_bins + b]);
    for (int r = 0; r < 8; ++r) push_double(a.peers[r]->inbox.histogram[pass][row][a.rank][b], tag, merged);
  }
  // poll the row histogram of all eight sources; thread t owns digits nbins-1-per*t-j (descending, contiguous).
  // All sixteen words of a digit are loaded back to back and only then tag-checked: one dependent
  // load-compare loop per word costs ~0.6 us each and was the per-pass cost.
  double mass[per];
#pragma unroll
  for (int j = 0; j < per; ++j) {
    const int digit = nbins - 1 - (per * t + j);
    unsigned long long w[16];
    const unsigned long long start = timer();
    for (;;) {
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        w[2 * r] = SysAtomic(a.peers[a.rank]->inbox.histogram[pass][row][r][digit][0]).load(cuda::memory_order_relaxed);
        w[2 * r + 1] = SysAtomic(a.peers[a.rank]->inbox.histogram[pass][row][r][digit][1]).load(cuda::memory_order_relaxed);
      }
      bool ok = true;
#pragma unroll
      for (int q = 0; q < 16; ++q) ok &= static_cast<unsigned>(w[q] >> 32) == tag;
      if (ok) break;
      if (timer() - start > 30000000000ull) {
        atomicCAS(a.watchdog, 0u, 0xD6FAu);
        break;
      }
    }
    double m = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r)
      m += __longlong_as_double(static_cast<long long>((static_cast<unsigned long long>(payload(w[2 * r + 1])) << 32) | payload(w[2 * r])));
    mass[j] = m;
  }
  const double remaining = st.remaining, previously_above = st.above;
  double above[per];
  cub::BlockScan<double, threads>(temp.sd).ExclusiveSum(mass, above);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < per; ++j) {
    if (above[j] < remaining && above[j] + mass[j] >= remaining) {
      const int digit = nbins - 1 - (per * t + j);
      const unsigned prefix = st.prefix | (unsigned(digit) << shift);
      st.prefix = prefix;
      st.remaining = remaining - above[j];
      st.above = previously_above + above[j];
      if constexpr (pass == passes - 1) {
        const double kept = previously_above + above[j] + mass[j];
        st.kept = kept;
        st.kept_inverse = 1.0 / kept;
        const int candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
        float raw = 0, q = 0;
        if (candidate >= 0) {
          raw = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_p[row], tag, a.watchdog));
          q = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_q[row], tag, a.watchdog));
        }
        const float p = __float_as_uint(raw) >= prefix ? float(raw / kept) : 0;
        st.reject = candidate >= 0 && !(st.u_accept * q < p);
      }
    }
  }
  __syncthreads();
}

// After pass 0 (stage 1) or after global pass 1 (stage 2): every CTA compacts the survivors of its tile
// (the selected prefix: top byte after pass 0, top 16 bits after pass 1; v > 0) and pushes their FP32 bits to
// all ranks' inboxes at row-global slots; a CTA whose slots would exceed the cap marks the row for fallback.
// The values are what the remaining passes need (indices are not). Stage 2 reuses the slots under a tag
// whose top bit is flipped, so its words cannot be mistaken for stage 1's leftovers.
__host__ __device__ constexpr unsigned stage_tag(unsigned tag, int stage) { return stage == 1 ? tag : tag ^ 0x80000000u; }
template <int Stage>
__device__ __forceinline__ void gather_survivors(const Args& a, Scratch* s, const RowState& st, int row, int part,
                                                 unsigned tag) {
  constexpr unsigned mask = Stage == 1 ? 0xff000000u : 0xffff0000u;
  const unsigned prefix = st.prefix & mask;
  unsigned* fallback = Stage == 1 ? &s->fallback[row] : &s->fallback2[row];
  const int t = threadIdx.x;
  float w[items];
  bool keep[items];
  int n = 0;
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + t + k * threads;
    w[k] = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    keep[k] = i < vocabulary && (__float_as_uint(w[k]) & mask) == prefix && w[k] > 0;
    n += keep[k];
  }
  __shared__ unsigned base_s;
  using ScanI = cub::BlockScan<int, threads>;
  __shared__ ScanI::TempStorage scan_i;
  int offset, total;
  ScanI(scan_i).ExclusiveSum(n, offset, total);
  if (t == 0) base_s = total ? atomicAdd(&s->cand_offset[row], unsigned(total)) : 0u;
  __syncthreads();
  const unsigned base = base_s;
  if (base + unsigned(total) > unsigned(cand_cap)) {
    if (t == 0) atomicExch(fallback, 1u);
    return;
  }
  unsigned slot = base + unsigned(offset);
#pragma unroll
  for (int k = 0; k < items; ++k) {
    if (keep[k]) {
      const unsigned long long word = tagged(tag, __float_as_uint(w[k]));
      for (int r = 0; r < 8; ++r) push(&a.peers[r]->inbox.cand[row][a.rank][slot], word);
      ++slot;
    }
  }
}

// The row leader after the survivors arrived: publish this rank's count, poll all counts, and if no rank
// overflowed, finish the selection locally with three exact passes over the gathered values (bucket sums
// are exact in any order and the scans are the same, so prefix / remaining / above / kept are bitwise).
// Gather the survivors of all eight ranks into shared memory once (batched tag-checked loads), so the three
// local passes read shared memory instead of re-polling the inbox per value.
__device__ __forceinline__ void gather_to_shared(const Args& a, float* cand_s, const unsigned counts[8], int row,
                                                 unsigned tag, unsigned* watchdog) {
  const int t = threadIdx.x;
  int offset[9];
  offset[0] = 0;
  for (int r = 0; r < 8; ++r) offset[r + 1] = offset[r] + int(counts[r]);
  const int total = offset[8];
  for (int base = 0; base < total; base += threads * 8) {
    unsigned long long w[8];
    const unsigned long long start = timer();
    for (;;) {
      bool ok = true;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int g = base + k * threads + t;
        if (g < total) {
          int r = 0;
          while (g >= offset[r + 1]) ++r;
          w[k] = SysAtomic(a.peers[a.rank]->inbox.cand[row][r][g - offset[r]]).load(cuda::memory_order_relaxed);
          ok &= static_cast<unsigned>(w[k] >> 32) == tag;
        }
      }
      if (ok) break;
      if (timer() - start > 30000000000ull) {
        atomicCAS(watchdog, 0u, 0xD6FAu);
        break;
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int g = base + k * threads + t;
      if (g < total) cand_s[g] = __uint_as_float(payload(w[k]));
    }
  }
  __syncthreads();
}

// Direct exact finish for a small gathered set (total <= threads): the survivors of the boundary bucket share
// one exponent pair, so every FP64 sum over them is exact in any order. The boundary t* is the value with
// sum(u > t*) < remaining <= sum(u >= t*) (ties kept, as the digit passes do); prefix, above and kept are
// bitwise those of the three remaining passes. One thread per value, O(n^2) over shared memory.
__device__ __forceinline__ void local_finish_direct(const Args& a, Temp& temp, RowState& st, int row, const float* cand_s,
                                                    int total, unsigned tag) {
  const int t = threadIdx.x;
  const double remaining = st.remaining, previously_above = st.above;
  const unsigned pass0_prefix = st.prefix;
  double above_v = 0, mass_v = 0;
  float v = 0.0f;
  bool boundary = false;
  if (t < total) {
    v = cand_s[t];
    for (int j = 0; j < total; ++j) {
      const float u = cand_s[j];
      if (u > v) above_v += double(u);
      else if (u == v) mass_v += double(u);
    }
    boundary = above_v < remaining && above_v + mass_v >= remaining;
  }
  // exactly one distinct value satisfies the boundary test; pick its lowest thread for the write
  __shared__ int winner_s;
  if (t == 0) winner_s = threads;
  __syncthreads();
  if (boundary) atomicMin(&winner_s, t);
  __syncthreads();
  if (t == winner_s) {
    st.prefix = __float_as_uint(v);   // full 32-bit boundary encoding (the passes would OR the lower digits in)
    st.remaining = remaining - above_v;
    st.above = previously_above + above_v;
    const double kept = previously_above + above_v + mass_v;
    st.kept = kept;
    st.kept_inverse = 1.0 / kept;
    const int candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
    float raw = 0, q = 0;
    if (candidate >= 0) {
      raw = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_p[row], tag, a.watchdog));
      q = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_q[row], tag, a.watchdog));
    }
    const float pp = __float_as_uint(raw) >= st.prefix ? float(raw / kept) : 0;
    st.reject = candidate >= 0 && !(st.u_accept * q < pp);
  }
  __syncthreads();
  (void)pass0_prefix;
}

template <int pass>
__device__ __forceinline__ void local_pass(const Args& a, Temp& temp, double* hist_d, RowState& st, int row,
                                           const float* cand_s, int total, unsigned tag) {
  namespace cg = cooperative_groups;
  constexpr int shift = pass_shift(pass), width = pass_bits(pass), nbins = 1 << width, per = nbins / threads;
  constexpr unsigned mask = 0xffffffffu << (shift + width);
  const int t = threadIdx.x;
  const unsigned prefix = st.prefix;
  for (int b = t; b < nbins; b += threads) hist_d[b] = 0;
  __syncthreads();
  {
    // v19: one shared-memory FP64 atomic per value. The gathered values share one exponent pair, so
    // every bucket sum is exact in any order (that is what made the local passes bitwise in the first
    // place); the former labeled-partition reduce per warp per iteration cost several hundred cycles for
    // 256 values, 5-6 us per pass on rows with 2-4K survivors.
    const int n = total;
    for (int base = 0; base < n; base += threads) {
      const int slot = base + t;
      const float v = slot < n ? cand_s[slot] : 0.0f;
      const unsigned bits = __float_as_uint(v);
      const bool active = slot < n && (bits & mask) == prefix && v > 0;
      if (active) atomicAdd(&hist_d[(bits >> shift) & (nbins - 1)], double(v));
    }
  }
  __syncthreads();
  double mass[per], above[per];
#pragma unroll
  for (int j = 0; j < per; ++j) mass[j] = hist_d[nbins - 1 - (per * t + j)];
  const double remaining = st.remaining, previously_above = st.above;
  cub::BlockScan<double, threads>(temp.sd).ExclusiveSum(mass, above);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < per; ++j) {
    if (above[j] < remaining && above[j] + mass[j] >= remaining) {
      const int digit = nbins - 1 - (per * t + j);
      const unsigned np = prefix | (unsigned(digit) << shift);
      st.prefix = np;
      st.remaining = remaining - above[j];
      st.above = previously_above + above[j];
      if constexpr (pass == passes - 1) {
        const double kept = previously_above + above[j] + mass[j];
        st.kept = kept;
        st.kept_inverse = 1.0 / kept;
        const int candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
        float raw = 0, q = 0;
        if (candidate >= 0) {
          raw = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_p[row], tag, a.watchdog));
          q = __uint_as_float(poll(&a.peers[a.rank]->inbox.proposal_q[row], tag, a.watchdog));
        }
        const float pp = __float_as_uint(raw) >= np ? float(raw / kept) : 0;
        st.reject = candidate >= 0 && !(st.u_accept * q < pp);
      }
    }
  }
  __syncthreads();
}

// The row leader after a gather stage: publish this rank's count (stage tag), poll all counts, and unless a
// rank overflowed cand_cap or the total exceeds cand_shared, gather to shared memory and finish locally
// (direct when the set fits one thread each, else the remaining exact digit passes). Returns false when the
// row must continue with global passes. Only the leader CTA calls it.
template <int Stage>
__device__ __forceinline__ bool leader_local_finish(const Args& a, Scratch* s, Temp& temp, double* hist_d, RowState& st,
                                                    int row, unsigned tag, unsigned* counts_s, unsigned* fallback_s,
                                                    float* cand_s) {
  const unsigned stag = stage_tag(tag, Stage);
  unsigned* fallback = Stage == 1 ? &s->fallback[row] : &s->fallback2[row];
  if (threadIdx.x == 0) {
    const unsigned total = __ldcg(&s->cand_offset[row]);
    const unsigned overflow = __ldcg(fallback);
    const unsigned count = overflow || total > unsigned(cand_cap) ? unsigned(cand_cap) + 1u : total;
    for (int r = 0; r < 8; ++r) push(&a.peers[r]->inbox.cand_count[row][a.rank], tagged(stag, count));
    atomicExch(&s->cand_offset[row], 0u);
    atomicExch(fallback, 0u);
  }
  if (threadIdx.x < 8) counts_s[threadIdx.x] = poll(&a.peers[a.rank]->inbox.cand_count[row][threadIdx.x], stag, a.watchdog);
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned f = 0;
    int total = 0;
    for (int r = 0; r < 8; ++r) { f |= counts_s[r] > unsigned(cand_cap); total += int(counts_s[r]); }
    if (total > cand_shared) f = 1u;
    cand_hist(tag, row, Stage, f, unsigned(total));
    *fallback_s = f;
    s->decision[row] = f;   // read by the other CTAs of the row after the state flag
  }
  __syncthreads();
  if (*fallback_s) return false;
  int total = 0;
  for (int r = 0; r < 8; ++r) total += int(counts_s[r]);
  stamp(Stage == 1 ? 17 : 20);   // counts exchanged
  gather_to_shared(a, cand_s, counts_s, row, stag, a.watchdog);
  stamp(Stage == 1 ? 18 : 21);   // survivors in shared memory
  if (total <= threads) {
    local_finish_direct(a, temp, st, row, cand_s, total, tag);
  } else {
    if constexpr (Stage == 1) local_pass<1>(a, temp, hist_d, st, row, cand_s, total, tag);
    local_pass<2>(a, temp, hist_d, st, row, cand_s, total, tag);
    local_pass<3>(a, temp, hist_d, st, row, cand_s, total, tag);
  }
  stamp(Stage == 1 ? 19 : 22);   // selection finished
  if (threadIdx.x == 0) {
    s->prefix[row] = st.prefix;
    s->remaining[row] = st.remaining;
    s->above[row] = st.above;
    s->kept[row] = st.kept;
    s->kept_inverse[row] = st.kept_inverse;
    s->reject[row] = st.reject;
  }
  return true;
}

// Publish the row state for (epoch, pass); the other CTAs of the row wait for it.
__device__ __forceinline__ void publish_state(Scratch* s, int row, unsigned value) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    DevAtomicU(s->state_epoch[row]).store(value, cuda::memory_order_release);
  }
}
__device__ __forceinline__ void wait_state(Scratch* s, int row, unsigned value, unsigned* watchdog) {
  if (threadIdx.x == 0) wait_flag(&s->state_epoch[row], value, watchdog);
  __syncthreads();
}

__device__ __forceinline__ void normalize_nucleus(const Args& a, Scratch* s, Temp& temp, const RowState& st, int row,
                                                  int part) {
  const double inverse = st.kept_inverse;
  const unsigned threshold = st.prefix;
  const bool reject = st.reject;
  float w[items], wq[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    w[k] = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    wq[k] = reject && i < vocabulary ? a.q[row * vocabulary + i] : 0.0f;
  }
  double sum = 0;
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = part * tile + threadIdx.x + k * threads;
    if (i < vocabulary) {
      float v = w[k];
      if (__float_as_uint(v) < threshold) v = 0;
      v = float(double(v) * inverse);
      a.p[row * vocabulary + i] = v;
      sum += reject ? fmaxf(v - wq[k], 0.0f) : v;
    }
  }
  sum = ReduceD(temp.rd).Sum(sum);
  if (threadIdx.x == 0) s->sums[row * tiles + part] = sum;
  __syncthreads();
}

// Gumbel-max draw. Sampling r_i / sum r over the row is argmax_i (log r_i + g_i) with i.i.d. standard
// Gumbel g_i. Each thread runs one Philox stream per (domain, row, first token of the thread) and takes
// eight uniforms from it; ties broken by the smaller index so every rank agrees. Exactly the same marginal
// as the CDF draw, but different tokens for the same seed.
__device__ __forceinline__ void gumbel_tile(const Args& a, Scratch* s, ReduceKI::TempStorage& temp, const RowState& st,
                                            int row, int part) {
  const bool reject = st.reject;
  const int t = threadIdx.x;
  const int first = part * tile + t;
  curandStatePhilox4_32_10_t rng;
  curand_init(static_cast<unsigned long long>(a.rng[0]),
              (static_cast<unsigned long long>(a.anchor ? 3u : 2u) << 32) | unsigned(first),
              static_cast<unsigned long long>(a.rng[1]) * 8 + row, &rng);
  const float4 u0 = curand_uniform4(&rng), u1 = curand_uniform4(&rng);
  const float u[items] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  float w[items], wq[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = first + k * threads;
    w[k] = i < vocabulary ? __ldcg(&a.p[row * vocabulary + i]) : 0.0f;
    wq[k] = reject && i < vocabulary ? __ldcg(&a.q[row * vocabulary + i]) : 0.0f;
  }
  KeyIndex best{-INFINITY, 0x7fffffff};
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = first + k * threads;
    const float r = reject ? fmaxf(w[k] - wq[k], 0.0f) : w[k];
    if (i < vocabulary && r > 0.0f) {
      const float uu = fminf(fmaxf(u[k], 0x1p-32f), 0x1.fffffep-1f);   // (0,1): curand_uniform is (0,1]
      const float key = __logf(r) - __logf(-__logf(uu));
      if (key > best.key || (key == best.key && i < best.index)) best = KeyIndex{key, i};
    }
  }
  best = ReduceKI(temp).Reduce(best, MaxKeyIndex());
  if (t == 0) {
    s->gumbel_key[row * tiles + part] = best.key;
    s->gumbel_index[row * tiles + part] = best.index;
  }
  __syncthreads();
}

// Fused: nucleus normalization (the same p_trunc stores as normalize_nucleus) + Gumbel key of the tile.
__device__ __forceinline__ void nucleus_gumbel_tile(const Args& a, Scratch* s, ReduceKI::TempStorage& temp,
                                                    const RowState& st, int row, int part) {
  const double inverse = st.kept_inverse;
  const unsigned threshold = st.prefix;
  const bool reject = st.reject;
  const int t = threadIdx.x;
  const int first = part * tile + t;
  curandStatePhilox4_32_10_t rng;
  curand_init(static_cast<unsigned long long>(a.rng[0]),
              (static_cast<unsigned long long>(a.anchor ? 3u : 2u) << 32) | unsigned(first),
              static_cast<unsigned long long>(a.rng[1]) * 8 + row, &rng);
  const float4 u0 = curand_uniform4(&rng), u1 = curand_uniform4(&rng);
  const float u[items] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  float w[items], wq[items];
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = first + k * threads;
    w[k] = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    wq[k] = reject && i < vocabulary ? a.q[row * vocabulary + i] : 0.0f;
  }
  KeyIndex best{-INFINITY, 0x7fffffff};
#pragma unroll
  for (int k = 0; k < items; ++k) {
    const int i = first + k * threads;
    if (i < vocabulary) {
      float v = w[k];
      if (__float_as_uint(v) < threshold) v = 0;
      v = float(double(v) * inverse);
      a.p[row * vocabulary + i] = v;
      const float r = reject ? fmaxf(v - wq[k], 0.0f) : v;
      if (r > 0.0f) {
        const float uu = fminf(fmaxf(u[k], 0x1p-32f), 0x1.fffffep-1f);
        const float key = __logf(r) - __logf(-__logf(uu));
        if (key > best.key || (key == best.key && i < best.index)) best = KeyIndex{key, i};
      }
    }
  }
  best = ReduceKI(temp).Reduce(best, MaxKeyIndex());
  if (t == 0) {
    s->gumbel_key[row * tiles + part] = best.key;
    s->gumbel_index[row * tiles + part] = best.index;
  }
  __syncthreads();
}

// Leader: reduce the row's tiles, push (key, global index) to all ranks, poll all eight, pick the token.
__device__ __forceinline__ void gumbel_choose(const Args& a, Scratch* s, ReduceKI::TempStorage& temp, const RowState& st,
                                              int row, unsigned tag) {
  const int t = threadIdx.x;
  KeyIndex v{-INFINITY, 0x7fffffff};
  if (t < tiles) v = KeyIndex{__ldcg(&s->gumbel_key[row * tiles + t]), __ldcg(&s->gumbel_index[row * tiles + t])};
  v = ReduceKI(temp).Reduce(v, MaxKeyIndex());
  __shared__ KeyIndex local_s;
  if (t == 0) {
    local_s = v;
    const int global_index = v.index < vocabulary ? a.rank * vocabulary + v.index : full_vocabulary;
    for (int r = 0; r < 8; ++r) {
      push(&a.peers[r]->inbox.gumbel[row][a.rank][0], tagged(tag, __float_as_uint(v.key)));
      push(&a.peers[r]->inbox.gumbel[row][a.rank][1], tagged(tag, unsigned(global_index)));
    }
  }
  __syncthreads();
  if (t < 32) {
    KeyIndex mine{-INFINITY, 0x7fffffff};
    if (t < 8) {
      unsigned long long w0, w1;
      const unsigned long long start = timer();
      for (;;) {
        w0 = SysAtomic(a.peers[a.rank]->inbox.gumbel[row][t][0]).load(cuda::memory_order_relaxed);
        w1 = SysAtomic(a.peers[a.rank]->inbox.gumbel[row][t][1]).load(cuda::memory_order_relaxed);
        if (static_cast<unsigned>(w0 >> 32) == tag && static_cast<unsigned>(w1 >> 32) == tag) break;
        if (timer() - start > 30000000000ull) {
          atomicCAS(a.watchdog, 0u, 0xD6FAu);
          break;
        }
      }
      mine = KeyIndex{__uint_as_float(payload(w0)), int(payload(w1))};
    }
    KeyIndex best = mine;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const KeyIndex o{__shfl_sync(0xffffffffu, mine.key, r), __shfl_sync(0xffffffffu, mine.index, r)};
      best = MaxKeyIndex()(best, o);
    }
    if (t == 0) {
      int token = best.index;
      if (!a.anchor && row < 7 && !st.reject) token = int(a.proposals[row]);
      if (!isfinite(st.kept) || st.kept <= 0 || best.key == -INFINITY || token >= full_vocabulary) {
        atomicCAS(a.watchdog, 0u, 0xD6FBu);
        token = 0;
      }
      a.tokens[row] = token;
    }
  }
  __syncthreads();
}

// local_cdf (publishes the rank mass) then choose_tile + choose_token, then the winner exchange.
__device__ __forceinline__ void cdf_choose_broadcast(const Args& a, Scratch* s, Temp& temp, const RowState& st, int row,
                                                     unsigned tag) {
  __shared__ double cut, base, mass_s, base_s;
  __shared__ int tile_s;
  const int t = threadIdx.x;
  const double total = t < tiles ? __ldcg(&s->sums[row * tiles + t]) : 0.0;
  const double local = ReduceD(temp.rd).Sum(total);
  if (t == 0)
    for (int r = 0; r < 8; ++r) push_double(a.peers[r]->inbox.mass[row][a.rank], tag, local);
  __syncthreads();
  double prefix;
  ScanD(temp.sd).ExclusiveSum(total, prefix);
  if (t < tiles) s->chunk_prefix[row * tiles + t] = prefix;
  __syncthreads();
  if (t < 32) {
    double preceding = 0;
    const double mass = warp_poll_sum8(a.peers[a.rank]->inbox.mass[row], tag, a.watchdog, a.rank, &preceding);
    if (t == 0) {
      mass_s = mass;
      base_s = preceding;
      tile_s = -1;
      base = preceding;
      cut = double(st.u_draw) * mass;
    }
  }
  __syncthreads();
  if (t < tiles) {
    const double p = base + __ldcg(&s->chunk_prefix[row * tiles + t]);
    if (p <= cut && p + total > cut) tile_s = t;
  }
  __syncthreads();
  const int part = tile_s;
  const bool reject = st.reject;
  double values[items], sum = 0;
  for (int j = 0; j < items; ++j) {
    const int i = part * tile + t * items + j;
    float v = part >= 0 && i < vocabulary ? __ldcg(&a.p[row * vocabulary + i]) : 0.0f;
    if (reject && part >= 0 && i < vocabulary) v = fmaxf(v - __ldcg(&a.q[row * vocabulary + i]), 0.0f);
    values[j] = v;
    sum += v;
  }
  double running;
  ScanD(temp.sd).ExclusiveSum(sum, running);
  __syncthreads();
  running += base_s + (part >= 0 ? __ldcg(&s->chunk_prefix[row * tiles + part]) : 0.0);
  const double draw = double(st.u_draw) * mass_s;
  int winner = vocabulary;
  for (int j = 0; j < items; ++j) {
    running += values[j];
    if (values[j] > 0 && running > draw) {
      winner = part * tile + t * items + j;
      break;
    }
  }
  winner = ReduceI(temp.ri).Reduce(winner, Minimum());
  if (t == 0) {
    winner = winner < vocabulary ? a.rank * vocabulary + winner : full_vocabulary;
    if (!a.anchor && row < 7 && !reject) winner = int(a.proposals[row]);
    if (!isfinite(st.kept) || st.kept <= 0 || !isfinite(mass_s) || mass_s <= 0) {
      atomicCAS(a.watchdog, 0u, 0xD6FBu);
      winner = 0;
    }
    for (int r = 0; r < 8; ++r) push(&a.peers[r]->inbox.winner[row][a.rank], tagged(tag, unsigned(winner)));
  }
  __syncthreads();
  if (t < 32) {   // broadcast_tokens: minimum over the eight winners, polled by eight lanes
    const int token_min = warp_poll_min8(a.peers[a.rank]->inbox.winner[row], tag, a.watchdog);
    if (t == 0) {
      int token = token_min;
      if (token >= full_vocabulary) {
        atomicCAS(a.watchdog, 0u, 0xD6FBu);
        token = 0;
      }
      a.tokens[row] = token;
    }
  }
  __syncthreads();
}

// ---- the kernel --------------------------------------------------------------

__global__ void __launch_bounds__(threads) sampler(Args a, Scratch* s, bool defer_consumed) {
  namespace cg = cooperative_groups;
  __shared__ Temp temp;
  __shared__ union {
    unsigned long long u[max_bins];
    double d[max_bins];
  } hist;
  __shared__ RowState st;
  const int rows = a.anchor ? 1 : max_rows;
  const int row = blockIdx.x / tiles, part = blockIdx.x - row * tiles;
  const bool lead = part == 0;
  const bool has_q = !a.anchor && row < 7;
  const int qd = 8 + row;
  Mailbox* own = a.peers[a.rank];

  // Tag generation for this invocation: the mailbox's own invocation counter (zeroed once with the
  // mailbox, never by a scratch reset, so tags never repeat). One grid barrier makes it visible.
  if (blockIdx.x == 0 && threadIdx.x == 0) s->epoch = static_cast<unsigned>(++own->inbox.epoch);
  cg::this_grid().sync();
  const unsigned tag = __ldcg(&s->epoch);
  stamp(0);

  if constexpr (PERSISTENT_ONLINE_SOFTMAX) {
    // A': tile maxima -> row arrival; the leader publishes the rank-local maxima to the row (scratch + flag).
    // CTA 0 runs the external ready rendezvous meanwhile; the first mailbox publish is after the exponentials.
    partial_maximum(a, s, temp, row, part);
    if (has_q) partial_maximum(a, s, temp, qd, part);
    row_arrive(s, row, 0);
    stamp(1);
    if (row == 0 && part == tiles - 1) {   // a non-leader CTA runs the external ready rendezvous off the critical path
      if (PERSISTENT_EXTERNAL && PERSISTENT_READY && threadIdx.x < 32) external_rendezvous(a, false);
      __syncthreads();
      if (threadIdx.x == 0) DevAtomicU(s->ready_epoch).store(tag, cuda::memory_order_release);
    }
    if (lead) {
      row_wait(s, row, 0, a.watchdog);
      for (int k = 0; k < (has_q ? 2 : 1); ++k) {
        const int d = k ? qd : row;
        const float v = threadIdx.x < tiles ? __ldcg(&s->maxima[d * tiles + threadIdx.x]) : -INFINITY;
        const float m = ReduceF(temp.rf).Reduce(v, Maximum());
        if (threadIdx.x == 0) s->local_max[d] = m;
        __syncthreads();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        DevAtomicU(s->lmax_epoch[row]).store(tag, cuda::memory_order_release);   // local maxima ready
      }
    }
    if (threadIdx.x == 0) wait_flag(&s->lmax_epoch[row], tag, a.watchdog);
    __syncthreads();
    stamp(2);
    __shared__ float lm_s[2];
    if (threadIdx.x < 2) lm_s[threadIdx.x] = __ldcg(&s->local_max[threadIdx.x ? qd : row]);
    __syncthreads();
    stamp(3);
    if (has_q) exponentiate2(a, s, temp, row, qd, part, lm_s[0], lm_s[1]);
    else exponentiate(a, s, temp, row, part, lm_s[0]);
    row_arrive(s, row, 1);
    stamp(4);
    if (lead) {
      row_wait(s, row, 1, a.watchdog);
      if (threadIdx.x == 0) wait_flag(&s->ready_epoch, tag, a.watchdog);   // no publish before ready (every leader)
      __syncthreads();
      for (int k = 0; k < (has_q ? 2 : 1); ++k) {
        const int d = k ? qd : row;
        const double sum = ReduceD(temp.rd).Sum(threadIdx.x < tiles ? __ldcg(&s->sums[d * tiles + threadIdx.x]) : 0.0);
        if (threadIdx.x == 0)
          for (int r = 0; r < 8; ++r) {
            push(&a.peers[r]->inbox.maximum[d][a.rank], tagged(tag, __float_as_uint(lm_s[k])));
            push_double(a.peers[r]->inbox.sum[d][a.rank], tag, sum);
          }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0) {
      st.prefix = 0;
      st.remaining = top_p_of(a);
      st.above = 0;
      st.kept = 0;
      st.kept_inverse = 0;
      st.reject = 0;
    }
    stamp(5);
    {
      // global maximum M and Z = sum_r s_r * exp(m_r - M); this rank's scale = exp(m_r - M) / Z
      __shared__ float inv_s[2];
      const int wp = threadIdx.x >> 5;
      if (threadIdx.x == 64) st.u_accept = naivert_dspark_ops::sampling_uniform(a.rng, 1, row);
      if (threadIdx.x == 96) st.u_draw = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row);
      if (wp < (has_q ? 2 : 1)) {
        const int d = wp ? qd : row;
        const int lane = threadIdx.x & 31;
        float m = -INFINITY;
        double sm = 0;
        if (lane < 8) {
          m = __uint_as_float(poll(&own->inbox.maximum[d][lane], tag, a.watchdog));
          sm = poll_double(own->inbox.sum[d][lane], tag, a.watchdog);
        }
        float M = -INFINITY;
#pragma unroll
        for (int r = 0; r < 8; ++r) M = fmaxf(M, __shfl_sync(0xffffffffu, m, r));
        double Z = 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float mr = __shfl_sync(0xffffffffu, m, r);
          const double sr = __shfl_sync(0xffffffffu, sm, r);
          Z += sr * exp(double(mr) - double(M));
        }
        if (lane == 0) inv_s[wp] = float(exp(double(lm_s[wp]) - double(M)) / Z);
      }
      __syncthreads();
      stamp(6);
      if (has_q) normalize2(a, row, qd, part, inv_s[0], inv_s[1], tag);
      else normalize(a, row, part, inv_s[0], tag);
      __syncthreads();
    }
  } else {
    // A: logits copy + tile maxima -> row arrival. CTA 0 runs the external ready rendezvous meanwhile.
    partial_maximum(a, s, temp, row, part);
    if (has_q) partial_maximum(a, s, temp, qd, part);
    row_arrive(s, row, 0);
    stamp(1);
    if (lead) {
      if (row == 0) {
        if (PERSISTENT_EXTERNAL && PERSISTENT_READY && threadIdx.x < 32) external_rendezvous(a, false);
        __syncthreads();
        if (threadIdx.x == 0) DevAtomicU(s->ready_epoch).store(tag, cuda::memory_order_release);
      }
      row_wait(s, row, 0, a.watchdog);
      if (row != 0 && threadIdx.x == 0) wait_flag(&s->ready_epoch, tag, a.watchdog);   // no publish before ready
      __syncthreads();
      for (int k = 0; k < (has_q ? 2 : 1); ++k) {
        const int d = k ? qd : row;
        const float v = threadIdx.x < tiles ? __ldcg(&s->maxima[d * tiles + threadIdx.x]) : -INFINITY;
        const float m = ReduceF(temp.rf).Reduce(v, Maximum());
        if (threadIdx.x == 0)
          for (int r = 0; r < 8; ++r) push(&a.peers[r]->inbox.maximum[d][a.rank], tagged(tag, __float_as_uint(m)));
        __syncthreads();
      }
    }
    stamp(2);
    // Every CTA of the row polls its local inbox for the eight maxima (same fmax order as sharded).
    {
      __shared__ float peak_s[2];
      const int wp = threadIdx.x >> 5;
      if (threadIdx.x == 64) st.u_accept = naivert_dspark_ops::sampling_uniform(a.rng, 1, row);
      if (threadIdx.x == 96) st.u_draw = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row);
      if (wp < (has_q ? 2 : 1)) {
        const float peak = warp_poll_fmax8(own->inbox.maximum[wp ? qd : row], tag, a.watchdog);
        if ((threadIdx.x & 31) == 0) peak_s[wp] = peak;
      }
      __syncthreads();
      stamp(3);
      if (has_q) exponentiate2(a, s, temp, row, qd, part, peak_s[0], peak_s[1]);
      else exponentiate(a, s, temp, row, part, peak_s[0]);
    }
    row_arrive(s, row, 1);
    stamp(4);
    if (lead) {
      row_wait(s, row, 1, a.watchdog);
      for (int k = 0; k < (has_q ? 2 : 1); ++k) {
        const int d = k ? qd : row;
        const double sum = ReduceD(temp.rd).Sum(threadIdx.x < tiles ? __ldcg(&s->sums[d * tiles + threadIdx.x]) : 0.0);
        if (threadIdx.x == 0)
          for (int r = 0; r < 8; ++r) push_double(a.peers[r]->inbox.sum[d][a.rank], tag, sum);
        __syncthreads();
      }
    }
    if (threadIdx.x == 0) {   // row state for the radix passes (identical in every CTA of the row)
      st.prefix = 0;
      st.remaining = top_p_of(a);
      st.above = 0;
      st.kept = 0;
      st.kept_inverse = 0;
      st.reject = 0;
    }
    stamp(5);
    {
      __shared__ float inv_s[2];
      const int wp = threadIdx.x >> 5;
      if (wp < (has_q ? 2 : 1)) {
        const double sum = warp_poll_sum8(own->inbox.sum[wp ? qd : row], tag, a.watchdog, a.rank);
        if ((threadIdx.x & 31) == 0) inv_s[wp] = float(1.0 / sum);
      }
      __syncthreads();
      stamp(6);
      if (has_q) normalize2(a, row, qd, part, inv_s[0], inv_s[1], tag);
      else normalize(a, row, part, inv_s[0], tag);
      __syncthreads();
    }
  }
  stamp(7);

  // Radix passes: histogram of the own tile -> row generation -> every CTA merges its digit slice,
  // pushes, polls the row histogram of all sources locally, and selects (identical state in every CTA).
  radix_histogram<0>(a, s, hist.u, hist.d, row, part, st.prefix);
  row_arrive(s, row, 2, tag);
  row_wait_all(s, row, 2, tag, a.watchdog);
  merge_exchange_select<0>(a, s, temp, st, row, part, tag);
  stamp(8);
  bool global_passes = true;
  __shared__ unsigned counts_s[8];
  __shared__ unsigned fallback_s;
  __shared__ float cand_s[cand_shared];
  auto copy_state = [&]() {
    if (!lead && threadIdx.x == 0) {
      st.prefix = __ldcg(&s->prefix[row]);
      st.remaining = __ldcg(&s->remaining[row]);
      st.above = __ldcg(&s->above[row]);
      st.kept = __ldcg(&s->kept[row]);
      st.kept_inverse = __ldcg(&s->kept_inverse[row]);
      st.reject = __ldcg(&s->reject[row]);
    }
    __syncthreads();
  };
  if constexpr (PERSISTENT_GATHER && passes == 4) {
    // stage 1: gather the boundary bucket's survivors; the leader finishes locally unless some rank overflowed
    gather_survivors<1>(a, s, st, row, part, tag);
    row_arrive(s, row, 3);   // counter only: the leader waits and resets (row_wait)
    if (lead) {
      row_wait(s, row, 3, a.watchdog);
      leader_local_finish<1>(a, s, temp, hist.d, st, row, tag, counts_s, &fallback_s, cand_s);
      publish_state(s, row, tag * 8 + 1);
    }
    wait_state(s, row, tag * 8 + 1, a.watchdog);
    global_passes = __ldcg(&s->decision[row]) != 0u;
    if (!global_passes) copy_state();
    stamp(9);
  }
  if (global_passes) {
    radix_histogram<1>(a, s, hist.u, hist.d, row, part, st.prefix);
    row_arrive(s, row, 4, tag);
    row_wait_all(s, row, 4, tag, a.watchdog);
    merge_exchange_select<1>(a, s, temp, st, row, part, tag);
    if constexpr (!(PERSISTENT_GATHER && passes == 4)) stamp(9);
    bool global_2 = true;
    if constexpr (PERSISTENT_GATHER && passes == 4) {
      // stage 2 (v18): the pass-1 bucket is 1/256 of the pass-0 bucket; gather it and finish locally
      // instead of two more global exchange passes. Phase 5 is a counter here; a global pass 2 reuses it.
      gather_survivors<2>(a, s, st, row, part, stage_tag(tag, 2));
      row_arrive(s, row, 5);
      if (lead) {
        row_wait(s, row, 5, a.watchdog);
        leader_local_finish<2>(a, s, temp, hist.d, st, row, tag, counts_s, &fallback_s, cand_s);
        publish_state(s, row, tag * 8 + 2);
      }
      wait_state(s, row, tag * 8 + 2, a.watchdog);
      global_2 = __ldcg(&s->decision[row]) != 0u;
      if (!global_2) copy_state();
      stamp(10);
    }
    if (global_2) {
      radix_histogram<2>(a, s, hist.u, hist.d, row, part, st.prefix);
      row_arrive(s, row, 5, tag);
      row_wait_all(s, row, 5, tag, a.watchdog);
      merge_exchange_select<2>(a, s, temp, st, row, part, tag);
      if constexpr (!(PERSISTENT_GATHER && passes == 4)) stamp(10);
      if constexpr (passes == 4) {
        radix_histogram<3>(a, s, hist.u, hist.d, row, part, st.prefix);
        row_arrive(s, row, 6, tag);
        row_wait_all(s, row, 6, tag, a.watchdog);
        merge_exchange_select<3>(a, s, temp, st, row, part, tag);
      }
    }
  }
  stamp(11);

  // H: nucleus normalization -> row arrival -> part-0: CDF, mass exchange, token, winner exchange.
  if constexpr (PERSISTENT_GUMBEL) nucleus_gumbel_tile(a, s, temp.ki, st, row, part);
  else normalize_nucleus(a, s, temp, st, row, part);
  row_arrive(s, row, 7);
  stamp(12);
  if (lead) {
    row_wait(s, row, 7, a.watchdog);
    if constexpr (PERSISTENT_GUMBEL) gumbel_choose(a, s, temp.ki, st, row, tag);
    else cdf_choose_broadcast(a, s, temp, st, row, tag);
    // grid-level arrival of the row leaders; the last runs the external consumed rendezvous
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      const unsigned previous = atomicAdd(&s->counter[0], 1u);
      last = previous + 1 == unsigned(rows);
      if (last) atomicExch(&s->counter[0], 0u);
    }
    __syncthreads();
    stamp(13);
    if (PERSISTENT_EXTERNAL && last && !defer_consumed && threadIdx.x < 32) external_rendezvous(a, true);
  }
  stamp(14);
}

#ifndef PERSISTENT_SMEM_KB
#define PERSISTENT_SMEM_KB 0
#endif
// Mailboxes/flags zero-initialized once; scratch zero before the first invocation
// (cudaMemset). Every rank must use the same anchor mode. Cooperative launch:
// rows x tiles CTAs, all co-resident.
inline cudaError_t run(Args a, Scratch* s, bool defer_consumed = false, cudaStream_t stream = nullptr) {
  const int rows = a.anchor ? 1 : max_rows;
  constexpr int smem = PERSISTENT_SMEM_KB * 1024;
  static bool configured[16] = {};
  int device = 0;
  cudaGetDevice(&device);
  if (smem > 0 && !configured[device]) {
    const auto e = cudaFuncSetAttribute(reinterpret_cast<const void*>(&sampler), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    configured[device] = true;
  }
  void* args[] = {&a, &s, &defer_consumed};
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(&sampler), dim3(tiles * rows), dim3(threads), args,
                                     smem, stream);
}
}  // namespace naivert_dspark_persistent
