#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "random.cuh"
#include "naivert/body/dspark_common/temperature.cuh"
#include "sync.cuh"

// TP8-sharded DSpark T=1, top-p=.95 sampler. Each rank handles 19,072
// vocabulary entries. Ordinary kernel boundaries and the existing TP rendezvous
// protocol synchronize phases; there is no full-logit gather or replay
// allocation.
namespace naivert_dspark_sharded {
constexpr int full_vocabulary = 152576;
constexpr int vocabulary = full_vocabulary / 8;
constexpr int threads = 256;
constexpr int items = 8;
constexpr int tile = threads * items;
constexpr int tiles = (vocabulary + tile - 1) / tile;
constexpr int bins = 256;

// Each published field has one writer. Phase-specific fields remain immutable
// until the next invocation's initial TP8 rendezvous, so readers cannot race a
// faster rank reusing a histogram or proposal probability.
struct Mailbox {
  unsigned long long flags[8][3];
  float maximum[15];
  double sum[15];
  double histogram[4][8][bins];
  float proposal_p[7], proposal_q[7];
  double mass[8];
  int winner[8];
};

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

struct Scratch {
  float maxima[15 * tiles];
  double sums[15 * tiles];
  float peak[15], inverse[15];
  double histogram[8 * tiles * bins];
  unsigned prefix[8];
  double remaining[8], above[8], kept[8], kept_inverse[8], mass[8],
      rank_base[8], chunk_prefix[8 * tiles];
  int reject[8], selected_tile[8];
};

struct Maximum {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Minimum {
  __device__ int operator()(int a, int b) const { return min(a, b); }
};
__device__ inline float* distribution(Args a, int row) {
  return row < 8 ? a.p + row * vocabulary : a.q + (row - 8) * vocabulary;
}

__global__ void partial_maximum(Args a, Scratch* s) {
  using R = cub::BlockReduce<float, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  float* x = distribution(a, row);
  const float* input = row < 8 ? a.logits + row * vocabulary
                               : a.draft_logits + (row - 8) * vocabulary;
  float v = -INFINITY;
  const float inv = inv_temperature_of(a);
  const bool penalize = row < 8 && a.sampling_params != nullptr && a.sampling_params[3] != 0.0f;
  const float theta = penalize ? a.sampling_params[0] : 1.0f, alpha = penalize ? a.sampling_params[1] : 0.0f,
              beta = penalize ? a.sampling_params[2] : 0.0f;
  const int64_t base = int64_t(a.rank) * vocabulary;
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) {
      float l = input[i];
      if (penalize) {
        int c = a.rep_count[base + i];
        for (int b = 0; b < row && b < 7; ++b) c += a.proposals[b] == base + i ? 1 : 0;
        if (c > 0) {
          l = l > 0.0f ? l / theta : l * theta;
          l -= alpha * float(c) + beta;
        }
      }
      const float t = naivert_dspark::tempered(l, inv);
      x[i] = t;
      v = fmaxf(v, t);
    }
  }
  v = R(temp).Reduce(v, Maximum());
  if (threadIdx.x == 0) s->maxima[row * tiles + part] = v;
}

__global__ void local_maximum(Args a, Scratch* s) {
  using R = cub::BlockReduce<float, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.x, t = threadIdx.x;
  float v = t < tiles ? s->maxima[row * tiles + t] : -INFINITY;
  v = R(temp).Reduce(v, Maximum());
  if (t == 0) a.peers[a.rank]->maximum[row] = v;
}

__global__ void exponentiate(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  float* x = distribution(a, row);
  const float peak = s->peak[row];
  double sum = 0;
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) {
      float v = expf(x[i] - peak);
      x[i] = v;
      sum += v;
    }
  }
  sum = R(temp).Sum(sum);
  if (threadIdx.x == 0) s->sums[row * tiles + part] = sum;
}

__global__ void local_sum(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  int row = blockIdx.x, t = threadIdx.x;
  double sum = R(temp).Sum(t < tiles ? s->sums[row * tiles + t] : 0.0);
  if (t == 0) a.peers[a.rank]->sum[row] = sum;
}

__global__ void rendezvous(Args a, int stage, bool consumed = false) {
  // Same release/acquire generation protocol as sampling_rendezvous, with the
  // eight peer polls issued by eight lanes instead of serially by one thread.
  using Atomic = naivert_dspark_ops::Atomic;
  const int lane = threadIdx.x;
  auto* own = stage < 0 ? a.flags[a.rank] : a.peers[a.rank]->flags[stage];
  unsigned long long generation = 0;
  if (lane == 0) {
    generation = consumed ? own[2] : ++own[2];
    __threadfence_system();
    Atomic(own[consumed ? 1 : 0]).store(generation, cuda::memory_order_release);
  }
  generation = __shfl_sync(0xffffffffu, generation, 0);
  if (lane < 8) {
    auto* peer = stage < 0 ? a.flags[lane] : a.peers[lane]->flags[stage];
    unsigned long long start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    while (Atomic(peer[consumed ? 1 : 0]).load(cuda::memory_order_acquire) <
           generation) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - start > 30000000000ull) {
        atomicCAS(a.watchdog, 0u, 0xD6FAu);
        break;
      }
      __nanosleep(64);
    }
  }
}

__global__ void global_maximum(Args a, Scratch* s) {
  int row = threadIdx.x;
  if (row < (a.anchor ? 1 : 15)) {
    float peak = -INFINITY;
    for (int r = 0; r < 8; ++r) peak = fmaxf(peak, a.peers[r]->maximum[row]);
    s->peak[row] = peak;
  }
}

__global__ void global_inverse(Args a, Scratch* s) {
  int row = threadIdx.x;
  if (row < (a.anchor ? 1 : 15)) {
    double sum = 0;
    for (int r = 0; r < 8; ++r) sum += a.peers[r]->sum[row];
    s->inverse[row] = float(1.0 / sum);
    if (row < 8) {
      s->prefix[row] = 0;
      s->remaining[row] = top_p_of(a);
      s->above[row] = 0;
    }
  }
}

__global__ void normalize(Args a, const Scratch* s) {
  const int row = blockIdx.y, part = blockIdx.x;
  float* x = distribution(a, row);
  const float inv = s->inverse[row];
  for (int j = threadIdx.x; j < tile; j += threads) {
    int i = part * tile + j;
    if (i < vocabulary) {
      const float v = x[i] * inv;
      x[i] = v;
      const int proposal_row = row < 8 ? row : row - 8;
      if (!a.anchor && proposal_row < 7 &&
          a.proposals[proposal_row] == a.rank * vocabulary + i) {
        if (row < 8)
          a.peers[a.rank]->proposal_p[proposal_row] = v;
        else
          a.peers[a.rank]->proposal_q[proposal_row] = v;
      }
    }
  }
}

// Weighted radix histograms. Labeled warp groups combine identical digits
// before the shared-memory atomic. Later passes skip warps with no survivors.
template <int shift>
__global__ void radix_histogram(Args a, Scratch* s) {
  namespace cg = cooperative_groups;
  if constexpr (shift == 24) {
    // A positive FP32 high-byte bucket spans at most two adjacent exponents.
    // Express each value in the bucket's common binary unit. At most 32 values
    // each less than 2^25 enter a warp reduction, so its uint32 sum cannot
    // overflow. A tile's uint64 sum is exact; scaling it back to FP64 is exact.
    __shared__ unsigned long long hist[bins];
    const int t = threadIdx.x, row = blockIdx.y, part = blockIdx.x;
    hist[t] = 0;
    __syncthreads();
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    for (int j = t; j < tile; j += threads) {
      const int i = part * tile + j;
      const float v = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
      const unsigned bits = __float_as_uint(v), digit = bits >> 24;
      const unsigned exponent = (bits >> 23) & 255;
      unsigned units = (bits & 0x7fffff) | (exponent ? 0x800000 : 0);
      if (digit) units <<= exponent & 1;
      const auto group = cg::labeled_partition(warp, digit);
      const unsigned mass = cg::reduce(group, units, cg::plus<unsigned>());
      if (group.thread_rank() == 0 && mass)
        atomicAdd(&hist[digit], static_cast<unsigned long long>(mass));
    }
    __syncthreads();
    const int power = t ? 2 * t - 150 : -149;
    const double unit = __longlong_as_double(static_cast<long long>(power + 1023) << 52);
    s->histogram[(row * tiles + part) * bins + t] = double(hist[t]) * unit;
  } else {
    __shared__ double hist[bins];
    const int t = threadIdx.x, row = blockIdx.y, part = blockIdx.x;
    hist[t] = 0;
    __syncthreads();
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const unsigned prefix = s->prefix[row];
    constexpr unsigned mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int j = t; j < tile; j += threads) {
      const int i = part * tile + j;
      const float v = i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
      const unsigned bits = __float_as_uint(v);
      const bool active = i < vocabulary && (bits & mask) == prefix;
      if (!__any_sync(0xffffffffu, active && v > 0)) continue;
      const unsigned digit = (bits >> shift) & 255;
      const auto group = cg::labeled_partition(warp, digit);
      const double mass =
          cg::reduce(group, active ? double(v) : 0.0, cg::plus<double>());
      if (group.thread_rank() == 0 && mass != 0) atomicAdd(&hist[digit], mass);
    }
    __syncthreads();
    s->histogram[(row * tiles + part) * bins + t] = hist[t];
  }
}

template <int shift>
__global__ void merge_histogram(Args a, const Scratch* s) {
  int row = blockIdx.x, digit = threadIdx.x;
  double sum = 0;
  for (int part = 0; part < tiles; ++part)
    sum += s->histogram[(row * tiles + part) * bins + digit];
  a.peers[a.rank]->histogram[(24 - shift) / 8][row][digit] = sum;
}

template <int shift>
__global__ void select_digit(Args a, Scratch* s) {
  using Scan = cub::BlockScan<double, threads>;
  __shared__ Scan::TempStorage temp;
  const int t = threadIdx.x, row = blockIdx.x, digit = 255 - t;
  double mass = 0;
  const double remaining = s->remaining[row];
  const double previously_above = s->above[row];
  for (int r = 0; r < 8; ++r)
    mass += a.peers[r]->histogram[(24 - shift) / 8][row][digit];
  double above;
  Scan(temp).ExclusiveSum(mass, above);
  __syncthreads();
  if (above < remaining && above + mass >= remaining) {
    s->prefix[row] |= unsigned(digit) << shift;
    s->remaining[row] = remaining - above;
    s->above[row] = previously_above + above;
    if constexpr (shift == 0) {
      // The retained set is all higher buckets plus the entire boundary bucket.
      // Its FP64 mass is already available; no extra vocabulary reduction.
      const double kept = previously_above + above + mass;
      s->kept[row] = kept;
      s->kept_inverse[row] = 1.0 / kept;
      const int candidate = !a.anchor && row < 7 ? int(a.proposals[row]) : -1;
      const int owner = candidate >= 0 ? candidate / vocabulary : 0;
      const float raw = candidate >= 0 ? a.peers[owner]->proposal_p[row] : 0;
      const float p =
          __float_as_uint(raw) >= s->prefix[row] ? float(raw / kept) : 0;
      s->reject[row] = candidate >= 0 &&
                       !(naivert_dspark_ops::sampling_uniform(a.rng, 1, row) *
                             a.peers[owner]->proposal_q[row] <
                         p);
    }
  }
}

__global__ void normalize_nucleus(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  __shared__ R::TempStorage temp;
  const int row = blockIdx.y, part = blockIdx.x;
  const double inverse = s->kept_inverse[row];
  const unsigned threshold = s->prefix[row];
  const bool reject = s->reject[row];
  double sum = 0;
  for (int j = threadIdx.x; j < tile; j += threads) {
    const int i = part * tile + j;
    if (i < vocabulary) {
      float& v = a.p[row * vocabulary + i];
      if (__float_as_uint(v) < threshold) v = 0;
      v = float(double(v) * inverse);
      sum += reject ? fmaxf(v - a.q[row * vocabulary + i], 0.0f) : v;
    }
  }
  sum = R(temp).Sum(sum);
  if (threadIdx.x == 0) s->sums[row * tiles + part] = sum;
}

__global__ void local_cdf(Args a, Scratch* s) {
  using R = cub::BlockReduce<double, threads>;
  using Scan = cub::BlockScan<double, threads>;
  __shared__ union {
    R::TempStorage r;
    Scan::TempStorage scan;
  } temp;
  int row = blockIdx.x, t = threadIdx.x;
  double total = t < tiles ? s->sums[row * tiles + t] : 0.0;
  double sum = R(temp.r).Sum(total);
  if (t == 0) a.peers[a.rank]->mass[row] = sum;
  __syncthreads();
  double prefix;
  Scan(temp.scan).ExclusiveSum(total, prefix);
  if (t < tiles) s->chunk_prefix[row * tiles + t] = prefix;
}

__device__ __forceinline__ void choose_tile(Args a, Scratch* s) {
  __shared__ double cut, base;
  int row = blockIdx.x, t = threadIdx.x;
  if (t == 0) {
    double mass = 0, preceding = 0;
    for (int r = 0; r < 8; ++r) {
      double v = a.peers[r]->mass[row];
      if (r < a.rank) preceding += v;
      mass += v;
    }
    s->mass[row] = mass;
    s->rank_base[row] = preceding;
    s->selected_tile[row] = -1;
    base = preceding;
    cut = naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row) *
          mass;
  }
  __syncthreads();
  if (t < tiles) {
    double prefix = base + s->chunk_prefix[row * tiles + t];
    double total = s->sums[row * tiles + t];
    if (prefix <= cut && prefix + total > cut) s->selected_tile[row] = t;
  }
}

__global__ void choose_token(Args a, Scratch* s) {
  choose_tile(a, s);
  __syncthreads();
  using Scan = cub::BlockScan<double, threads>;
  using R = cub::BlockReduce<int, threads>;
  __shared__ union {
    Scan::TempStorage scan;
    R::TempStorage r;
  } temp;
  const int row = blockIdx.x, t = threadIdx.x, part = s->selected_tile[row];
  const bool reject = s->reject[row];
  double values[items], total = 0;
  // Contiguous intervals preserve vocabulary order, as in the old sampler.
  for (int j = 0; j < items; ++j) {
    int i = part * tile + t * items + j;
    float v = part >= 0 && i < vocabulary ? a.p[row * vocabulary + i] : 0.0f;
    if (reject && part >= 0 && i < vocabulary)
      v = fmaxf(v - a.q[row * vocabulary + i], 0.0f);
    values[j] = v;
    total += v;
  }
  double prefix;
  Scan(temp.scan).ExclusiveSum(total, prefix);
  __syncthreads();
  prefix += s->rank_base[row] +
            (part >= 0 ? s->chunk_prefix[row * tiles + part] : 0.0);
  const double cut =
      naivert_dspark_ops::sampling_uniform(a.rng, a.anchor ? 3 : 2, row) *
      s->mass[row];
  int winner = vocabulary;
  for (int j = 0; j < items; ++j) {
    prefix += values[j];
    if (values[j] > 0 && prefix > cut) {
      winner = part * tile + t * items + j;
      break;
    }
  }
  winner = R(temp.r).Reduce(winner, Minimum());
  if (t == 0) {
    winner =
        winner < vocabulary ? a.rank * vocabulary + winner : full_vocabulary;
    if (!a.anchor && row < 7 && !reject) winner = int(a.proposals[row]);
    if (!isfinite(s->kept[row]) || s->kept[row] <= 0 ||
        !isfinite(s->mass[row]) || s->mass[row] <= 0) {
      atomicCAS(a.watchdog, 0u, 0xD6FBu);
      winner = 0;
    }
    a.peers[a.rank]->winner[row] = winner;
  }
}

__global__ void broadcast_tokens(Args a) {
  int row = threadIdx.x;
  if (row < (a.anchor ? 1 : 8)) {
    int token = full_vocabulary;
    for (int r = 0; r < 8; ++r) token = min(token, a.peers[r]->winner[row]);
    if (token >= full_vocabulary) {
      atomicCAS(a.watchdog, 0u, 0xD6FBu);
      token = 0;
    }
    a.tokens[row] = token;
  }
}

// Mailboxes/flags are zero-initialized once. All scratch fields used by compute
// are overwritten each invocation. Every rank must use the same anchor mode.
inline void run(Args a, Scratch* s, bool defer_consumed = false,
                cudaStream_t stream = nullptr) {
  int rows = a.anchor ? 1 : 8, distributions = a.anchor ? 1 : 15;
  rendezvous<<<1, 32, 0, stream>>>(a, -1);
  partial_maximum<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  local_maximum<<<distributions, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 0);
  global_maximum<<<1, 32, 0, stream>>>(a, s);
  exponentiate<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  local_sum<<<distributions, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 1);
  global_inverse<<<1, 32, 0, stream>>>(a, s);
  normalize<<<dim3(tiles, distributions), threads, 0, stream>>>(a, s);
  radix_histogram<24><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  merge_histogram<24><<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 2);
  select_digit<24><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<16><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  merge_histogram<16><<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 3);
  select_digit<16><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<8><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  merge_histogram<8><<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 4);
  select_digit<8><<<rows, threads, 0, stream>>>(a, s);
  radix_histogram<0><<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  merge_histogram<0><<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 5);
  select_digit<0><<<rows, threads, 0, stream>>>(a, s);
  normalize_nucleus<<<dim3(tiles, rows), threads, 0, stream>>>(a, s);
  local_cdf<<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 6);
  choose_token<<<rows, threads, 0, stream>>>(a, s);
  rendezvous<<<1, 32, 0, stream>>>(a, 7);
  broadcast_tokens<<<1, 32, 0, stream>>>(a);
  if (!defer_consumed) rendezvous<<<1, 32, 0, stream>>>(a, -1, true);
}
}  // namespace naivert_dspark_sharded
