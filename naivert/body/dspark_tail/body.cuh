#pragma once
#include "naivert/body/dspark_tail/args.hpp"
#include "naivert/body/dspark_tail/commit.cuh"
#include "naivert/body/dspark_sampling/sync.cuh"
namespace naivert_dspark_tail {
// Non-final CTAs exit instead of spinning: the grid need not fit concurrently.
// Old control scalars stay immutable until every CTA has finished reading them.
struct TailBody {
  template<bool Staged>
  __device__ static bool run(const EpilogueArgs& e) {
  const auto& a = e.a;
  const auto& g = e.g;
  const int tid = threadIdx.x;
  __shared__ int prefix, anchor, rows, cache;
  if (tid < 32) {
    int token = tid < kRows ? e.commit.token_out[tid] : 0;
    bool match = tid < kRows - 1 && token == a.draft_tokens[tid];
    unsigned missing = (~__ballot_sync(0xffffffffu, match)) & 127u;
    int accepted = missing ? __ffs(missing) - 1 : 7;
    int next = __shfl_sync(0xffffffffu, token, accepted);
    if (tid == 0) {
      prefix = accepted;
      anchor = next;
      rows = accepted + 1;
      cache = std::int32_t(std::uint32_t(*a.cache_len) + std::uint32_t(*a.ctx_rows));
    }
  }
  __syncthreads();
  const int i = int(blockIdx.x) * blockDim.x + tid;
  const std::int64_t start = std::int64_t(cache) + rows;
  if (i == 0) {
    *g.ctx_rows_l = rows;
    *g.cache_len_l = cache;
    *g.start_l = start;
  }
  if (i < kRows) {
    auto row = g.row_pos[i];
    bool valid = row < rows;
    g.ctx_valid_b[i] = valid;
    g.ctx_valid[i] = __float2bfloat16_rn(valid ? 1.f : 0.f);
    auto dst = valid ? row + cache : g.trash_row[i];
    g.kv_dst[i] = dst < g.max_seq_len ? dst : g.max_seq_len;
    g.ctx_pos[i] = min(row + cache, std::int64_t(g.max_kv - 1));
    g.blk_pos[i] = min(row + start, std::int64_t(g.max_kv - 1));
  }
  // The graph joins staged D12 before this epilogue. Publish retained KV
  // before the existing arrival counter permits control commit and tail launch.
  if constexpr (Staged) {
  if (i < 5 * kRows * 128) {
    const int layer = i / (kRows * 128), row = (i / 128) % kRows, col = i % 128;
    if (row < rows) {
      const auto dst = min(std::int64_t(cache) + row, std::int64_t(g.max_seq_len));
      const auto out = (std::size_t(layer) * e.cache_rows + dst) * 128 + col;
      e.cache_k[out] = e.staged_k[i];
      e.cache_v[out] = e.staged_v[i];
    }
  }
  }
  if (i < g.max_kv) {
    bool valid = g.kv_pos[i] < start;
    g.key_valid[i] = valid;
    g.key_add[i] = valid ? 0.f : -1e30f;
  }
  if (i < kRows * kHidden) {
    float mask = g.row_pos[i / kHidden] < rows ? 1.f : 0.f;
    g.target_hidden[i] = __float2bfloat16_rn(
        __fmul_rn(__bfloat162float(e.context[i]), mask));
    a.resid0[i] = i < kHidden ? a.embed[std::int64_t(anchor) * kHidden + i]
                             : a.mask_embedding[i % kHidden];
  }
  __syncthreads();
  if (tid >= 32) return false;
  // Warp 0 of every CTA arrives; only the last CTA's warp continues. Lanes 0..7 then run the
  // pushed consumed rendezvous (eight local polls instead of eight sequential remote ones), and
  // lane 0 alone runs the serial commit. The two host cancel words are read before the
  // rendezvous so their PCIe round trip overlaps the wait; the rendezvous' system fence orders
  // them ahead of everything that follows (an acquire by fence, as before by load).
  unsigned last = 0;
  cuda::atomic_ref<unsigned, cuda::thread_scope_device> done(*e.arrivals);
  if (tid == 0) last = done.fetch_add(1u, cuda::memory_order_acq_rel) == unsigned(gridDim.x - 1);
  if (!__shfl_sync(0xffffffffu, last, 0)) return false;
  std::uint32_t cancel_at = 0u, cancel_requested = 0u;
  if (tid == 0) {
    done.store(0u, cuda::memory_order_relaxed);
    if (e.commit.cancel != nullptr) {
      cancel_at = naivert_seq1_commit::system_load_relaxed(
          &e.commit.cancel[naivert_seq1_commit::kSeq1CancelAtGenerationWord]);
      cancel_requested = naivert_seq1_commit::system_load_relaxed(
          &e.commit.cancel[naivert_seq1_commit::kSeq1CancelRequestedWord]);
    }
#if NAIVERT_DSPARK_SAMPLER_NO_READY
    // READY=0 leaves the external epoch to the deferred consumer. Advance once
    // per rank before the helper fences, publishes CONSUMED and polls peers.
    ++e.flags[e.rank][2];
#endif
  }
  __syncwarp();
  const bool met = naivert_dspark_ops::sampling_rendezvous_push(e.flags, e.rank, e.watchdog);
  if (tid != 0) return false;
  if (!met) {
    naivert_seq1_commit::fail_closed(e.commit.control, e.commit.control->flags);
    return false;
  }
  for (int j = 0; j < kRows; ++j) {
    auto token = e.commit.token_out[j];
    a.target_argmax[j] = token;
    a.accepted_tokens[j] = token;
    a.block_ids[j] = j == 0 ? anchor : a.mask_token;
    if (j < 7) {
      bool match = token == a.draft_tokens[j];
      a.match[j] = match;
      a.matchf[j] = match ? 1.f : 0.f;
      a.cumf[j] = j < prefix ? 1.f : 0.f;
    }
  }
  *a.acc_f = float(rows);
  *a.accepted_count = rows;
  *a.acc_idx = prefix;
  *a.anchor_token = anchor;
  *a.cache_len = cache;
  *a.ctx_rows = rows;
  ++a.sampling_state[1];
  a.verify_counts[0] += prefix;
  ++a.verify_counts[1];
  ++a.verify_counts[2 + prefix];
  *e.pos = std::int32_t(std::uint32_t(*e.pos) + unsigned(rows));
  *e.generation = std::int32_t(std::uint32_t(*e.generation) + 1u);
  *e.epoch = std::int64_t(std::uint64_t(*e.epoch) + 1u);
  return naivert_seq1_commit::commit_body(e.commit, cancel_at, cancel_requested);
}

};
struct GeometryBody {
__device__ static void run(const GeometryArgs& a) {
  const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t rows = *a.ctx_rows;
  const std::int64_t cache = *a.cache_len;
  const std::int64_t start = cache + rows;
  if (i == 0) {
    *a.ctx_rows_l = rows;
    *a.cache_len_l = cache;
    *a.start_l = start;
  }
  if (i < kRows) {
    const std::int64_t row = a.row_pos[i];
    const bool valid = row < rows;
    a.ctx_valid_b[i] = valid;
    a.ctx_valid[i] = __float2bfloat16_rn(valid ? 1.0f : 0.0f);
    const std::int64_t destination = valid ? row + cache : a.trash_row[i];
    a.kv_dst[i] = destination < a.max_seq_len ? destination : a.max_seq_len;
    const std::int64_t ctx = row + cache;
    const std::int64_t block = row + start;
    a.ctx_pos[i] = ctx < a.max_kv - 1 ? ctx : a.max_kv - 1;
    a.blk_pos[i] = block < a.max_kv - 1 ? block : a.max_kv - 1;
  }
  if (i < a.max_kv) {
    const bool valid = a.kv_pos[i] < start;
    a.key_valid[i] = valid;
    a.key_add[i] = valid ? 0.0f : -1e30f;
  }
  if (i < kRows * kHidden) {
    const float mask = a.row_pos[i / kHidden] < rows ? 1.0f : 0.0f;
    // Match mul_, including the sign of a masked zero.
    a.target_hidden[i] = __float2bfloat16_rn(
        __fmul_rn(__bfloat162float(a.target_hidden[i]), mask));
  }
}

};
struct AcceptBody {
__device__ static void run(const AcceptArgs& a) {
  __shared__ std::int64_t next_anchor;
  const int tid = threadIdx.x;
  if (tid < 32) {
    const long long target = tid < kRows ? a.target_argmax[tid] : 0;
    const bool match = tid < kRows - 1 && a.draft_tokens[tid] == target;
    const unsigned matches = __ballot_sync(0xffffffffu, match);
    const unsigned missing = (~matches) & ((1u << (kRows - 1)) - 1);
    const int prefix = missing ? __ffs(missing) - 1 : kRows - 1;
    const long long anchor = __shfl_sync(0xffffffffu, target, prefix);
    if (tid == 0) next_anchor = anchor;
    if (blockIdx.x == 0) {
      if (tid < kRows - 1) {
        a.match[tid] = match;
        a.matchf[tid] = match ? 1.0f : 0.0f;
        a.cumf[tid] = tid < prefix ? 1.0f : 0.0f;
      }
      if (tid < kRows) {
        a.accepted_tokens[tid] = target;
        a.block_ids[tid] = tid == 0 ? anchor : a.mask_token;
      }
      if (tid == 0) {
        const int count = prefix + 1;
        *a.acc_f = float(count);
        *a.accepted_count = count;
        *a.acc_idx = prefix;
        *a.anchor_token = anchor;
        *a.cache_len = std::int32_t(std::uint32_t(*a.cache_len) + std::uint32_t(*a.ctx_rows));
        *a.ctx_rows = count;
        if (a.sampling_state) ++a.sampling_state[1];
        if (a.verify_counts) {
          a.verify_counts[0] += prefix;
          ++a.verify_counts[1];
          ++a.verify_counts[2 + prefix];
        }
      }
    }
  }
  __syncthreads();
  const int i = int(blockIdx.x) * blockDim.x + tid;
  if (i < kRows * kHidden) {
    a.resid0[i] = i < kHidden ? a.embed[next_anchor * kHidden + i]
                             : a.mask_embedding[i % kHidden];
  }
}

};
}  // namespace naivert_dspark_tail
