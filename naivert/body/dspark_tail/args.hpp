#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include "naivert/body/dspark_tail/commit_args.hpp"
namespace naivert_dspark_tail {
inline constexpr int kRows=8, kHidden=4096, kThreads=256;
#define NAIVERT_DSPARK_GEOMETRY_FIELDS(F) \
  F(ctx_rows, const std::int32_t, Int, 1) \
  F(cache_len, const std::int32_t, Int, 1) \
  F(ctx_rows_l, std::int64_t, Long, 1) \
  F(cache_len_l, std::int64_t, Long, 1) \
  F(start_l, std::int64_t, Long, 1) \
  F(row_pos, const std::int64_t, Long, kRows) \
  F(ctx_valid_b, bool, Bool, kRows) \
  F(ctx_valid, __nv_bfloat16, BFloat16, kRows) \
  F(kv_dst, std::int64_t, Long, kRows) \
  F(trash_row, const std::int64_t, Long, kRows) \
  F(ctx_pos, std::int64_t, Long, kRows) \
  F(blk_pos, std::int64_t, Long, kRows) \
  F(kv_pos, const std::int64_t, Long, max_kv) \
  F(key_valid, bool, Bool, max_kv) \
  F(key_add, float, Float, max_kv) \
  F(target_hidden, __nv_bfloat16, BFloat16, kRows * kHidden)

#define NAIVERT_DSPARK_ACCEPT_FIELDS(F) \
  F(draft_tokens, const std::int64_t, Long, kRows - 1) \
  F(target_argmax, std::int64_t, Long, kRows) \
  F(match, bool, Bool, kRows - 1) \
  F(matchf, float, Float, kRows - 1) \
  F(cumf, float, Float, kRows - 1) \
  F(acc_f, float, Float, 1) \
  F(accepted_count, std::int32_t, Int, 1) \
  F(accepted_tokens, std::int64_t, Long, kRows) \
  F(acc_idx, std::int64_t, Long, 1) \
  F(anchor_token, std::int64_t, Long, 1) \
  F(block_ids, std::int64_t, Long, kRows) \
  F(resid0, __nv_bfloat16, BFloat16, kRows * kHidden) \
  F(cache_len, std::int32_t, Int, 1) \
  F(ctx_rows, std::int32_t, Int, 1) \
  F(embed, const __nv_bfloat16, BFloat16, -1) \
  F(mask_embedding, const __nv_bfloat16, BFloat16, kHidden)

#define FIELD(name, type, dtype, count) type* name;
struct GeometryArgs {
  NAIVERT_DSPARK_GEOMETRY_FIELDS(FIELD)
  int max_seq_len;
  int max_kv;
};
struct AcceptArgs {
  NAIVERT_DSPARK_ACCEPT_FIELDS(FIELD)
  std::int64_t mask_token;
  std::int64_t* sampling_state;
  std::int64_t* verify_counts;
};
struct EpilogueArgs {
  GeometryArgs g;
  AcceptArgs a;
  naivert_seq1_commit::NaiveN0p5Seq1CommitArgs commit;
  const __nv_bfloat16* context;
  std::int32_t* pos;
  std::int32_t* generation;
  std::int64_t* epoch;
  unsigned* arrivals;
  unsigned* watchdog;
  unsigned long long* flags[8];
  int rank;
  const __nv_bfloat16 *staged_k, *staged_v;
  __nv_bfloat16 *cache_k, *cache_v;
  int cache_rows;
};
#undef FIELD

static_assert(sizeof(GeometryArgs) == 136);
static_assert(sizeof(AcceptArgs) == 152);
static_assert(sizeof(EpilogueArgs) == 536);
static_assert(std::is_trivially_copyable_v<EpilogueArgs>);
}  // namespace naivert_dspark_tail
