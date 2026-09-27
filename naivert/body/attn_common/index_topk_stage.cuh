// Class-scope fragment shared by the sparse-attention bodies (attn_dsa for Naive N0.5). It is #included inside the body
// struct, after these names exist:
//   using Args = ...;  using Scratch = ...;  static constexpr int kWmmaWarps;
//   struct ProjectionTile; using TopKBlockRadixSort; kTopKSortItemsPerThread;
//   kTopKWindowItemsPerLane; kFusedReadyShards; Scratch::{kQueryRows, kIndexHeads,
//   kIndexDim, kIndexTopK, kMaxSeqLen, kRadixControlBytes}; args.ki [max_seq_len, kIndexDim]
//   bf16 index-key cache; args.max_seq_len. Defines RadixControl.
  __device__ static std::size_t packed_iq_index(
      int query, int head, int dimension) {
    int head_block = head >> 3;
    int head_lane = head & 7;
    int k_tile = dimension >> 4;
    int k_pair = (dimension & 15) >> 1;
    int pair_lane = dimension & 1;
    return (((((std::size_t(query) * 8 + head_block) * 8 +
                k_tile) *
                   8 +
               k_pair) *
                  8 +
              head_lane) *
                 2 +
             pair_lane);
  }

  __device__ __forceinline__ static float index_weighted_pair(
      float first, float second, const float* weights,
      int head) {
    constexpr float kIndexScale =
        0.011048543456039806f;
    return fmaf(
        fmaxf(first * kIndexScale, 0.0f),
        weights[head],
        fmaxf(second * kIndexScale, 0.0f) *
            weights[head + 1]);
  }

  __device__ __forceinline__ static void
  compute_index_logits_packed_body(
      const Args& args, const __nv_bfloat16* iq,
      const float* index_weights, float* logits, int base_pos) {
    int max_visible = base_pos + Scratch::kQueryRows;
    constexpr int kPositionTile = 16;
    int position_tiles =
        (max_visible + kPositionTile - 1) / kPositionTile;
    int tid = int(threadIdx.x);
    int warp = tid >> 5;
    int lane = tid & 31;
    int lane_group = lane >> 2;
    int lane_quad = lane & 3;
    int total_tasks =
        Scratch::kQueryRows * position_tiles;
    for (int task =
             int(blockIdx.x) + warp * int(gridDim.x);
         task < total_tasks;
         task += int(gridDim.x) * kWmmaWarps) {
        int query = task / position_tiles;
        int position_tile =
            task - query * position_tiles;
        int position0 = position_tile * kPositionTile;
        int position_row0 =
            position0 + lane_group;
        int position_row8 = position_row0 + 8;
        bool position_row0_valid =
            position_row0 < args.max_seq_len;
        bool position_row8_valid =
            position_row8 < args.max_seq_len;
        int safe_position_row0 =
            position_row0_valid
                ? position_row0
                : args.max_seq_len - 1;
        int safe_position_row8 =
            position_row8_valid
                ? position_row8
                : args.max_seq_len - 1;
        float row0 = 0.0f;
        float row8 = 0.0f;
#pragma unroll
        for (int head_tile = 0; head_tile < 4;
             ++head_tile) {
          int output_low =
              head_tile * 16 + lane_quad * 2;
          int output_high = output_low + 8;
          float low0 = 0.0f;
          float low1 = 0.0f;
          float low2 = 0.0f;
          float low3 = 0.0f;
          float high0 = 0.0f;
          float high1 = 0.0f;
          float high2 = 0.0f;
          float high3 = 0.0f;
#pragma unroll
          for (int k_tile = 0;
               k_tile < Scratch::kIndexDim / 16;
               ++k_tile) {
              int k_lane =
                  k_tile * 16 + lane_quad * 2;
              unsigned a0 =
                  *reinterpret_cast<const unsigned*>(
                      args.ki +
                      std::size_t(safe_position_row0) *
                          Scratch::kIndexDim +
                      k_lane);
              unsigned a1 =
                  *reinterpret_cast<const unsigned*>(
                      args.ki +
                      std::size_t(safe_position_row8) *
                          Scratch::kIndexDim +
                      k_lane);
              unsigned a2 =
                  *reinterpret_cast<const unsigned*>(
                      args.ki +
                      std::size_t(safe_position_row0) *
                          Scratch::kIndexDim +
                      k_lane + 8);
              unsigned a3 =
                  *reinterpret_cast<const unsigned*>(
                      args.ki +
                      std::size_t(safe_position_row8) *
                          Scratch::kIndexDim +
                      k_lane + 8);
              if (!position_row0_valid) {
                a0 = 0u;
                a2 = 0u;
              }
              if (!position_row8_valid) {
                a1 = 0u;
                a3 = 0u;
              }
              int head_low =
                  head_tile * 16 + lane_group;
              int head_high = head_low + 8;
              unsigned b0 =
                  *reinterpret_cast<const unsigned*>(
                      iq + packed_iq_index(
                               query, head_low, k_lane));
              unsigned b1 =
                  *reinterpret_cast<const unsigned*>(
                      iq + packed_iq_index(
                               query, head_low,
                               k_lane + 8));
              mma_m16n8k16_bf16(
                  low0, low1, low2, low3,
                  a0, a1, a2, a3, b0, b1);
              b0 = *reinterpret_cast<const unsigned*>(
                  iq + packed_iq_index(
                           query, head_high, k_lane));
              b1 = *reinterpret_cast<const unsigned*>(
                  iq + packed_iq_index(
                           query, head_high, k_lane + 8));
              mma_m16n8k16_bf16(
                  high0, high1, high2, high3,
                  a0, a1, a2, a3, b0, b1);
          }
          const float* weights =
              index_weights +
              query * Scratch::kIndexHeads;
          float p0 = index_weighted_pair(
              low0, low1, weights, output_low);
          float p8 = index_weighted_pair(
              low2, low3, weights, output_low);
          float p1 = index_weighted_pair(
              high0, high1, weights, output_high);
          float p9 = index_weighted_pair(
              high2, high3, weights, output_high);
#pragma unroll
          for (int offset = 2; offset > 0;
               offset >>= 1) {
            p0 += __shfl_xor_sync(
                0xffffffffu, p0, offset);
            p8 += __shfl_xor_sync(
                0xffffffffu, p8, offset);
            p1 += __shfl_xor_sync(
                0xffffffffu, p1, offset);
            p9 += __shfl_xor_sync(
                0xffffffffu, p9, offset);
          }
          if (lane_quad == 0) {
            row0 += p0 + p1;
            row8 += p8 + p9;
          }
        }
        if (lane_quad == 0) {
          int visible = base_pos + query + 1;
          if (position_row0 < visible) {
            logits[
                std::size_t(query) *
                        Scratch::kMaxSeqLen +
                    position_row0] = row0;
          }
          if (position_row8 < visible) {
            logits[
                std::size_t(query) *
                        Scratch::kMaxSeqLen +
                    position_row8] = row8;
          }
        }
    }
  }

  __device__ __noinline__ static void
  compute_index_logits_packed_medium(
      const Args& args, const __nv_bfloat16* iq,
      const float* index_weights, float* logits, int base_pos) {
    compute_index_logits_packed_body(
        args, iq, index_weights, logits, base_pos);
  }

  __device__ static void compute_index_logits(
      const Args& args, const __nv_bfloat16* iq,
      const float* index_weights, float* logits, int base_pos) {
    int max_visible = base_pos + Scratch::kQueryRows;
    if (max_visible > 4096) {
      compute_index_logits_packed_body(
          args, iq, index_weights, logits, base_pos);
      return;
    }
    compute_index_logits_packed_medium(
        args, iq, index_weights, logits, base_pos);
  }

  __device__ static unsigned ordered_float_key(float value) {
    unsigned bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0u ? ~bits
                                      : (bits ^ 0x80000000u);
  }

  struct RadixControl {
    unsigned prefix[Scratch::kQueryRows];
    unsigned remaining[Scratch::kQueryRows];
    unsigned threshold[Scratch::kQueryRows];
    unsigned fused_ready[kFusedReadyShards];
    unsigned qnorm_ready;
  };
  static_assert(
      sizeof(RadixControl) == Scratch::kRadixControlBytes,
      "rows radix workspace layout must match its control object");

  __device__ __forceinline__ static unsigned long long
  warp_min_u64(unsigned long long value, int lane) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      unsigned long long other =
          __shfl_down_sync(0xffffffffu, value, offset);
      if (lane + offset < 32) {
        value = value < other ? value : other;
      }
    }
    return __shfl_sync(0xffffffffu, value, 0);
  }

  __device__ __noinline__ static void select_near_topk(
      const float* logits, int* selections, int query,
      int visible, float* shared_storage) {
    int tid = int(threadIdx.x);
    int warp = tid >> 5;
    int lane = tid & 31;
    constexpr unsigned long long kInvalid = ~0ull;
    constexpr int kTokensPerThread = 6;
    unsigned long long local_candidates[kTokensPerThread];
#pragma unroll
    for (int item = 0; item < kTokensPerThread; ++item) {
      int token = tid + item * int(blockDim.x);
      if (token < visible) {
        unsigned key = ordered_float_key(
            logits[
                std::size_t(query) *
                    Scratch::kMaxSeqLen +
                token]);
        local_candidates[item] =
            (static_cast<unsigned long long>(key) << 32) |
            static_cast<unsigned>(~unsigned(token));
      } else {
        local_candidates[item] = kInvalid;
      }
    }

    auto* warp_bottom =
        reinterpret_cast<unsigned long long*>(shared_storage);
#pragma unroll
    for (int rank = 0; rank < 4; ++rank) {
      unsigned long long local_min = kInvalid;
#pragma unroll
      for (int item = 0; item < kTokensPerThread; ++item) {
        local_min =
            local_min < local_candidates[item]
                ? local_min
                : local_candidates[item];
      }
      unsigned long long selected =
          warp_min_u64(local_min, lane);
      if (local_min == selected) {
#pragma unroll
        for (int item = 0; item < kTokensPerThread; ++item) {
          if (local_candidates[item] == selected) {
            local_candidates[item] = kInvalid;
          }
        }
      }
      if (lane == 0) {
        warp_bottom[warp * 4 + rank] = selected;
      }
    }
    __syncthreads();

    if (warp == 0) {
      unsigned long long candidate0 =
          lane < 32 ? warp_bottom[lane] : kInvalid;
      unsigned long long candidate1 =
          lane + 32 < kWmmaWarps * 4
              ? warp_bottom[lane + 32]
              : kInvalid;
#pragma unroll
      for (int rank = 0; rank < 4; ++rank) {
        unsigned long long local_min =
            candidate0 < candidate1
                ? candidate0
                : candidate1;
        unsigned long long selected =
            warp_min_u64(local_min, lane);
        if (candidate0 == selected) {
          candidate0 = kInvalid;
        }
        if (candidate1 == selected) {
          candidate1 = kInvalid;
        }
        if (lane == 0) {
          warp_bottom[rank] = selected;
        }
      }
      if (lane == 0) {
        int exclusion_count =
            visible - Scratch::kIndexTopK;
        if (exclusion_count < 0) {
          exclusion_count = 0;
        }
        int excluded0 =
            exclusion_count > 0
                ? int(~static_cast<unsigned>(warp_bottom[0]))
                : visible;
        int excluded1 =
            exclusion_count > 1
                ? int(~static_cast<unsigned>(warp_bottom[1]))
                : visible;
        int excluded2 =
            exclusion_count > 2
                ? int(~static_cast<unsigned>(warp_bottom[2]))
                : visible;
        int excluded3 =
            exclusion_count > 3
                ? int(~static_cast<unsigned>(warp_bottom[3]))
                : visible;
#define MLA_ROWS_COMPARE_SWAP(a, b) \
        if ((a) > (b)) {            \
          int temporary = (a);      \
          (a) = (b);                \
          (b) = temporary;          \
        }
        MLA_ROWS_COMPARE_SWAP(excluded0, excluded1);
        MLA_ROWS_COMPARE_SWAP(excluded2, excluded3);
        MLA_ROWS_COMPARE_SWAP(excluded0, excluded2);
        MLA_ROWS_COMPARE_SWAP(excluded1, excluded3);
        MLA_ROWS_COMPARE_SWAP(excluded1, excluded2);
#undef MLA_ROWS_COMPARE_SWAP
        auto* excluded =
            reinterpret_cast<int*>(warp_bottom);
        excluded[0] = excluded0;
        excluded[1] = excluded1;
        excluded[2] = excluded2;
        excluded[3] = excluded3;
      }
    }
    __syncthreads();

    int exclusion_count =
        visible - Scratch::kIndexTopK;
    if (exclusion_count < 0) {
      exclusion_count = 0;
    }
    int count = visible < Scratch::kIndexTopK
                    ? visible
                    : Scratch::kIndexTopK;
    auto* excluded_tokens =
        reinterpret_cast<const int*>(warp_bottom);
    int vector_count = (count + 3) / 4;
    for (int vector = tid; vector < vector_count;
         vector += int(blockDim.x)) {
      int ordinal0 = vector * 4;
      int mapped[4];
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        int token = ordinal0 + item;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
          if (slot < exclusion_count &&
              excluded_tokens[slot] <= token) {
            ++token;
          }
        }
        mapped[item] = token;
      }
      int* destination =
          selections +
          query * Scratch::kIndexTopK + ordinal0;
      if (ordinal0 + 3 < count) {
        *reinterpret_cast<int4*>(destination) =
            make_int4(
                mapped[0], mapped[1],
                mapped[2], mapped[3]);
      } else {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          if (ordinal0 + item < count) {
            destination[item] = mapped[item];
          }
        }
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ static void
  materialize_general_topk(
      const float* logits, int* selections, RadixControl* control,
      int base_pos, float* shared_storage) {
    auto* local_histogram =
        reinterpret_cast<unsigned*>(shared_storage);
    int tid = int(threadIdx.x);
    int selection_query =
        int(blockIdx.x) % Scratch::kQueryRows;
    int visible = base_pos + selection_query + 1;
    constexpr int kCount = Scratch::kIndexTopK;
    unsigned threshold =
        control->threshold[selection_query];
    bool unique_threshold =
        control->prefix[selection_query] == 1u;
    if (unique_threshold) {
      int lane = tid & 31;
      int warp = tid >> 5;
      int worker =
          int(blockIdx.x) / Scratch::kQueryRows;
      int workers =
          int(gridDim.x) / Scratch::kQueryRows;
      for (int chunk =
               worker * int(blockDim.x);
           chunk < visible;
           chunk += workers * int(blockDim.x)) {
        int token = chunk + tid;
        bool selected = false;
        if (token < visible) {
          unsigned key = ordered_float_key(
              logits[
                  std::size_t(selection_query) *
                          Scratch::kMaxSeqLen +
                      token]);
          selected = key >= threshold;
        }
        unsigned selected_mask =
            __ballot_sync(0xffffffffu, selected);
        unsigned lower_lanes =
            lane == 0
                ? 0u
                : (1u << unsigned(lane)) - 1u;
        unsigned lane_rank =
            unsigned(
                __popc(selected_mask & lower_lanes));
        unsigned warp_count =
            unsigned(__popc(selected_mask));
        if (lane == 0) {
          local_histogram[warp] = warp_count;
        }
        __syncthreads();
        if (warp == 0) {
          unsigned value =
              lane < kWmmaWarps
                  ? local_histogram[lane]
                  : 0u;
          unsigned inclusive = value;
#pragma unroll
          for (int offset = 1; offset < 16;
               offset <<= 1) {
            unsigned previous =
                __shfl_up_sync(
                    0xffffffffu, inclusive, offset);
            if (lane >= offset) {
              inclusive += previous;
            }
          }
          if (lane < kWmmaWarps) {
            local_histogram[lane] =
                inclusive - value;
          }
          if (lane == kWmmaWarps - 1) {
            local_histogram[kWmmaWarps] =
                atomicAdd(
                    control->remaining +
                        selection_query,
                    inclusive);
          }
        }
        __syncthreads();
        unsigned output_index =
            local_histogram[kWmmaWarps] +
            local_histogram[warp] + lane_rank;
        if (selected &&
            output_index < unsigned(kCount)) {
          selections[
              selection_query * Scratch::kIndexTopK +
              output_index] = token;
        }
      }
      return;
    }
    if (int(blockIdx.x) >= Scratch::kQueryRows) {
      return;
    }

    int query = int(blockIdx.x);
    int query_visible = base_pos + query + 1;
    int lane = tid & 31;
    if (tid == 0) {
      local_histogram[255] = 0u;
    }
    __syncthreads();
    for (int chunk = 0; chunk < query_visible;
         chunk += int(blockDim.x)) {
      int token = chunk + tid;
      bool selected = false;
      if (token < query_visible) {
        unsigned key = ordered_float_key(
            logits[
                std::size_t(query) *
                        Scratch::kMaxSeqLen +
                    token]);
        selected = key > threshold;
      }
      unsigned selected_mask =
          __ballot_sync(0xffffffffu, selected);
      unsigned lower_lanes =
          lane == 0
              ? 0u
              : (1u << unsigned(lane)) - 1u;
      unsigned lane_rank =
          unsigned(
              __popc(selected_mask & lower_lanes));
      unsigned warp_count =
          unsigned(__popc(selected_mask));
      unsigned warp_base = 0u;
      if (lane == 0 && warp_count != 0u) {
        warp_base = atomicAdd(
            local_histogram + 255, warp_count);
      }
      warp_base = __shfl_sync(
          0xffffffffu, warp_base, 0);
      unsigned output_index =
          warp_base + lane_rank;
      if (selected &&
          output_index < unsigned(kCount)) {
        selections[
            query * Scratch::kIndexTopK +
            output_index] = token;
      }
    }
    __syncthreads();

    // An exact radix threshold may be shared by many tokens.  All values
    // strictly above it are already present, so a single deterministic scan
    // fills the remaining slots with the lowest token indices.  This branch
    // is cold for the model-derived acceptance inputs and preserves the
    // stable value-descending/index-ascending tie contract.
    if (tid == 0) {
      unsigned output_index = local_histogram[255];
      for (int token = 0;
           token < query_visible &&
           output_index < unsigned(kCount);
           ++token) {
        unsigned key = ordered_float_key(
            logits[
                std::size_t(query) *
                        Scratch::kMaxSeqLen +
                    token]);
        if (key == threshold) {
          selections[
              query * Scratch::kIndexTopK +
              output_index] = token;
          ++output_index;
        }
      }
      local_histogram[255] = output_index;
    }
    __syncthreads();
  }

  // The two top-k paths compact selected positions with atomically claimed
  // slots, so their order is not repeatable. Sorting the query's selections
  // ascending makes the attention sum order (and the output bits) stable.
  using SelectionSort = cub::BlockRadixSort<unsigned, 384, 6, cub::NullType, 5>;
  static_assert(384 * 6 >= Scratch::kIndexTopK, "selection sort capacity");
  static_assert(sizeof(typename SelectionSort::TempStorage) <=
                    sizeof(typename TopKBlockRadixSort::TempStorage),
                "selection sort reuses the top-k sort storage");
  static_assert(Scratch::kMaxSeqLen <= (1 << 21),
                "position keys (the 1M Naive N build selects through the grid path, not this sort)");

  __device__ static void sort_selections(
      int* selections, int count,
      typename TopKBlockRadixSort::TempStorage* sort_storage) {
    constexpr int kItems = 6;
    int tid = int(threadIdx.x);
    unsigned keys[kItems];
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      int index = tid * kItems + item;
      keys[item] = index < count ? unsigned(selections[index]) : 0xffffffffu;
    }
    __syncthreads();
    SelectionSort(*reinterpret_cast<typename SelectionSort::TempStorage*>(sort_storage))
        .Sort(keys, 0, 18);
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      int index = tid * kItems + item;
      if (index < count) {
        selections[index] = int(keys[item]);
      }
    }
    __syncthreads();
  }

  __device__ static void select_per_query_topk(
      const float* logits, int* selections, RadixControl* control,
      int base_pos, float* shared_storage, float* attention_temp,
      typename TopKBlockRadixSort::TempStorage* sort_storage,
      unsigned* topk_scratch,
      ProjectionTile* projection_tile) {
    auto grid = cg::this_grid();
    int tid = int(threadIdx.x);
    int global_thread =
        int(blockIdx.x) * int(blockDim.x) + tid;
    int global_stride = int(gridDim.x) * int(blockDim.x);

    bool all_identity =
        base_pos + Scratch::kQueryRows <= Scratch::kIndexTopK;
    if (all_identity) {
      int max_visible = base_pos + Scratch::kQueryRows;
      for (int work = global_thread;
           work < Scratch::kQueryRows * max_visible;
           work += global_stride) {
        int query = work / max_visible;
        int token = work - query * max_visible;
        int visible = base_pos + query + 1;
        if (token < visible) {
          selections[query * Scratch::kIndexTopK + token] = token;
        }
      }
      grid.sync();
      return;
    }

    bool near_topk =
        base_pos + Scratch::kQueryRows <=
        Scratch::kIndexTopK + Scratch::kQueryRows;
    if (near_topk) {
      int query = int(blockIdx.x);
      if (query < Scratch::kQueryRows) {
        int visible = base_pos + query + 1;
        select_near_topk(
            logits, selections, query, visible,
            shared_storage);
        sort_selections(
            selections + query * Scratch::kIndexTopK,
            visible < Scratch::kIndexTopK ? visible : Scratch::kIndexTopK,
            sort_storage);
        if (tid == 0) {
          __threadfence();
          atomicExch(control->threshold + query, 1u);
        }
      }
      return;
    }

    int query = int(blockIdx.x);
    if (query < Scratch::kQueryRows) {
      int visible = base_pos + query + 1;
      unsigned keys[kTopKSortItemsPerThread];
#pragma unroll
      for (int item = 0;
           item < kTopKSortItemsPerThread; ++item) {
        int token = item * int(blockDim.x) + tid;
        keys[item] =
            token < visible
                ? ordered_float_key(
                      logits[
                          std::size_t(query) *
                                  Scratch::kMaxSeqLen +
                              token])
                : 0u;
      }
      TopKBlockRadixSort(*sort_storage)
          .SortDescending(keys, 17, 32);
      constexpr int kThresholdRank =
          Scratch::kIndexTopK - 1;
      constexpr int kThresholdThread =
          kThresholdRank / kTopKSortItemsPerThread;
      constexpr int kThresholdItem =
          kThresholdRank % kTopKSortItemsPerThread;
      constexpr int kWindowThreadBase = 89;
      constexpr int kWindowThreads = 7;
      constexpr int kWindowRankBase =
          kWindowThreadBase * kTopKSortItemsPerThread;
      constexpr int kWindowItems =
          kWindowThreads * kTopKSortItemsPerThread;
      constexpr int kWideFlag = kWindowItems;
      constexpr int kGroupStartSlot = kWideFlag + 1;
      constexpr int kGroupEndSlot = kWideFlag + 2;
      constexpr int kWindowHistogramBase =
          kGroupEndSlot + 1;
      if (tid >= kWindowThreadBase &&
          tid < kWindowThreadBase + kWindowThreads) {
        int window_thread = tid - kWindowThreadBase;
#pragma unroll
        for (int item = 0;
             item < kTopKSortItemsPerThread; ++item) {
          topk_scratch[
              window_thread *
                      kTopKSortItemsPerThread +
                  item] =
              keys[item];
        }
      }
      __syncthreads();
      if (tid < 32) {
        int lane = tid;
        constexpr unsigned kPrefixMask = 0xfffe0000u;
        unsigned threshold_prefix =
            lane == 0
                ? topk_scratch[
                      kThresholdRank -
                      kWindowRankBase] &
                      kPrefixMask
                : 0u;
        threshold_prefix = __shfl_sync(
            0xffffffffu, threshold_prefix, 0);
        constexpr int kBoundaryItemsPerLane =
            (kWindowItems + 31) / 32;
        unsigned local_first = ~0u;
        unsigned local_last = 0u;
#pragma unroll
        for (int item = 0;
             item < kBoundaryItemsPerLane; ++item) {
          int window_rank =
              lane * kBoundaryItemsPerLane + item;
          if (window_rank < kWindowItems &&
              (topk_scratch[window_rank] &
               kPrefixMask) == threshold_prefix) {
            unsigned rank =
                unsigned(
                    kWindowRankBase + window_rank);
            local_first =
                local_first < rank
                    ? local_first
                    : rank;
            local_last =
                local_last > rank
                    ? local_last
                    : rank;
          }
        }
#pragma unroll
        for (int offset = 16; offset > 0;
             offset >>= 1) {
          unsigned other_first = __shfl_down_sync(
              0xffffffffu, local_first, offset);
          unsigned other_last = __shfl_down_sync(
              0xffffffffu, local_last, offset);
          if (lane + offset < 32) {
            local_first =
                local_first < other_first
                    ? local_first
                    : other_first;
            local_last =
                local_last > other_last
                    ? local_last
                    : other_last;
          }
        }
        if (lane == 0) {
          int group_start = int(local_first);
          int group_end = int(local_last + 1u);
          bool wide_group =
              group_start == kWindowRankBase ||
              group_end ==
                  kWindowRankBase + kWindowItems ||
              group_end - group_start >
                  32 * kTopKWindowItemsPerLane;
          topk_scratch[kWideFlag] =
              wide_group ? 1u : 0u;
          if (!wide_group) {
            topk_scratch[kGroupStartSlot] =
                local_first;
            topk_scratch[kGroupEndSlot] =
                local_last + 1u;
            control->threshold[query] =
                threshold_prefix;
            control->remaining[query] =
                unsigned(
                    kThresholdRank - group_start + 1);
          }
        }
      }
      __syncthreads();
      bool wide_group =
          topk_scratch[kWideFlag] != 0u;
      if (wide_group) {
        TopKBlockRadixSort(*sort_storage)
            .SortDescending(keys);
        if (tid == kThresholdThread) {
          unsigned threshold = keys[kThresholdItem];
          bool unique =
              keys[kThresholdItem - 1] != threshold &&
              keys[kThresholdItem + 1] != threshold;
          control->threshold[query] = threshold;
          control->prefix[query] =
              unique ? 1u : 2u;
          control->remaining[query] = 0u;
        }
      } else if (tid < 32) {
        int lane = tid;
        int group_start =
            int(topk_scratch[kGroupStartSlot]);
        int group_end =
            int(topk_scratch[kGroupEndSlot]);
        int candidate_count =
            group_end - group_start;
        int candidate_base =
            group_start - kWindowRankBase;
        unsigned* histogram =
            topk_scratch + kWindowHistogramBase;
        unsigned prefix =
            control->threshold[query];
        unsigned remaining =
            control->remaining[query];
#pragma unroll 1
        for (int shift = 12; shift >= 0;
             shift -= 6) {
          for (int digit = lane; digit < 64;
               digit += 32) {
            histogram[digit] = 0u;
          }
          __syncwarp();
          unsigned higher_mask =
              shift == 12
                  ? 0xfffe0000u
                  : (shift == 6
                         ? 0xfffff000u
                         : 0xffffffc0u);
#pragma unroll
          for (int item = 0;
               item < kTopKWindowItemsPerLane;
               ++item) {
            int candidate =
                lane * kTopKWindowItemsPerLane +
                item;
            unsigned digit = ~0u;
            if (candidate < candidate_count) {
              unsigned key =
                  topk_scratch[
                      candidate_base + candidate];
              if ((key & higher_mask) == prefix) {
                digit =
                    (key >> unsigned(shift)) &
                    0x3fu;
                if (shift == 12) {
                  digit &= 0x1fu;
                }
              }
            }
            unsigned peers = __match_any_sync(
                0xffffffffu, digit);
            int leader = __ffs(peers) - 1;
            if (digit != ~0u && lane == leader) {
              atomicAdd(
                  histogram + digit,
                  unsigned(__popc(peers)));
            }
          }
          __syncwarp();
          unsigned high_count =
              histogram[63 - lane];
          unsigned high_inclusive = high_count;
#pragma unroll
          for (int offset = 1; offset < 32;
               offset <<= 1) {
            unsigned previous = __shfl_up_sync(
                0xffffffffu, high_inclusive,
                offset);
            if (lane >= offset) {
              high_inclusive += previous;
            }
          }
          unsigned high_before =
              high_inclusive - high_count;
          bool owns_high =
              remaining > high_before &&
              remaining <= high_inclusive;
          unsigned high_mask = __ballot_sync(
              0xffffffffu, owns_high);
          unsigned selected_digit = 0u;
          unsigned selected_count = 0u;
          unsigned selected_remaining = 0u;
          if (high_mask != 0u) {
            int owner = __ffs(high_mask) - 1;
            selected_digit = unsigned(63 - owner);
            selected_count = __shfl_sync(
                0xffffffffu, high_count, owner);
            unsigned owner_before = __shfl_sync(
                0xffffffffu, high_before, owner);
            selected_remaining =
                remaining - owner_before;
          } else {
            unsigned high_total = __shfl_sync(
                0xffffffffu, high_inclusive, 31);
            unsigned low_remaining =
                remaining - high_total;
            unsigned low_count =
                histogram[31 - lane];
            unsigned low_inclusive = low_count;
#pragma unroll
            for (int offset = 1; offset < 32;
                 offset <<= 1) {
              unsigned previous = __shfl_up_sync(
                  0xffffffffu, low_inclusive,
                  offset);
              if (lane >= offset) {
                low_inclusive += previous;
              }
            }
            unsigned low_before =
                low_inclusive - low_count;
            bool owns_low =
                low_remaining > low_before &&
                low_remaining <= low_inclusive;
            unsigned low_mask = __ballot_sync(
                0xffffffffu, owns_low);
            int owner = __ffs(low_mask) - 1;
            selected_digit = unsigned(31 - owner);
            selected_count = __shfl_sync(
                0xffffffffu, low_count, owner);
            unsigned owner_before = __shfl_sync(
                0xffffffffu, low_before, owner);
            selected_remaining =
                low_remaining - owner_before;
          }
          if (lane == 0) {
            prefix |= selected_digit
                      << unsigned(shift);
            control->threshold[query] = prefix;
            if (shift == 0) {
              control->prefix[query] =
                  selected_count == 1u
                      ? 1u
                      : 2u;
              control->remaining[query] = 0u;
            } else {
              control->remaining[query] =
                  selected_remaining;
            }
          }
          __syncwarp();
          prefix = control->threshold[query];
          remaining = control->remaining[query];
        }
      }
    }
    grid.sync();
    record_phase(attention_temp, 10);
    materialize_general_topk(
        logits, selections, control, base_pos, shared_storage);
    record_phase(attention_temp, 11);
    grid.sync();
    if (int(blockIdx.x) < Scratch::kQueryRows) {
      int sorted_query = int(blockIdx.x);
      int sorted_visible = base_pos + sorted_query + 1;
      sort_selections(
          selections + sorted_query * Scratch::kIndexTopK,
          sorted_visible < Scratch::kIndexTopK ? sorted_visible : Scratch::kIndexTopK,
          sort_storage);
    }
    grid.sync();
  }

  __device__ static void publish_identity_selections(
      int* selections, int base_pos) {
    int global_thread =
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    int global_stride = int(gridDim.x) * int(blockDim.x);
    int max_visible = base_pos + Scratch::kQueryRows;
    for (int work = global_thread;
         work < Scratch::kQueryRows * max_visible;
         work += global_stride) {
      int query = work / max_visible;
      int token = work - query * max_visible;
      if (token < base_pos + query + 1) {
        selections[query * Scratch::kIndexTopK + token] = token;
      }
    }
  }
