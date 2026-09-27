// Class-scope fragment of the attn_dsa body (after index_topk_stage.cuh): the index-logit sweep
// on tcgen05 (Naive N only).
//
// Keys: args.ki, e4m3 [pos][128 B] in the UMMA K-major SW128 image (chunk ^ (pos & 7)) with
// args.ki_scale[pos]; queries: the same image with column = (row, head) (rotate_index_queries),
// their scales folded into index_weights_scaled. Per 128-position tile: one 16 KB bulk copy into a
// 3-stage ring; warp 0 issues tcgen05.mma.kind::f8f6f4 (M128 x N x K32, four K steps) per
// pass of up to 256 query columns into one of up to two TMEM accumulators and commits to an
// mbarrier; warps 4..11 (two groups of four, one per accumulator, each warp its 32-lane quadrant)
// read the accumulator with tcgen05.ld, apply relu x w', sum the index heads of each row, multiply by
// ki_scale[pos] and store logits[row][pos] for pos < visible(row). Same math as the mma.sync sweep.
  __device__ __forceinline__ static unsigned tc_smem_u32(const void* p) {
    return cute::cast_smem_ptr_to_uint(p);
  }
  __device__ __forceinline__ static void tc_mbar_init(unsigned long long* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tc_smem_u32(bar)), "r"(count));
  }
  __device__ __forceinline__ static void tc_mbar_expect_tx(unsigned long long* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(tc_smem_u32(bar)),
                 "r"(bytes)
                 : "memory");
  }
  __device__ __forceinline__ static void tc_mbar_arrive(unsigned long long* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tc_smem_u32(bar)) : "memory");
  }
  __device__ __forceinline__ static void tc_mbar_wait(unsigned long long* bar, unsigned phase) {
    const unsigned addr = tc_smem_u32(bar);
    unsigned done = 0;
    do {
      asm volatile(
          "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
          "selp.u32 %0, 1, 0, p;\n}"
          : "=r"(done)
          : "r"(addr), "r"(phase)
          : "memory");
    } while (!done);
  }
  __device__ __forceinline__ static void tc_bulk_g2s(void* smem, const void* gmem, unsigned bytes,
                                                     unsigned long long* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
            tc_smem_u32(smem)),
        "l"(gmem), "r"(bytes), "r"(tc_smem_u32(bar))
        : "memory");
  }
  __device__ __forceinline__ static void tc_umma_commit(unsigned long long* bar) {  // one thread
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                     tc_smem_u32(bar))
                 : "memory");
  }
  __device__ __forceinline__ static void tc_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }
  __device__ __forceinline__ static void tc_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  }
  __device__ __forceinline__ static void tc_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  }
  __device__ static void compute_index_logits_tcgen05(const Args& args,
                                                      const unsigned char* iq_image,
                                                      const float* index_weights_scaled,
                                                      float* logits, int base_pos,
                                                      SweepStorage& sw) {
    using S = SweepStorage;
    using E4M3 = cutlass::float_e4m3_t;
    constexpr float kIndexScale = 0.08838834764831845f * kIndexWeightScale;
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int max_visible = base_pos + Scratch::kQueryRows;
    const int tiles = (max_visible + S::kTileM - 1) / S::kTileM;
    const unsigned char* ki_image = reinterpret_cast<const unsigned char*>(args.ki);
    if (tid == 0) {
      for (int s = 0; s < S::kStages; ++s) {
        tc_mbar_init(sw.full + s, 1u);
        tc_mbar_init(sw.empty + s, 1u);
      }
      for (int p = 0; p < S::kPasses; ++p) {
        tc_mbar_init(sw.acc_full + p, 1u);
        tc_mbar_init(sw.acc_empty + p, 4u);
      }
      tc_mbar_init(&sw.b_full, 1u);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    for (int i = tid; i < S::kPaddedCols; i += kThreads) {
      sw.weights[i] = i < S::kCols ? index_weights_scaled[i] : 0.0f;
    }
    for (int i = S::kCols * Scratch::kIndexDim + tid;
         i < S::kPaddedCols * Scratch::kIndexDim; i += kThreads) {
      sw.b_image[i] = 0;
    }
    // Publish ordinary shared stores to the padded B columns before UMMA.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.allocate(S::kTmemCols, &sw.tmem_slot);
      allocator.release_allocation_lock();
    }
    __syncthreads();
    const unsigned tmem_base = sw.tmem_slot;
    using AtomK = cute::UMMA::Layout_K_SW128_Atom<E4M3>;  // (8, 128) e4m3, 128-byte swizzle
    auto a_layout = cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kTileM>, cute::Int<Scratch::kIndexDim>>{});
    auto b_layout = cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kPassN>, cute::Int<Scratch::kIndexDim>>{});
    const cute::UMMA::InstrDescriptor idesc =
        cute::UMMA::make_instr_desc<E4M3, E4M3, float, S::kTileM, S::kPassN, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>();
    const unsigned long long idescE = static_cast<unsigned long long>(static_cast<unsigned>(idesc)) << 32;
    using Mma = cute::SM100_MMA_F8F6F4_SS<E4M3, E4M3, float, S::kTileM, S::kPassN, cute::UMMA::Major::K,
                                          cute::UMMA::Major::K>;
    if (warp == 1) {
      // Producer: the query image once, then the A tile ring (CTA-strided tiles).
      if (lane == 0) {
        constexpr unsigned kBBytes = unsigned(S::kCols * Scratch::kIndexDim);
        tc_mbar_expect_tx(&sw.b_full, kBBytes);
        for (unsigned off = 0; off < kBBytes; off += unsigned(S::kTileBytes)) {
          const unsigned bytes = kBBytes - off < unsigned(S::kTileBytes) ? kBBytes - off : unsigned(S::kTileBytes);
          tc_bulk_g2s(sw.b_image + off, iq_image + off, bytes, &sw.b_full);
        }
        int i = 0;
        for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x), ++i) {
          const int s = i % S::kStages;
          tc_mbar_wait(sw.empty + s, unsigned(((i / S::kStages) & 1) ^ 1));
          tc_mbar_expect_tx(sw.full + s, unsigned(S::kTileBytes));
          tc_bulk_g2s(sw.a_ring + s * S::kTileBytes, ki_image + std::size_t(tile) * S::kTileBytes,
                      unsigned(S::kTileBytes), sw.full + s);
        }
      }
    } else if (warp == 0) {
      // MMA issuer.
      tc_mbar_wait(&sw.b_full, 0u);
      tc_fence_after();
      int i = 0;
      for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x), ++i) {
        const int s = i % S::kStages;
        tc_mbar_wait(sw.full + s, unsigned((i / S::kStages) & 1));
        tc_fence_after();
        auto tA = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<E4M3*>(sw.a_ring + s * S::kTileBytes)),
                                    a_layout);
        const cute::UMMA::SmemDescriptor da = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tA);
        for (int p = 0; p < S::kPasses; ++p) {
          tc_mbar_wait(sw.acc_empty + p, unsigned((i & 1) ^ 1));
          tc_fence_after();
          auto tB = cute::make_tensor(
              cute::make_smem_ptr(reinterpret_cast<E4M3*>(sw.b_image + p * S::kPassN * Scratch::kIndexDim)),
              b_layout);
          const cute::UMMA::SmemDescriptor db = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tB);
          const unsigned tmem_c = tmem_base + unsigned(p * S::kPassN);
#pragma unroll
          for (int k = 0; k < Scratch::kIndexDim / 32; ++k) {
            cute::UMMA::SmemDescriptor dak = da;
            cute::UMMA::SmemDescriptor dbk = db;
            dak.start_address_ = static_cast<unsigned short>(da.start_address_ + 2 * k);  // +32 B per K step
            dbk.start_address_ = static_cast<unsigned short>(db.start_address_ + 2 * k);
            Mma::fma(static_cast<unsigned long long>(dak), static_cast<unsigned long long>(dbk), tmem_c,
                     unsigned(k > 0), idescE);
          }
          if (lane == 0) {
            tc_umma_commit(sw.acc_full + p);
          }
        }
        if (lane == 0) {
          tc_umma_commit(sw.empty + s);  // the A stage is free once every MMA of this tile completed
        }
      }
    } else if (warp >= 4) {
      // Epilogue: group p = (warp - 4) / 4 owns accumulator p; quadrant q = warp & 3 owns TMEM lanes 32q..
      const int p = (warp - 4) >> 2;
      const int q = warp & 3;
      if (p < S::kPasses) {
        int i = 0;
        for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x), ++i) {
          tc_mbar_wait(sw.acc_full + p, unsigned(i & 1));
          tc_fence_after();
          const unsigned addr = tmem_base + (unsigned(32 * q) << 16) + unsigned(p * S::kPassN);
          constexpr int kRowsPerPass = S::kPassN / Scratch::kIndexHeads;
          float row_sum[kRowsPerPass];
#pragma unroll
          for (int r = 0; r < kRowsPerPass; ++r) {
            row_sum[r] = 0.0f;
          }
#pragma unroll
          for (int chunk = 0; chunk < S::kPassN / 32; ++chunk) {
            unsigned v[32];
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(addr + unsigned(chunk * 32), v[0], v[1], v[2], v[3], v[4],
                                                   v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13],
                                                   v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21],
                                                   v[22], v[23], v[24], v[25], v[26], v[27], v[28], v[29],
                                                   v[30], v[31]);
            tc_wait_ld();
            const float* w = sw.weights + p * S::kPassN + chunk * 32;
            constexpr int kGroup = Scratch::kIndexHeads < 32 ? Scratch::kIndexHeads : 32;
#pragma unroll
            for (int first = 0; first < 32; first += kGroup) {
              const int r = (chunk * 32 + first) / Scratch::kIndexHeads;
              float acc = 0.0f;
#pragma unroll
              for (int j = first; j < first + kGroup; ++j) {
                acc = fmaf(fmaxf(__uint_as_float(v[j]) * kIndexScale, 0.0f), w[j], acc);
              }
              row_sum[r] += acc;
            }
          }
          tc_fence_before();
          __syncwarp();
          if (lane == 0) {
            tc_mbar_arrive(sw.acc_empty + p);
          }
          const int pos = tile * S::kTileM + 32 * q + lane;
          const float scale = pos < args.max_seq_len ? args.ki_scale[pos] : 0.0f;
#pragma unroll
          for (int r = 0; r < kRowsPerPass; ++r) {
            const int query = p * kRowsPerPass + r;
            if (query < Scratch::kQueryRows && pos < base_pos + query + 1) {
              logits[std::size_t(query) * Scratch::kMaxSeqLen + pos] = row_sum[r] * scale;
            }
          }
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      cute::TMEM::Allocator1Sm allocator;
      allocator.free(tmem_base, S::kTmemCols);
    }
  }
