// Class-scope fragment shared by the sparse-attention bodies (attn_dsa for Naive N0.5). It is #included inside the body
// struct, after these names exist:
//   using Args = ...;  using Scratch = ...;  static constexpr int kWmmaWarps;
//   Scratch::kPhaseTimerFloatOffset, Scratch::kPhaseTimerCount
  __device__ __forceinline__ static void record_phase(
      float* attention_temp, int phase) {
#if NAIVERT_MLA_ROWS_ENABLE_PHASE_TIMERS
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      unsigned long long timestamp;
      asm volatile(
          "mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
      reinterpret_cast<unsigned long long*>(
          attention_temp + kPhaseTimerFloatOffset)[phase] =
          timestamp;
    }
#else
    (void)attention_temp;
    (void)phase;
#endif
  }

  __device__ static float block_sum(float value, float* shared) {
    int tid = int(threadIdx.x);
    shared[tid] = value;
    if (tid < 128) {
      shared[384 + tid] = 0.0f;
    }
    __syncthreads();
    for (int stride = 256; stride >= 64; stride >>= 1) {
      if (tid < stride) {
        shared[tid] += shared[tid + stride];
      }
      __syncthreads();
    }
    if (tid < 32) {
      value = shared[tid] + shared[tid + 32];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffffu, value, offset);
      }
      if (tid == 0) {
        shared[0] = value;
      }
    }
    __syncthreads();
    return shared[0];
  }

  __device__ __forceinline__ static float block_sum_warp_first(
      float value, float* shared) {
    int tid = int(threadIdx.x);
    int warp = tid >> 5;
    int lane = tid & 31;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(
          0xffffffffu, value, offset);
    }
    if (lane == 0) {
      shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
      value = lane < kWmmaWarps ? shared[lane] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffffu, value, offset);
      }
      if (lane == 0) {
        shared[0] = value;
      }
    }
    __syncthreads();
    return shared[0];
  }

  __device__ static float block_max(float value, float* shared) {
    int tid = int(threadIdx.x);
    shared[tid] = value;
    if (tid < 128) {
      shared[384 + tid] = -3.402823466e+38f;
    }
    __syncthreads();
    for (int stride = 256; stride >= 64; stride >>= 1) {
      if (tid < stride) {
        shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
      }
      __syncthreads();
    }
    if (tid < 32) {
      value = fmaxf(shared[tid], shared[tid + 32]);
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(
              0xffffffffu, value, offset));
      }
      if (tid == 0) {
        shared[0] = value;
      }
    }
    __syncthreads();
    return shared[0];
  }

  __device__ static float bf16(const __nv_bfloat16* ptr,
                               std::size_t index) {
    return __bfloat162float(ptr[index]);
  }

  __device__ __forceinline__ static void mma_m16n8k16_bf16(
      float& d0, float& d1, float& d2, float& d3, unsigned a0,
      unsigned a1, unsigned a2, unsigned a3, unsigned b0,
      unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
          "r"(b1));
  }

  __device__ __forceinline__ static float horizontal_add_bf16(
      __nv_bfloat162 value) {
    __nv_bfloat162 low =
        __halves2bfloat162(value.x, value.x);
    __nv_bfloat162 high =
        __halves2bfloat162(value.y, value.y);
    return __bfloat162float(__hadd2(low, high).x);
  }

  __device__ __forceinline__ static unsigned scaled_fp8x2_to_bf16x2(
      const __nv_fp8_e4m3* values, std::size_t index,
      __nv_bfloat16 scale) {
    unsigned short raw =
        *reinterpret_cast<const unsigned short*>(
            values + index);
    unsigned converted;
    asm(
        "cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(converted)
        : "h"(raw));
    __nv_bfloat162 unpacked =
        *reinterpret_cast<const __nv_bfloat162*>(&converted);
    __nv_bfloat162 packed = __hmul2(
        unpacked, __bfloat162bfloat162(scale));
    return *reinterpret_cast<const unsigned*>(&packed);
  }

  __device__ __forceinline__ static unsigned fp8x2_to_bf16x2(
      const __nv_fp8_e4m3* values, std::size_t index) {
    unsigned short raw =
        *reinterpret_cast<const unsigned short*>(
            values + index);
    unsigned converted;
    asm(
        "cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(converted)
        : "h"(raw));
    return converted;
  }

  __device__ __forceinline__ static unsigned load_fp8x4_raw(
      const __nv_fp8_e4m3* values, std::size_t index) {
    return *reinterpret_cast<const unsigned*>(
        values + index);
  }

  __device__ __forceinline__ static unsigned fp8x2_raw_to_bf16x2(
      unsigned short raw) {
    unsigned converted;
    asm(
        "cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(converted)
        : "h"(raw));
    return converted;
  }

  __device__ __forceinline__ static float horizontal_add4_bf16(
      __nv_bfloat162 even, __nv_bfloat162 odd) {
    __nv_bfloat162 value = __hadd2(
        __halves2bfloat162(even.x, even.x),
        __halves2bfloat162(even.y, even.y));
    value = __hadd2(
        value, __halves2bfloat162(odd.x, odd.x));
    value = __hadd2(
        value, __halves2bfloat162(odd.y, odd.y));
    return __bfloat162float(value.x);
  }

  __device__ __forceinline__ static void cp_async_16(
      void* destination, const void* source, int source_bytes) {
    unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(destination));
    if (source_bytes == 16) {
      asm volatile(
          "cp.async.ca.shared.global [%0], [%1], 16;\n"
          :
          : "r"(shared_address), "l"(source)
          : "memory");
    } else {
      asm volatile(
          "cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
          :
          : "r"(shared_address), "l"(source)
          : "memory");
    }
  }

  __device__ __forceinline__ static void prefetch_l2_bulk(
      const void* source, unsigned bytes) {
    asm volatile(
        "cp.async.bulk.prefetch.L2.global [%0], %1;\n"
        :
        : "l"(source), "r"(bytes)
        : "memory");
  }

  __device__ __forceinline__ static void cp_async_wait_all() {
    asm volatile(
        "cp.async.commit_group;\n"
        "cp.async.wait_group 0;\n"
        :
        :
        : "memory");
  }

  __device__ __forceinline__ static void cp_async_commit_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory");
  }

  __device__ __forceinline__ static void cp_async_wait_group0() {
    asm volatile(
        "cp.async.wait_group 0;\n"
        :
        :
        : "memory");
  }
