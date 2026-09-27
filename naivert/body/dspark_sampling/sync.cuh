#pragma once
#include <cuda/atomic>
namespace naivert_dspark_ops {
using Atomic = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
// Runtime flags array per rank (int64 words): [0] ready, [1] consumed, [2] generation,
// [kRendezvousInbox + src] = the consumed generation rank `src` pushed into THIS rank's memory.
// The tail's rendezvous used to poll every peer's own [1] word over NVLink, one sequential
// acquire load per peer (8 remote round trips after the last arrival). The pushed inbox keeps
// [1] as the peer-visible truth and turns the wait into eight local polls by eight lanes.
constexpr int kRendezvousInbox = 3;
constexpr int kRendezvousWords = 16;
__device__ __forceinline__ bool sampling_rendezvous(
    unsigned long long* const* flags, int rank, unsigned* watchdog, bool consumed) {
  auto* own = flags[rank];
  const auto generation = consumed ? own[2] : ++own[2];
  __threadfence_system();
  Atomic(own[consumed ? 1 : 0]).store(generation, cuda::memory_order_release);
  unsigned long long start;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  for (int r = 0; r < 8; ++r) {
    while (Atomic(flags[r][consumed ? 1 : 0]).load(cuda::memory_order_acquire) < generation) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - start > 30000000000ull) {
        atomicCAS(watchdog, 0u, 0xD6FAu);
        return false;
      }
      __nanosleep(64);
    }
  }
  return true;
}
// Consumed rendezvous, push form. One full warp calls it; the caller has already advanced
// own[2]. Lane 0 fences and publishes own[1] exactly as before (protocol.py and any peer reader
// see the same word), then lanes 0..7 push the generation into every peer's inbox slot for this
// rank and poll only their own local inbox word. Returns the same value in every lane.
__device__ __forceinline__ bool sampling_rendezvous_push(
    unsigned long long* const* flags, int rank, unsigned* watchdog) {
  const int lane = int(threadIdx.x) & 31;
  auto* own = flags[rank];
  unsigned long long generation = 0;
  if (lane == 0) {
    generation = own[2];
    __threadfence_system();
    Atomic(own[1]).store(generation, cuda::memory_order_release);
  }
  generation = __shfl_sync(0xffffffffu, generation, 0);
  if (lane < 8) Atomic(flags[lane][kRendezvousInbox + rank]).store(generation, cuda::memory_order_relaxed);
  bool ok = true;
  if (lane < 8) {
    unsigned long long start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    while (Atomic(own[kRendezvousInbox + lane]).load(cuda::memory_order_acquire) < generation) {
      unsigned long long now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      if (now - start > 30000000000ull) {
        atomicCAS(watchdog, 0u, 0xD6FAu);
        ok = false;
        break;
      }
      __nanosleep(64);
    }
  }
  return __all_sync(0xffffffffu, ok);
}
}
