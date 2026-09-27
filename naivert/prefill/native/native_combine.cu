// Return expert outputs for a 4K prefill chunk using tagged-word transport.
// Each input is a complete BF16 expert row; combine weights and sums in FP32.
#include "naivert/common/tp8_mailbox.cuh"
#include <cuda_runtime.h>

namespace prefill_combine {
constexpr int H = 4096, LocalTokens = 512, Experts = 256, Pairs = H / 2;
constexpr int Slots = NAIVERT_COMBINE_COMPACT ? 8 : Experts;
constexpr size_t ControlWords = 16;
constexpr size_t BankWords = size_t(LocalTokens) * Slots * Pairs;
// Kernel arguments are copied by value. The host's pointer array itself is not
// CUDA memory, even though each element names a peer-accessible allocation.
struct PeerBases { unsigned long long* base[8]; };

__device__ unsigned long long checked_wait(const unsigned long long* ptr,
                                           unsigned tag, int* error) {
  const auto start = clock64();
  for (;;) {
    const auto word = naivert::load_tagged<true>(ptr);
    if (naivert::tagged_tag(word) == tag) return word;
    if (clock64() - start > 3000000000ull) {
      atomicExch(error, 1);
      return naivert::tagged_word(tag, 0);
    }
  }
}

__global__ void publish(const unsigned* input, const long long* source_rows,
                        const long long* destinations, int rows, const int* rows_dev,
                        PeerBases peers, int rank,
                        const unsigned long long* epoch) {
  // naivert prefill: the live pair count may come from device memory (static-shape graphs);
  // ``rows`` then bounds the arrays.
  if (rows_dev != nullptr) rows = min(rows, *rows_dev);
  const unsigned generation = unsigned(*epoch);
  const unsigned phase = generation & 1u;
  // Even an empty sender participates. Before a rank can publish g+2, its
  // receive(g+1) waits for every sender to enter g+1, which follows receive(g).
  // Thus no g+2 write can overwrite a bank while any rank still consumes g.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int peer = 0; peer < 8; ++peer)
      naivert::store_tagged(peers.base[peer] + phase * 8 + rank,
                           naivert::tagged_word(generation, 0));
  }
  // naivert prefill: grid-stride over the live pairs (the grid is sized for a typical count,
  // the array bound may be far larger).
  const size_t total = size_t(rows) * Pairs;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x; linear < total; linear += stride) {
    const size_t row = linear / Pairs, pair = linear % Pairs;
    const size_t destination = size_t(destinations[row]);
    const int owner = int(destination / (LocalTokens * Slots));
    const size_t local = destination % (LocalTokens * Slots);
    const unsigned payload = input[size_t(source_rows[row]) * Pairs + pair];
    naivert::store_tagged(peers.base[owner] + ControlWords + phase * BankWords + local * Pairs + pair,
                         naivert::tagged_word(generation, payload));
  }
}

__global__ void complete(const unsigned long long* local, const long long* ids,
                         const float* probabilities, unsigned* output,
                         const unsigned long long* epoch, int* error) {
  const unsigned generation = unsigned(*epoch), phase = generation & 1u;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (int peer = 0; peer < 8; ++peer)
      checked_wait(local + phase * 8 + peer, generation, error);
  const int linear = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= LocalTokens * Pairs) return;
  const int token = linear / Pairs, pair = linear % Pairs;
  unsigned long long words[8];
  const unsigned long long* addresses[8];
  unsigned ready = 0;
#pragma unroll
  for (int slot = 0; slot < 8; ++slot)
    addresses[slot] = local + ControlWords + phase * BankWords +
                      (size_t(token) * Slots + (NAIVERT_COMBINE_COMPACT ? slot : ids[token * 8 + slot])) * Pairs + pair;
  const auto start = clock64();
  while (ready != 255u) {
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (!(ready & (1u << slot))) {
        words[slot] = naivert::load_tagged<true>(addresses[slot]);
        if (naivert::tagged_tag(words[slot]) == generation) ready |= 1u << slot;
      }
    }
    if (clock64() - start > 3000000000ull) {
      atomicExch(error, 2);
      output[linear] = 0;
      return;
    }
  }
  float lo = 0.0f, hi = 0.0f;
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const unsigned payload = naivert::tagged_payload(words[slot]);
    const float p = probabilities[token * 8 + slot];
    lo = __fadd_rn(lo, __fmul_rn(naivert::payload_lane(payload, 0), p));
    hi = __fadd_rn(hi, __fmul_rn(naivert::payload_lane(payload, 1), p));
  }
  output[linear] = naivert::pack_bf16x2(lo, hi);
}

void launch(const void* input, const long long* source_rows, const long long* destinations,
            int rows, const int* rows_dev, void* const* peers, int rank, const long long* ids,
            const float* probabilities, void* output, const void* epoch, int* error,
            cudaStream_t stream) {
  // grid for ~4096 pairs (one card's average share); larger live counts loop
  const int blocks = int((min(size_t(rows), size_t(4096)) * Pairs + 255) / 256);
  PeerBases bases;
  for (int peer = 0; peer < 8; ++peer)
    bases.base[peer] = static_cast<unsigned long long*>(peers[peer]);
  publish<<<blocks > 0 ? blocks : 1, 256, 0, stream>>>(
      static_cast<const unsigned*>(input), source_rows, destinations, rows, rows_dev,
      bases, rank,
      static_cast<const unsigned long long*>(epoch));
  complete<<<LocalTokens * Pairs / 256, 256, 0, stream>>>(
      reinterpret_cast<const unsigned long long*>(peers[rank]), ids, probabilities,
      static_cast<unsigned*>(output), static_cast<const unsigned long long*>(epoch), error);
}

size_t mailbox_bytes() { return (ControlWords + 2 * BankWords) * sizeof(unsigned long long); }
} // namespace prefill_combine
