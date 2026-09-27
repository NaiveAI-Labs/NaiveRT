#include "naivert/entry/dspark_head_markov_launch.hpp"
#include "naivert/body/dspark_head_markov/head_body.cuh"
#include "naivert/body/dspark_head_markov/cooperative.cuh"
#include "naivert/body/attn_common/strided_rows_mma.cuh"

namespace naivert_dspark_ops {
constexpr int kHeadLogitStride = 146;
constexpr int kHeadMarkovSmem = kHeadSmem + 7 * kHeadLogitStride * sizeof(float);
static_assert(kHeadMarkovSmem <= 232448 && kHeadCtas == 148);

__global__ __launch_bounds__(kHeadThreads, 1) void naivert_dspark_head_rows8_kernel(
    const __grid_constant__ HeadArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  HeadBody::run<false>(memory, args);
}
__global__ __launch_bounds__(kMkThreads, 2) void naivert_dspark_markov_block_kernel(
    const __grid_constant__ MarkovArgs args) {
  MarkovBody::run(args);
}
__global__ __launch_bounds__(kHeadThreads, 1) void naivert_dspark_head_markov_kernel(
    const __grid_constant__ HeadArgs head, const __grid_constant__ MarkovArgs markov,
    unsigned long long* partials, unsigned long long* winner) {
  extern __shared__ __align__(128) unsigned char memory[];
  HeadBody::run<true, kHeadLogitStride>(memory, head);
  markov_cooperative_steps<160, false, kHeadLogitStride>(
      markov, reinterpret_cast<const float*>(memory + kHeadSmem), partials, winner);
}

// Initialization only. Retain the reference's BF16 bias boundary for every
// possible previous token and every column in this rank's vocabulary shard.
__global__ __launch_bounds__(128) void naivert_dspark_precompute_bias_table(
    const __nv_bfloat16* w1, const __nv_bfloat16* w2, __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 queries[8 * kMkRank];
  __shared__ float partials[4 * 128];
  const int first = int(blockIdx.y) * 8;
  for (int i = int(threadIdx.x); i < 8 * kMkRank; i += 128)
    queries[i] = __float2bfloat16_rn(__fadd_rn(0.0f, __bfloat162float(w1[std::size_t(first) * kMkRank + i])));
  __syncthreads();
  naivert::attn_rows::project_strided_rows_mma<8, kMkRank, 1, 4, 4>(
      0, kMkVocab / 16,
      [&](int n) { return w2 + std::size_t(n) * kMkRank; }, queries, partials,
      [&](int n, int row, float value) {
        output[std::size_t(first + row) * kMkVocab + n] = __float2bfloat16_rn(value);
      });
}

cudaError_t configure_head_markov(int device) {
  auto rc = cudaSetDevice(device);
  if (rc != cudaSuccess) return rc;
  int sms = 0;
  rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (rc != cudaSuccess) return rc;
  const void* kernels[] = {reinterpret_cast<const void*>(naivert_dspark_head_rows8_kernel),
      reinterpret_cast<const void*>(naivert_dspark_markov_block_kernel),
      reinterpret_cast<const void*>(naivert_dspark_head_markov_kernel)};
  const int threads[] = {kHeadThreads, kMkThreads, kHeadThreads};
  const int bytes[] = {kHeadSmem, 0, kHeadMarkovSmem};
  const int ctas[] = {kHeadCtas, kMkCtas, kHeadCtas};
  for (int i = 0; i < 3; ++i) {
    if (bytes[i]) {
      rc = cudaFuncSetAttribute(kernels[i], cudaFuncAttributeMaxDynamicSharedMemorySize, bytes[i]);
      if (rc != cudaSuccess) return rc;
    }
    int active = 0;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernels[i], threads[i], bytes[i]);
    if (rc != cudaSuccess) return rc;
    if (static_cast<long long>(active) * sms < ctas[i]) return cudaErrorInvalidConfiguration;
  }
  return cudaSuccess;
}
cudaError_t launch_head_rows8(const HeadArgs& args, cudaStream_t stream) {
  naivert_dspark_head_rows8_kernel<<<kHeadCtas, kHeadThreads, kHeadSmem, stream>>>(args);
  return cudaGetLastError();
}
cudaError_t launch_markov_block(const MarkovArgs& args, cudaStream_t stream) {
  naivert_dspark_markov_block_kernel<<<kMkCtas, kMkThreads, 0, stream>>>(args);
  return cudaGetLastError();
}
cudaError_t launch_head_markov(const HeadArgs& head, const MarkovArgs& markov,
                              unsigned long long* control, cudaStream_t stream) {
  naivert_dspark_head_markov_kernel<<<kHeadCtas, kHeadThreads, kHeadMarkovSmem, stream>>>(
      head, markov, control, control + 2 * kHeadCtas);
  return cudaGetLastError();
}
cudaError_t launch_precompute_bias_table(const __nv_bfloat16* w1, const __nv_bfloat16* w2,
                                       __nv_bfloat16* output, cudaStream_t stream) {
  naivert_dspark_precompute_bias_table<<<dim3(kMkVocab / 64, kMkVocab), 128, 0, stream>>>(w1, w2, output);
  return cudaGetLastError();
}
int head_rows8_ctas() { return kHeadCtas; }
int head_rows8_vocab() { return kHeadVocab; }
int head_markov_control_elements() { return 2 * kHeadCtas + 2; }
int head_markov_threads(int operation) { return operation == 1 ? kMkThreads : kHeadThreads; }
int head_markov_shared_bytes(int operation) {
  return operation == 0 ? kHeadSmem : operation == 1 ? 0 : operation == 2 ? kHeadMarkovSmem : -1;
}
int markov_block_ctas() { return kMkCtas; }
int markov_block_scratch_elements() { return kMkScratchElements; }
int markov_block_latent() { return kMkRank; }
int head_markov_mailbox_bytes() { return kMailboxSlabsPerRank * naivert::kMailboxSlabStride; }
std::int64_t markov_bias_table_elements() { return std::int64_t(8) * kMkVocab * kMkVocab; }
std::vector<std::int64_t> head_rows8_weight_shape() { return {kTiles, 4, 32, 16, 32}; }
std::vector<std::int64_t> head_norm_gamma_layout() { return {32, 2, 4}; }
}  // namespace naivert_dspark_ops
