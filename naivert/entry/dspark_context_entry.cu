#include "naivert/entry/dspark_context_launch.hpp"
#include "naivert/body/dspark_context/body.cuh"

namespace naivert_dspark_ops {
constexpr int kFcSmem = kWindow * kHidden * 2 +
    (2 * kWindow * kColsPerCta + kWindow + kShellWarps * 128) * sizeof(float);
constexpr int kKvSmem = kWindow * kHidden * 2 + kWarps * sizeof(float) +
    kKvRowsR * sizeof(__nv_bfloat16) + kWarps * 128 * sizeof(float);
constexpr int kFusedSmem = kD12WeightOffset + kD12WeightBytes;
static_assert(kRowSlabs == kWindow);

__global__ __launch_bounds__(kThreads, 1) void naivert_dspark_fc_ctx_kernel(
    const __grid_constant__ FcCtxArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  __syncthreads();  // Collective initialization join.
  FcCtxBody::run(memory, args);
}
__global__ __launch_bounds__(kThreads) void naivert_dspark_ctx_kv_kernel(
    const __grid_constant__ CtxKvArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  __syncthreads();
  CtxKvBody::run(memory, args);
}
__global__ __launch_bounds__(kThreads, 1) void naivert_dspark_fc_ctx_kv_kernel(
    const __grid_constant__ FcCtxKvArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  __syncthreads();
  FcCtxKvBody::run(memory, args);
}

cudaError_t configure_context(int device) {
  auto rc = cudaSetDevice(device);
  if (rc != cudaSuccess) return rc;
  int sms = 0;
  rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (rc != cudaSuccess) return rc;
  const void* kernels[] = {reinterpret_cast<const void*>(naivert_dspark_fc_ctx_kernel),
      reinterpret_cast<const void*>(naivert_dspark_ctx_kv_kernel), reinterpret_cast<const void*>(naivert_dspark_fc_ctx_kv_kernel)};
  const int bytes[] = {kFcSmem, kKvSmem, kFusedSmem};
  for (int i = 0; i < 3; ++i) {
    rc = cudaFuncSetAttribute(kernels[i], cudaFuncAttributeMaxDynamicSharedMemorySize, bytes[i]);
    if (rc != cudaSuccess) return rc;
    int per_sm = 0;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernels[i], kThreads, bytes[i]);
    if (rc != cudaSuccess) return rc;
    if (static_cast<long long>(per_sm) * sms < kCtas) return cudaErrorInvalidConfiguration;
  }
  return cudaSuccess;
}
cudaError_t launch_fc_ctx(const FcCtxArgs& args, cudaStream_t stream) {
  naivert_dspark_fc_ctx_kernel<<<kCtas, kThreads, kFcSmem, stream>>>(args);
  return cudaGetLastError();
}
cudaError_t launch_ctx_kv(const CtxKvArgs& args, cudaStream_t stream) {
  naivert_dspark_ctx_kv_kernel<<<kCtasD2, kThreads, kKvSmem, stream>>>(args);
  return cudaGetLastError();
}
cudaError_t launch_fc_ctx_kv(const FcCtxKvArgs& args, cudaStream_t stream) {
  naivert_dspark_fc_ctx_kv_kernel<<<kCtas, kThreads, kFusedSmem, stream>>>(args);
  return cudaGetLastError();
}
int context_ctas() { return kCtas; }
int context_threads() { return kThreads; }
int context_shared_bytes(int operation) {
  return operation == 0 ? kFcSmem : operation == 1 ? kKvSmem : operation == 2 ? kFusedSmem : -1;
}
int fc_ctx_kv_partial_elements() { return kD12Splits * kTotalCols * kWindow; }
int context_mailbox_bytes() { return kMailboxSlabsPerRank * naivert::kMailboxSlabStride; }
}  // namespace naivert_dspark_ops
