#include "naivert/entry/dspark_attention_launch.hpp"
#include "naivert/body/dspark_attention/body.cuh"

namespace naivert_dspark_ops {
__global__ __launch_bounds__(kAttnThreads, 1) void naivert_dspark_attn_rows8_kernel(
    const __grid_constant__ AttnArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  AttnBody::run<false>(args, memory);
}
__global__ __launch_bounds__(kAttnThreads, 1) void naivert_dspark_attn_rows8_swa_kernel(
    const __grid_constant__ SwaAttnArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  AttnBody::run<true>(args, memory);
}

cudaError_t configure_attn_rows8(int device) {
  auto rc = cudaSetDevice(device);
  if (rc != cudaSuccess) return rc;
  int sms = 0;
  rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (rc != cudaSuccess) return rc;
  for (auto kernel : {reinterpret_cast<const void*>(naivert_dspark_attn_rows8_kernel),
                     reinterpret_cast<const void*>(naivert_dspark_attn_rows8_swa_kernel)}) {
    rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem);
    if (rc != cudaSuccess) return rc;
    int per_sm = 0;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kAttnThreads, kAttnSmem);
    if (rc != cudaSuccess) return rc;
    if (static_cast<long long>(per_sm) * sms < kAttnCtas) return cudaErrorInvalidConfiguration;
  }
  return cudaSuccess;
}
cudaError_t launch_attn_rows8(const AttnArgs& args, cudaStream_t stream) {
  naivert_dspark_attn_rows8_kernel<<<kAttnCtas, kAttnThreads, kAttnSmem, stream>>>(args);
  return cudaGetLastError();
}
cudaError_t launch_attn_rows8_swa(const SwaAttnArgs& args, cudaStream_t stream) {
  naivert_dspark_attn_rows8_swa_kernel<<<kAttnCtas, kAttnThreads, kAttnSmem, stream>>>(args);
  return cudaGetLastError();
}
int attn_rows8_ctas() { return kAttnCtas; }
int attn_rows8_threads() { return kAttnThreads; }
int attn_rows8_shared_bytes() { return kAttnSmem; }
int attn_rows8_split_elems() { return kSplitScratch + kReadyWords; }
int attn_rows8_salt(int layer) { return int(kSaltD3Base) + 2 * layer; }
int attn_rows8_max_padded_cols() { return kAttnMain / int(sizeof(float)); }
int attn_rows8_qk_raw_elems() { return kAttnWindow * kQkRaw; }
}  // namespace naivert_dspark_ops
