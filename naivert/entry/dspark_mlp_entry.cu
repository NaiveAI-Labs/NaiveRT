#include "naivert/entry/dspark_mlp_launch.hpp"
#include "naivert/body/dspark_mlp/body.cuh"

namespace naivert_dspark_ops {
__global__ __launch_bounds__(kMlpThreads, 1) void naivert_dspark_mlp_rows8_kernel(
    const __grid_constant__ MlpArgs args) {
  extern __shared__ __align__(128) unsigned char memory[];
  MlpBody::run(args, memory);
}

cudaError_t configure_mlp_rows8(int device) {
  auto rc = cudaSetDevice(device);
  if (rc != cudaSuccess) return rc;
  rc = cudaFuncSetAttribute(naivert_dspark_mlp_rows8_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kMlpSmem);
  if (rc != cudaSuccess) return rc;
  int per_sm = 0, sms = 0;
  rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, naivert_dspark_mlp_rows8_kernel, kMlpThreads, kMlpSmem);
  if (rc != cudaSuccess) return rc;
  rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (rc != cudaSuccess) return rc;
  return static_cast<long long>(per_sm) * sms >= kMlpCtas
      ? cudaSuccess : cudaErrorInvalidConfiguration;
}

cudaError_t launch_mlp_rows8(const MlpArgs& args, cudaStream_t stream) {
  naivert_dspark_mlp_rows8_kernel<<<kMlpCtas, kMlpThreads, kMlpSmem, stream>>>(args);
  return cudaGetLastError();
}

int mlp_rows8_ctas() { return kMlpCtas; }
int mlp_rows8_threads() { return kMlpThreads; }
int mlp_rows8_shared_bytes() { return kMlpSmem; }
}  // namespace naivert_dspark_ops
