#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

namespace naivert_tile_mailbox_test {
void launch_tile_allreduce(const void* input, void* output, const void* const* peers, int rank,
                           unsigned generation, int device, cudaStream_t stream);
int tile_count();
int mailbox_bytes_per_rank();
}  // namespace naivert_tile_mailbox_test

namespace {

void enable_peer_access(int64_t ranks) {
  int restore = 0;
  C10_CUDA_CHECK(cudaGetDevice(&restore));
  for (int a = 0; a < ranks; ++a) {
    C10_CUDA_CHECK(cudaSetDevice(a));
    for (int b = 0; b < ranks; ++b) {
      if (a == b) continue;
      cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); continue; }
      C10_CUDA_CHECK(err);
    }
  }
  C10_CUDA_CHECK(cudaSetDevice(restore));
}

void tile_allreduce(at::Tensor input, at::Tensor output, at::Tensor peers, int64_t rank, int64_t generation) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && peers.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "bf16 expected");
  TORCH_CHECK(peers.scalar_type() == at::kLong && peers.numel() == 8, "peers: int64[8] of mailbox bases");
  const int device = static_cast<int>(input.get_device());
  TORCH_CHECK(output.get_device() == device && peers.get_device() == device, "tensors must share the rank device");
  c10::cuda::CUDAGuard guard(device);
  naivert_tile_mailbox_test::launch_tile_allreduce(
      input.data_ptr(), output.data_ptr(), reinterpret_cast<const void* const*>(peers.data_ptr<int64_t>()),
      static_cast<int>(rank), static_cast<unsigned>(generation), device,
      at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enable_peer_access", &enable_peer_access);
  m.def("tile_allreduce", &tile_allreduce);
  m.def("tile_count", &naivert_tile_mailbox_test::tile_count);
  m.def("mailbox_bytes_per_rank", &naivert_tile_mailbox_test::mailbox_bytes_per_rank);
}
