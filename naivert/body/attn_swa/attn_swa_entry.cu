// Cooperative entry for the Naive N0.5 sliding-window attention layer step.
#include "naivert/body/attn_swa/attn_swa_body.cuh"

namespace naivert::attn_swa {

__global__ void __launch_bounds__(AttnSwaBody::kThreads, 1)
    attn_swa_rows1(const AttnSwaArgs args) {
  AttnSwaBody::run(args);
}

template <bool kWideGamma>
__global__ void __launch_bounds__(AttnSwaBody::kThreads, 1)
    attn_swa_rows1_gamma_aligned(const AttnSwaArgs args) {
  AttnSwaBodyImpl<kWideGamma>::run(args);
}

// Launches one CTA per SM (grid) with a grid-wide barrier available.
cudaError_t launch_attn_swa_rows1(const AttnSwaArgs& args, int grid,
                                  cudaStream_t stream) {
  // The projection tile / W_o panes live in dynamic shared memory (opt-in above 48 KiB). The
  // attribute is per device: set it once on every device this process launches on (TP8 = 8 devices).
  bool aligned = false;
  const void* kernel = reinterpret_cast<const void*>(&attn_swa_rows1);
  if constexpr (AttnSwaScratchLayout::kUseProjectionTma) {
    // Selection happens before graph capture. The fallback preserves the
    // existing float4 alignment contract without a per-load device branch.
    aligned = (reinterpret_cast<std::uintptr_t>(args.input_norm) & 31u) == 0;
    if (aligned) kernel = reinterpret_cast<const void*>(&attn_swa_rows1_gamma_aligned<true>);
  }
  static unsigned long long attribute_set_by_alignment[2] = {};
  auto& attribute_set_devices = attribute_set_by_alignment[aligned];
  int device = 0;
  if (const cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) {
    return err;
  }
  if (!(attribute_set_devices & (1ull << (device & 63)))) {
    const cudaError_t err = cudaFuncSetAttribute(kernel,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 AttnSwaBody::kDynamicSharedBytes);
    if (err != cudaSuccess) {
      return err;
    }
    attribute_set_devices |= 1ull << (device & 63);
  }
  const void* params[] = {&args};
  return cudaLaunchCooperativeKernel(
      kernel, dim3(grid),
      dim3(AttnSwaBody::kThreads), const_cast<void**>(params), AttnSwaBody::kDynamicSharedBytes,
      stream);
}

}  // namespace naivert::attn_swa
