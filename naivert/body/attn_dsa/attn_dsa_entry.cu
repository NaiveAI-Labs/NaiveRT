// Cooperative entry for the Naive N0.5 full-attention layer step.
#include "naivert/body/attn_dsa/attn_dsa_body.cuh"

namespace naivert::attn_dsa {

__global__ void __launch_bounds__(AttnDsaBody::kThreads, 1)
    attn_dsa_rows1(const AttnDsaArgs args) {
  AttnDsaBody::run(args);
}

template <bool kWideGamma>
__global__ void __launch_bounds__(AttnDsaBody::kThreads, 1)
    attn_dsa_rows1_gamma_aligned(const AttnDsaArgs args) {
  AttnDsaBodyImpl<kWideGamma>::run(args);
}

// Launches one CTA per SM (grid) with a grid-wide barrier available.
cudaError_t launch_attn_dsa_rows1(const AttnDsaArgs& args, int grid,
                                  cudaStream_t stream) {
  const void* kernel = reinterpret_cast<const void*>(&attn_dsa_rows1);
  if constexpr (AttnDsaScratchLayout::kUseProjectionTma) {
    if ((reinterpret_cast<std::uintptr_t>(args.input_norm) & 31u) == 0)
      kernel = reinterpret_cast<const void*>(&attn_dsa_rows1_gamma_aligned<true>);
  }
  const void* params[] = {&args};
  return cudaLaunchCooperativeKernel(
      kernel, dim3(grid),
      dim3(AttnDsaBody::kThreads), const_cast<void**>(params), 0, stream);
}

}  // namespace naivert::attn_dsa
