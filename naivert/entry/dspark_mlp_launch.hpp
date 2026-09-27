#pragma once

#include "naivert/body/dspark_mlp/args.hpp"
#include <cuda_runtime_api.h>

namespace naivert_dspark_ops {
// Configure and prove whole-grid residency on the selected device before
// capture/launch. A failed configuration must never be followed by a launch.
cudaError_t configure_mlp_rows8(int device);
cudaError_t launch_mlp_rows8(const MlpArgs& args, cudaStream_t stream);
int mlp_rows8_ctas();
int mlp_rows8_threads();
int mlp_rows8_shared_bytes();
}  // namespace naivert_dspark_ops
