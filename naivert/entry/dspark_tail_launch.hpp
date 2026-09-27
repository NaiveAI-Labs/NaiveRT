#pragma once
#include <cuda_runtime_api.h>
#include "naivert/body/dspark_tail/args.hpp"

namespace naivert_dspark_tail {
cudaError_t launch_geometry(const GeometryArgs&, cudaStream_t);
cudaError_t launch_accept(const AcceptArgs&, cudaStream_t);
cudaError_t launch_tail(const EpilogueArgs&, cudaStream_t, bool chain, bool staged);
}  // namespace naivert_dspark_tail
