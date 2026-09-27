#pragma once
#include "naivert/body/dspark_attention/args.hpp"
#include <cuda_runtime_api.h>

namespace naivert_dspark_ops {
// Configure both kernels before capture. Caller owns exclusive residency,
// peer access, distinct scratch buffers and lockstep epoch/layer execution.
cudaError_t configure_attn_rows8(int device);
cudaError_t launch_attn_rows8(const AttnArgs&, cudaStream_t);
cudaError_t launch_attn_rows8_swa(const SwaAttnArgs&, cudaStream_t);
int attn_rows8_ctas();
int attn_rows8_threads();
int attn_rows8_shared_bytes();
int attn_rows8_split_elems();
int attn_rows8_salt(int layer);
int attn_rows8_max_padded_cols();
int attn_rows8_qk_raw_elems();
}  // namespace naivert_dspark_ops
