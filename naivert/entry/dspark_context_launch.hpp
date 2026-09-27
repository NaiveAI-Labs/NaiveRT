#pragma once
#include "naivert/body/dspark_context/args.hpp"

namespace naivert_dspark_ops {
// Configure before capture. Caller owns peer access, initialized state, valid
// positions/destinations, buffer lifetimes and lockstep resident-grid scheduling.
cudaError_t configure_context(int device);
cudaError_t launch_fc_ctx(const FcCtxArgs&, cudaStream_t);
cudaError_t launch_ctx_kv(const CtxKvArgs&, cudaStream_t);
cudaError_t launch_fc_ctx_kv(const FcCtxKvArgs&, cudaStream_t);
int context_ctas();
int context_threads();
int context_shared_bytes(int operation);  // 0: D1, 1: D2, 2: D12
int fc_ctx_kv_partial_elements();
int context_mailbox_bytes();
}  // namespace naivert_dspark_ops
