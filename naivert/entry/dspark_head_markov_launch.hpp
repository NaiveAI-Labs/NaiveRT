#pragma once
#include "naivert/body/dspark_head_markov/args.hpp"
#include <vector>

namespace naivert_dspark_ops {
// Prepare on each GPU before capture. Caller owns peer access, initialized
// controls, shared epoch, buffer lifetime and exclusive resident-grid scheduling.
cudaError_t configure_head_markov(int device);
cudaError_t launch_head_rows8(const HeadArgs&, cudaStream_t);
cudaError_t launch_markov_block(const MarkovArgs&, cudaStream_t);
cudaError_t launch_head_markov(const HeadArgs&, const MarkovArgs&,
                              unsigned long long* control, cudaStream_t);
cudaError_t launch_precompute_bias_table(const __nv_bfloat16*, const __nv_bfloat16*,
                                       __nv_bfloat16*, cudaStream_t);
int head_rows8_ctas();
int head_rows8_vocab();
int head_markov_control_elements();
int head_markov_threads(int operation);  // 0: D5, 1: D6, 2: D56
int head_markov_shared_bytes(int operation);
int markov_block_ctas();
int markov_block_scratch_elements();
int markov_block_latent();
int head_markov_mailbox_bytes();
std::int64_t markov_bias_table_elements();
std::vector<std::int64_t> head_rows8_weight_shape();
std::vector<std::int64_t> head_norm_gamma_layout();
}  // namespace naivert_dspark_ops
