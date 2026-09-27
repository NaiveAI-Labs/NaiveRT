#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace naivert_tp8_argmax_rows {

struct Tp8ArgmaxRowsArgs {
  const float* logits;
  std::int32_t* token_out;
  float* top_p_scores;
  const std::uint8_t* mailbox_peers[8];
  int rows;
  int vocab;
  int rank;
};

void launch_tp8_argmax_rows(
    const Tp8ArgmaxRowsArgs& args, cudaStream_t stream);

}  // namespace naivert_tp8_argmax_rows
