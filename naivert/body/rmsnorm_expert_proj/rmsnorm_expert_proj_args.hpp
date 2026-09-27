#pragma once

#include <cuda_bf16.h>
#include "naivert/common/expert_activation.hpp"

namespace naivert_replica_rmsnorm_expert_proj {

struct RMSNormExpertProjArgs {
  const __nv_bfloat16* hidden;
  const float* gamma;
  const naivert::RouterWeight* router_weight;
  float* scores;
  __nv_bfloat16* x_mlp;
  int tokens;
  int hidden_dim;
  int experts;
};

}  // namespace naivert_replica_rmsnorm_expert_proj
