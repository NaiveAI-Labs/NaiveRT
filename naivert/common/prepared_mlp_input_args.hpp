#pragma once
#include "naivert/common/prepared_mlp_input_layout.hpp"
#include "naivert/body/rmsnorm_expert_proj/rmsnorm_expert_proj_args.hpp"
#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_args.hpp"
// Separate descriptors retain every existing operator's parameter ABI.
namespace naivert_prepared_router {
struct Args {
  naivert_replica_rmsnorm_expert_proj::RMSNormExpertProjArgs base;
  unsigned char* prepared_x;
};
}
namespace naivert_prepared_select {
struct Args {
  naivert_replica_expert_select_up_gate_silu::ExpertSelectUpGateSiluArgs base;
  const unsigned char* prepared_x;
};
}
