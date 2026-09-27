#include "naivert/entry/expert_down_domain.hpp"

#include "naivert/body/expert_down_allreduce/expert_down_allreduce_traits.cuh"
#include "naivert/entry/expert_down_allreduce_launch.hpp"
#include "naivert/entry/expert_down_runtime_domain.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace naivert_expert_down_domain {

namespace expert = naivert_replica_expert_down_allreduce;

std::size_t activation_row_elements() {
  return expert::kOutputSlots * expert::kInter;
}

std::size_t domain_size() {
  return sizeof(expert::ExpertDownRuntimeDomain);
}

std::size_t mailbox_row_bytes() {
  return expert::kExpertDownMailboxRowBytes;
}

int descriptor_capacity() {
  return expert::kExpertDownDescriptorCapacity;
}

cudaError_t prepare_source_domain(
    void* domain_ptr, std::size_t domain_bytes,
    const void* up_gate, const void* w_down, const void* w_down_scale,
    const float* selected_probs, const int* selected_indices, void* output,
    const void* residual, const void* const mailbox[8],
    const std::uint64_t* replay_iteration, std::uint32_t mailbox_tag,
    int descriptor_index, int rank, int device, cudaStream_t stream) {
  if (domain_ptr == nullptr ||
      domain_bytes < sizeof(expert::ExpertDownRuntimeDomain) ||
      up_gate == nullptr || w_down == nullptr || w_down_scale == nullptr ||
      selected_probs == nullptr || selected_indices == nullptr ||
      output == nullptr || residual == nullptr || mailbox == nullptr ||
      mailbox_tag == 0u || descriptor_index < 0 ||
      descriptor_index >= expert::kExpertDownDescriptorCapacity ||
      rank < 0 || rank >= expert::kExpertDownDomainRanks) {
    return cudaErrorInvalidValue;
  }
  for (int peer = 0; peer < expert::kExpertDownDomainRanks; ++peer) {
    if (mailbox[peer] == nullptr) {
      return cudaErrorInvalidValue;
    }
  }
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }

  expert::ExpertDownRuntimeDomain domain{};
  domain.magic = expert::kExpertDownDomainMagic;
  domain.version = expert::kExpertDownDomainVersion;
  domain.rank = static_cast<std::uint32_t>(rank);
  domain.nranks = expert::kExpertDownDomainRanks;
  domain.native_mailbox_tag = mailbox_tag;
  domain.native_mailbox_slab = expert::kExpertDownMailboxSlab;
  domain.step_block_ptr = reinterpret_cast<std::uint64_t>(replay_iteration);
  for (int peer = 0; peer < expert::kExpertDownDomainRanks; ++peer) {
    domain.peer_mailbox_base_ptrs[peer] =
        reinterpret_cast<std::uint64_t>(mailbox[peer]);
  }

  expert::ExpertDownDescriptor descriptor{};
  descriptor.args.up_gate = reinterpret_cast<const naivert::ExpertActivation*>(up_gate);
  descriptor.args.moe_w_down =
      reinterpret_cast<const __nv_fp8_e4m3*>(w_down);
  descriptor.args.moe_w_down_scale =
      reinterpret_cast<const naivert::ExpertWeightScale*>(w_down_scale);
  descriptor.args.sel_probs = selected_probs;
  descriptor.args.sel_indices = selected_indices;
  descriptor.args.exp_out = reinterpret_cast<__nv_bfloat16*>(output);
  descriptor.args.residual =
      reinterpret_cast<const __nv_bfloat16*>(residual);
  descriptor.args.tp_result = reinterpret_cast<__nv_bfloat16*>(output);
  descriptor.args.dim = expert::kHidden;
  descriptor.args.moe_inter_dim = expert::kInter;
  descriptor.args.n_activated = expert::kSelectedExperts;
  descriptor.args.rank = rank;
  descriptor.args.stable_domain =
      reinterpret_cast<const expert::ExpertDownRuntimeDomain*>(domain_ptr);

  status = cudaMemcpyAsync(
      domain_ptr, &domain, sizeof(domain), cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = expert::initialize_expert_down_allreduce_stable_descriptor(
      &descriptor, descriptor_index, device, stream);
  if (status != cudaSuccess) {
    return status;
  }
  // The records above are host stack objects and preparation runs outside
  // capture; synchronizing here makes their lifetime unambiguous.
  return cudaStreamSynchronize(stream);
}

void launch_source_domain(
    const void* domain_ptr, int descriptor_index, int rank, int device,
    cudaStream_t stream) {
  expert::launch_expert_down_allreduce_with_source_domain(
      domain_ptr, descriptor_index, rank, device, stream);
}

void launch_source_domain_rows(
    const void* domain_ptr, int descriptor_index, int rank, int active_rows,
    int device, cudaStream_t stream) {
  expert::launch_expert_down_allreduce_with_source_domain_rows(
      domain_ptr, descriptor_index, rank, active_rows, device, stream);
}

void launch_source_domain_rows_pdl_secondary(
    const void* domain_ptr, int descriptor_index, int rank, int active_rows,
    int device, cudaStream_t stream) {
  expert::launch_expert_down_allreduce_with_source_domain_rows_pdl_secondary(
      domain_ptr, descriptor_index, rank, active_rows, device, stream);
}

}  // namespace naivert_expert_down_domain
