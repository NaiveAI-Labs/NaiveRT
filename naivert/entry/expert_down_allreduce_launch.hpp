#pragma once

#include <cuda_runtime_api.h>

namespace naivert_replica_expert_down_allreduce {

// Source-owned production stable-domain specialization. The caller supplies
// direct source tensors, source layer tags, and allocator-owned destination
// mailbox tensors before graph capture; replay keeps this parameter tuple.
void launch_expert_down_allreduce_with_source_domain(
    const void* domain, int descriptor_index, int tp_rank, int device,
    cudaStream_t stream);

void launch_expert_down_allreduce_with_source_domain_pdl_producer(
    const void* domain, int descriptor_index, int tp_rank, int device,
    cudaStream_t stream);

void launch_expert_down_allreduce_with_source_domain_rows(
    const void* domain, int descriptor_index, int tp_rank, int active_rows,
    int device, cudaStream_t stream);

void launch_expert_down_allreduce_with_source_domain_rows_pdl_secondary(
    const void* domain, int descriptor_index, int tp_rank, int active_rows,
    int device, cudaStream_t stream);

cudaError_t initialize_expert_down_allreduce_stable_descriptor(
    const void* host_descriptor, int descriptor_index, int device,
    cudaStream_t stream);

}  // namespace naivert_replica_expert_down_allreduce
