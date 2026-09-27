#pragma once

// Host-side writer of the ExpertDown stable domain record and its constant-memory descriptor,
// shared by the seq1 model-ops extension and the MTP draft-ops extension.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace naivert_expert_down_domain {

std::size_t domain_size();
std::size_t mailbox_row_bytes();
std::size_t activation_row_elements();
int descriptor_capacity();

// Writes one layer's domain record and publishes its descriptor. `replay_iteration` may be
// null (the layer tag alone is the mailbox generation); when set, the kernel reads the replay
// epoch from it and uses (epoch << 8) | layer_tag. Synchronizes the stream before returning.
cudaError_t prepare_source_domain(
    void* domain_ptr, std::size_t domain_bytes,
    const void* up_gate, const void* w_down, const void* w_down_scale,
    const float* selected_probs, const int* selected_indices, void* output,
    const void* residual, const void* const mailbox[8],
    const std::uint64_t* replay_iteration, std::uint32_t mailbox_tag,
    int descriptor_index, int rank, int device, cudaStream_t stream);

void launch_source_domain(
    const void* domain_ptr, int descriptor_index, int rank, int device,
    cudaStream_t stream);
void launch_source_domain_rows(
    const void* domain_ptr, int descriptor_index, int rank, int active_rows,
    int device, cudaStream_t stream);
void launch_source_domain_rows_pdl_secondary(
    const void* domain_ptr, int descriptor_index, int rank, int active_rows,
    int device, cudaStream_t stream);

}  // namespace naivert_expert_down_domain
