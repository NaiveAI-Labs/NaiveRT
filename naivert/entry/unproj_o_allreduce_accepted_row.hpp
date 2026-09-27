#pragma once
// Hand-written UnprojO variant: device-side accepted-row selection (see
// unproj_o_allreduce_accepted_row_entry.cu).  The plain rows1 launches
// are generated from naivert/kernels/ops/unproj_o_allreduce.py.
#include <cuda_runtime_api.h>
#include <cstdint>

namespace naivert_replica_unproj_o_allreduce {

cudaError_t launch_unproj_o_allreduce_selected_rank(
    const void* proj_o_rows, const void* weight, const void* scales,
    const void* residual_rows, const std::uint32_t* accepted_count,
    void* output, const void* const mailbox_peers[8], int rank,
    std::uint32_t mailbox_tag, const std::uint64_t* replay_iteration,
    int device, cudaStream_t stream);

}  // namespace naivert_replica_unproj_o_allreduce
