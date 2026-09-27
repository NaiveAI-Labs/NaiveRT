// ExpertDown all-reduce entry: the production instantiations of the direct-run body.
//
// Every kernel takes the fixed (domain, descriptor index, rank) triple that the
// captured graphs replay; the per-layer arguments live in constant memory
// (`g_stable_expert_down_args`), indexed by the descriptor. One-row kernels
// launch one CTA per SM; multi-row kernels launch CTAs x active_rows and offset one row per
// blockIdx.y. The profiled one-row kernel is selected at capture time by
// NAIVERT_MOE_BLOCK_WAIT_POLL_METRICS=1 and records mailbox wait metrics when
// the domain enables them (see tests/correctness/mailbox_wait_probe.py).
#include "naivert/body/expert_down_allreduce/expert_down_allreduce_body.cuh"
#include "naivert/entry/expert_down_allreduce_entry.cuh"
#include "naivert/entry/expert_down_runtime_domain.hpp"
#include "naivert/pdl/grid_dependency.cuh"

#include <cstddef>
#include <cstdlib>

namespace naivert_replica_expert_down_allreduce {

// Native bulk staging needs one helper warp. Keep common role types identical
// across translation units and trim unused helper warps at the launch sites.
// This body supplies counted producer arrivals and uses no helper-group barrier.
constexpr int kLaunchThreads = kW8A8 ? 288 : kBlockThreads;
static_assert(!kW8A8 || (kConsumerThreads == 256 &&
    slot_ring::StandardRoleLayout::kHelperBaseThread == 256 &&
    kLaunchThreads == kConsumerThreads + 32));

namespace {

static_assert(kExpertDownOutputsPerTile == kOutputsPerTile);

constexpr int kNativeLaunchCtas = kW8A8 ? kTiles : kExpertDownLaunchCtas;

__device__ __constant__ ExpertDownAllreduceArgs
    g_stable_expert_down_args[kExpertDownDescriptorCapacity];
static_assert(
    sizeof(g_stable_expert_down_args) <= 64 * 1024,
    "production ExpertDown descriptor table exceeds constant memory");

// Multi-row kernels: shift every per-row pointer of the descriptor by
// blockIdx.y rows; each row owns one 0x70000 slab of the mailbox allocation.
__device__ __forceinline__ ExpertDownAllreduceArgs row_arguments(
    const ExpertDownRuntimeDomain* domains, std::uint32_t descriptor_index) {
  ExpertDownAllreduceArgs row_args =
      g_stable_expert_down_args[descriptor_index];
  const int row = static_cast<int>(blockIdx.y);
  // One up/gate row per slot (shared + selected experts) per query row.
  row_args.up_gate += row * kOutputSlots * row_args.moe_inter_dim;
  row_args.sel_probs += row * row_args.n_activated;
  row_args.sel_indices += row * row_args.n_activated;
  row_args.residual += row * row_args.dim;
  row_args.exp_out += row * row_args.dim;
  row_args.tp_result += row * row_args.dim;
  row_args.stable_domain = domains + row;
  return row_args;
}

template <typename Kernel>
void configure_once(Kernel kernel, bool* configured, int device) {
  if (configured[device]) {
    return;
  }
  cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  configured[device] = true;
}

}  // namespace

__global__ __launch_bounds__(kLaunchThreads, 1) void
naivert_expert_down_allreduce_rows1_kernel(
    const ExpertDownRuntimeDomain* domain, std::uint32_t descriptor_index,
    int rank) {
  extern __shared__ __align__(128) unsigned char shared[];
  (void)domain;
  (void)rank;
  ExpertDownAllreduceDecodeBody::run(
      g_stable_expert_down_args[descriptor_index], shared);
}

__global__ __launch_bounds__(kLaunchThreads, 1) void
naivert_expert_down_allreduce_rows1_pdl_producer_kernel(
    const ExpertDownRuntimeDomain* domain, std::uint32_t descriptor_index,
    int rank) {
  extern __shared__ __align__(128) unsigned char shared[];
  (void)domain;
  (void)rank;
  // Head helpers may stage independent LM-head pages as soon as all 148
  // ExpertDown CTAs are resident. Head consumers still acquire the
  // programmatic dependency before their first exp_out read.
  naivert::pdl::trigger_dependents<true>();
  ExpertDownAllreduceDecodeBody::run(
      g_stable_expert_down_args[descriptor_index], shared);
}

__global__ __launch_bounds__(kLaunchThreads, 4) void
naivert_expert_down_allreduce_rows_kernel(
    const ExpertDownRuntimeDomain* domains,
    std::uint32_t descriptor_index,
    int rank) {
  extern __shared__ __align__(128) unsigned char shared[];
  (void)rank;
  ExpertDownAllreduceBody::run(
      row_arguments(domains, descriptor_index), shared);
}

__global__ __launch_bounds__(kLaunchThreads, 4) void
naivert_expert_down_allreduce_rows_pdl_secondary_kernel(
    const ExpertDownRuntimeDomain* domains,
    std::uint32_t descriptor_index,
    int rank) {
  extern __shared__ __align__(128) unsigned char shared[];
  (void)rank;
  if constexpr (kW8A8) {
    naivert::pdl::trigger_dependents<true>();
  }
  ExpertDownAllreducePdlSecondaryBody::run(
      row_arguments(domains, descriptor_index), shared);
}

__global__ __launch_bounds__(kLaunchThreads, 1) void
naivert_expert_down_allreduce_rows1_profiled_kernel(
    const ExpertDownRuntimeDomain* domain, std::uint32_t descriptor_index,
    int rank) {
  extern __shared__ __align__(128) unsigned char shared[];
  (void)domain;
  (void)rank;
  if (descriptor_index >= kExpertDownDescriptorCapacity) {
    return;
  }
  ExpertDownAllreduceProfiledBody::run(
      g_stable_expert_down_args[descriptor_index], shared);
}

void launch_expert_down_allreduce_with_source_domain(
    const void* domain, int descriptor_index, int tp_rank, int device,
    cudaStream_t stream) {
  if (descriptor_index < 0 ||
      descriptor_index >= kExpertDownDescriptorCapacity) {
    return;
  }
  const auto* typed_domain =
      reinterpret_cast<const ExpertDownRuntimeDomain*>(domain);
  const auto typed_descriptor_index =
      static_cast<std::uint32_t>(descriptor_index);
  const char* metrics_env =
      std::getenv("NAIVERT_MOE_BLOCK_WAIT_POLL_METRICS");
  if (metrics_env != nullptr && std::atoi(metrics_env) != 0) {
    static bool configured[64] = {};
    configure_once(
        naivert_expert_down_allreduce_rows1_profiled_kernel, configured,
        device);
    naivert_expert_down_allreduce_rows1_profiled_kernel
        <<<kExpertDownLaunchCtas, kLaunchThreads, kSharedBytes, stream>>>(
            typed_domain, typed_descriptor_index, tp_rank);
    return;
  }
  static bool configured[64] = {};
  configure_once(naivert_expert_down_allreduce_rows1_kernel, configured,
                 device);
  naivert_expert_down_allreduce_rows1_kernel
      <<<kExpertDownLaunchCtas, kLaunchThreads, kSharedBytes, stream>>>(
          typed_domain, typed_descriptor_index, tp_rank);
}

void launch_expert_down_allreduce_with_source_domain_pdl_producer(
    const void* domain, int descriptor_index, int tp_rank, int device,
    cudaStream_t stream) {
  if (descriptor_index < 0 ||
      descriptor_index >= kExpertDownDescriptorCapacity) {
    return;
  }
  static bool configured[64] = {};
  configure_once(
      naivert_expert_down_allreduce_rows1_pdl_producer_kernel, configured,
      device);
  naivert_expert_down_allreduce_rows1_pdl_producer_kernel
      <<<kExpertDownLaunchCtas, kLaunchThreads, kSharedBytes, stream>>>(
          reinterpret_cast<const ExpertDownRuntimeDomain*>(domain),
          static_cast<std::uint32_t>(descriptor_index), tp_rank);
}

void launch_expert_down_allreduce_with_source_domain_rows(
    const void* domain,
    int descriptor_index,
    int tp_rank,
    int active_rows,
    int device,
    cudaStream_t stream) {
  if (active_rows == 1) {
    launch_expert_down_allreduce_with_source_domain(
        domain, descriptor_index, tp_rank, device, stream);
    return;
  }
  // 2..8 rows: one CTA row (blockIdx.y) and one mailbox slab per query row.
  if (active_rows < 2 || active_rows > 8 || descriptor_index < 0 ||
      descriptor_index >= kExpertDownDescriptorCapacity) {
    return;
  }
  static bool configured[64] = {};
  configure_once(naivert_expert_down_allreduce_rows_kernel, configured,
                 device);
  naivert_expert_down_allreduce_rows_kernel
      <<<dim3(kNativeLaunchCtas, active_rows, 1), kLaunchThreads, kSharedBytes, stream>>>(
          reinterpret_cast<const ExpertDownRuntimeDomain*>(domain),
          static_cast<std::uint32_t>(descriptor_index), tp_rank);
}

void launch_expert_down_allreduce_with_source_domain_rows_pdl_secondary(
    const void* domain,
    int descriptor_index,
    int tp_rank,
    int active_rows,
    int device,
    cudaStream_t stream) {
  // 2..8 rows: one CTA row (blockIdx.y) and one mailbox slab per query row.
  if (active_rows < 2 || active_rows > 8 || descriptor_index < 0 ||
      descriptor_index >= kExpertDownDescriptorCapacity) {
    return;
  }
  static bool configured[64] = {};
  configure_once(
      naivert_expert_down_allreduce_rows_pdl_secondary_kernel, configured,
      device);
  naivert_expert_down_allreduce_rows_pdl_secondary_kernel
      <<<dim3(kNativeLaunchCtas, active_rows, 1), kLaunchThreads, kSharedBytes, stream>>>(
          reinterpret_cast<const ExpertDownRuntimeDomain*>(domain),
          static_cast<std::uint32_t>(descriptor_index), tp_rank);
}

cudaError_t initialize_expert_down_allreduce_stable_descriptor(
    const void* host_descriptor, int descriptor_index, int device,
    cudaStream_t stream) {
  if (host_descriptor == nullptr || descriptor_index < 0 ||
      descriptor_index >= kExpertDownDescriptorCapacity) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  const auto* descriptor =
      reinterpret_cast<const ExpertDownDescriptor*>(host_descriptor);
  return cudaMemcpyToSymbolAsync(
      g_stable_expert_down_args, &descriptor->args,
      sizeof(ExpertDownAllreduceArgs),
      std::size_t(descriptor_index) * sizeof(ExpertDownAllreduceArgs),
      cudaMemcpyHostToDevice, stream);
}

}  // namespace naivert_replica_expert_down_allreduce
