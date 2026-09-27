#include "naivert/entry/dspark_tail_launch.hpp"
#include "naivert/body/dspark_tail/body.cuh"
#include <algorithm>

namespace naivert_dspark_tail {
__global__ void naivert_dspark_geometry_entry(const __grid_constant__ GeometryArgs a) {
  GeometryBody::run(a);
}
__global__ void naivert_dspark_accept_entry(const __grid_constant__ AcceptArgs a) {
  AcceptBody::run(a);
}
template<bool Staged>
__global__ void naivert_dspark_tail_entry(const __grid_constant__ EpilogueArgs a) {
  TailBody::run<Staged>(a);
}
template<bool Staged>
__global__ void naivert_dspark_tail_chained_entry(const __grid_constant__ EpilogueArgs a) {
  if (!TailBody::run<Staged>(a)) return;
  auto self = cudaGetCurrentGraphExec();
  if (self == nullptr || cudaGraphLaunch(self, cudaStreamGraphTailLaunch) != cudaSuccess)
    naivert_seq1_commit::fail_closed(a.commit.control,
        a.commit.control->flags | naivert_seq1_commit::kSeq1ChainLaunchFailed);
}
cudaError_t launch_geometry(const GeometryArgs& a, cudaStream_t stream) {
  const int count = std::max(kRows * kHidden, a.max_kv);
  naivert_dspark_geometry_entry<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(a);
  return cudaGetLastError();
}
cudaError_t launch_accept(const AcceptArgs& a, cudaStream_t stream) {
  naivert_dspark_accept_entry<<<kRows * kHidden / kThreads, kThreads, 0, stream>>>(a);
  return cudaGetLastError();
}
cudaError_t launch_tail(const EpilogueArgs& a, cudaStream_t stream, bool chain, bool staged) {
  const int blocks = (std::max(kRows * kHidden, a.g.max_kv) + kThreads - 1) / kThreads;
  if (staged) {
    if (chain) naivert_dspark_tail_chained_entry<true><<<blocks,kThreads,0,stream>>>(a);
    else naivert_dspark_tail_entry<true><<<blocks,kThreads,0,stream>>>(a);
  } else {
    if (chain) naivert_dspark_tail_chained_entry<false><<<blocks,kThreads,0,stream>>>(a);
    else naivert_dspark_tail_entry<false><<<blocks,kThreads,0,stream>>>(a);
  }
  return cudaGetLastError();
}
}  // namespace naivert_dspark_tail
