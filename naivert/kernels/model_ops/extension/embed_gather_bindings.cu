// Eight-row BF16 embedding gather for the verify graph's first node.
// torch.index_select's generic indexSelectSmallIndex kernel took 6.6 us for the 64 KB the
// verify-8 input needs; one CTA per row with 16-byte copies is one launch plus one HBM round trip.
// Output bytes are the selected rows verbatim, so the result is bitwise index_select's.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace naivert::embed_gather {
using Tensor = at::Tensor;
constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void gather_rows_kernel(
    const uint4* __restrict__ table, const std::int64_t* __restrict__ ids, uint4* __restrict__ out,
    std::int64_t rows_in_table, int vec_per_row) {
  std::int64_t id = ids[blockIdx.x];
  id = id < 0 ? 0 : (id >= rows_in_table ? rows_in_table - 1 : id);
  const uint4* src = table + id * vec_per_row;
  uint4* dst = out + std::int64_t(blockIdx.x) * vec_per_row;
  for (int v = int(threadIdx.x); v < vec_per_row; v += kThreads) dst[v] = src[v];
}

// out[i, :] = table[ids[i], :] for i < ids.numel(); table [vocab, hidden] bf16, hidden % 8 == 0.
void gather_rows_bf16(Tensor table, Tensor ids, Tensor out) {
  TORCH_CHECK(table.is_cuda() && table.dim() == 2 && table.is_contiguous() &&
              table.scalar_type() == at::kBFloat16, "table must be a contiguous CUDA bf16 matrix");
  TORCH_CHECK(ids.is_cuda() && ids.dim() == 1 && ids.is_contiguous() && ids.scalar_type() == at::kLong &&
              ids.get_device() == table.get_device(), "ids must be a contiguous CUDA int64 vector on the table's device");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
              out.get_device() == table.get_device() && out.size(0) == ids.numel() && out.size(1) == table.size(1),
              "out must be a contiguous CUDA bf16 [ids, hidden] matrix on the table's device");
  TORCH_CHECK(table.size(1) % 8 == 0 && ids.numel() >= 1 && ids.numel() <= 65535, "hidden must be a multiple of 8");
  const c10::cuda::CUDAGuard guard(table.device());
  auto stream = at::cuda::getCurrentCUDAStream(table.get_device());
  gather_rows_kernel<<<unsigned(ids.numel()), kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(table.data_ptr()), ids.data_ptr<std::int64_t>(),
      reinterpret_cast<uint4*>(out.data_ptr()), table.size(0), int(table.size(1) / 8));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bind(pybind11::module_& m) { m.def("gather_rows_bf16", &gather_rows_bf16); }
}  // namespace naivert::embed_gather
