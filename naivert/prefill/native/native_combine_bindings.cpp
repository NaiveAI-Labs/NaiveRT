#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

namespace prefill_combine {
void launch(const void*, const long long*, const long long*, int, const int*, void* const*, int,
            const long long*, const float*, void*, const void*, int*, cudaStream_t);
size_t mailbox_bytes();
}

void run(at::Tensor input, at::Tensor source_rows, at::Tensor destinations,
         std::vector<at::Tensor> mailboxes, int rank, at::Tensor ids,
         at::Tensor probabilities, at::Tensor output, at::Tensor epoch, at::Tensor error,
         c10::optional<at::Tensor> rows_dev) {
  TORCH_CHECK(rank >= 0 && rank < 8 && mailboxes.size() == 8, "eight ranks required");
  const int device = input.get_device();
  TORCH_CHECK(device == rank, "rank/device mismatch");
  for (const auto& t : {input, source_rows, destinations, ids, probabilities, output, epoch, error})
    TORCH_CHECK(t.is_cuda() && t.get_device() == device && t.is_contiguous(), "contiguous rank-local tensors required");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.numel() % 4096 == 0, "BF16 source rows required");
  TORCH_CHECK(source_rows.scalar_type() == at::kLong && destinations.scalar_type() == at::kLong &&
              source_rows.numel() == destinations.numel(), "INT64 row/destination lists required");
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.numel() == 512 * 8, "INT64 local top8 required");
  TORCH_CHECK(probabilities.scalar_type() == at::kFloat && probabilities.numel() == 512 * 8, "FP32 local probabilities required");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 && output.numel() == 512 * 4096, "BF16 local output required");
  TORCH_CHECK(epoch.scalar_type() == at::kLong && epoch.numel() == 1, "INT64 epoch required");
  TORCH_CHECK(error.scalar_type() == at::kInt && error.numel() == 1, "INT32 error required");
  void* peers[8];
  for (int peer = 0; peer < 8; ++peer) {
    TORCH_CHECK(mailboxes[peer].is_cuda() && mailboxes[peer].get_device() == peer &&
                mailboxes[peer].scalar_type() == at::kByte && mailboxes[peer].is_contiguous() &&
                size_t(mailboxes[peer].numel()) >= prefill_combine::mailbox_bytes(), "mailbox allocation mismatch");
    peers[peer] = mailboxes[peer].data_ptr();
  }
  const int* rows_ptr = nullptr;
  if (rows_dev.has_value()) {
    TORCH_CHECK(rows_dev->is_cuda() && rows_dev->get_device() == device && rows_dev->scalar_type() == at::kInt &&
                rows_dev->numel() == 1, "INT32 device row count required");
    rows_ptr = rows_dev->data_ptr<int>();
  }
  c10::cuda::CUDAGuard guard(device);
  prefill_combine::launch(input.data_ptr(), reinterpret_cast<const long long*>(source_rows.data_ptr<int64_t>()),
      reinterpret_cast<const long long*>(destinations.data_ptr<int64_t>()), int(source_rows.numel()), rows_ptr, peers, rank,
      reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), probabilities.data_ptr<float>(), output.data_ptr(),
      epoch.data_ptr(), error.data_ptr<int>(), at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("input"), py::arg("source_rows"), py::arg("destinations"), py::arg("mailboxes"),
        py::arg("rank"), py::arg("ids"), py::arg("probabilities"), py::arg("output"), py::arg("epoch"), py::arg("error"),
        py::arg("rows_dev") = py::none());
  m.def("mailbox_bytes", &prefill_combine::mailbox_bytes);
}
