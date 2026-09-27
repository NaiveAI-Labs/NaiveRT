#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>
#include "naivert/entry/dspark_mlp_launch.hpp"
#include "naivert/common/tp8_mailbox.cuh"

namespace {
namespace d4 = naivert_dspark_ops;
using G = d4::MlpGeometry;

void check(const at::Tensor& tensor, at::ScalarType dtype,
           int64_t elements, int device, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device &&
              tensor.scalar_type() == dtype && tensor.is_contiguous() &&
              tensor.numel() == elements, "invalid D4 tensor: ", name);
}

void configure(int device) {
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(d4::configure_mlp_rows8(device));
}

void mlp(at::Tensor input, at::Tensor gamma, at::Tensor gate_up,
         at::Tensor down, at::Tensor activation, at::Tensor partial,
         at::Tensor output, at::Tensor barrier,
         std::vector<at::Tensor> mailboxes, at::Tensor epoch,
         at::Tensor watchdog, int rank, int layer, uint64_t watchdog_ns,
         float epsilon) {
  TORCH_CHECK(input.is_cuda(), "D4 input must be CUDA");
  const int device = input.get_device();
  c10::cuda::CUDAGuard guard(device);
  TORCH_CHECK(rank >= 0 && rank < G::Ranks && mailboxes.size() == G::Ranks &&
              layer >= 0 && layer < 5 && epsilon > 0 && std::isfinite(epsilon),
              "invalid D4 rank/layer/epsilon");
  check(input, at::kBFloat16, G::Rows * G::Hidden, device, "input");
  check(gamma, at::kFloat, G::Hidden, device, "packed gamma");
  check(gate_up, at::kBFloat16, 2 * G::IntermediatePerRank * G::Hidden, device, "packed gate/up");
  check(down, at::kBFloat16, G::Hidden * G::IntermediatePerRank, device, "packed down");
  check(activation, at::kBFloat16, G::Rows * G::IntermediatePerRank, device, "activation");
  check(output, at::kBFloat16, G::Rows * G::Hidden, device, "output");
  check(barrier, at::kInt, 2, device, "grid barrier");
  check(epoch, at::kInt, 1, device, "epoch");
  check(watchdog, at::kInt, 1, device, "watchdog");
  check(partial, at::kBFloat16, NAIVERT_DSPARK_DIAGNOSTICS ? G::Rows * G::Hidden : 0,
        device, "diagnostic partial");
  TORCH_CHECK(input.data_ptr() != output.data_ptr(), "D4 input/output must not alias");
  d4::MlpArgs args{};
  args.hid_a = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  args.post_ln = gamma.data_ptr<float>();
  args.gate_w = reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr());
  args.up_w = args.gate_w + G::IntermediatePerRank * G::Hidden;
  args.down_w = reinterpret_cast<const __nv_bfloat16*>(down.data_ptr());
  args.mlp_h = reinterpret_cast<__nv_bfloat16*>(activation.data_ptr());
  args.down_partial = reinterpret_cast<__nv_bfloat16*>(partial.data_ptr());
  args.resid_out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  args.barrier = reinterpret_cast<unsigned*>(barrier.data_ptr<int32_t>());
  for (int r = 0; r < G::Ranks; ++r) {
    const auto& box = mailboxes[r];
    TORCH_CHECK(box.is_cuda() && box.is_contiguous() && box.scalar_type() == at::kByte &&
                box.numel() >= G::Rows * naivert::kMailboxSlabStride,
                "invalid D4 peer mailbox");
    if (r == rank) TORCH_CHECK(box.get_device() == device, "D4 local mailbox device mismatch");
    for (int p = 0; p < r; ++p)
      TORCH_CHECK(box.get_device() != mailboxes[p].get_device(), "D4 peers must use distinct GPUs");
    args.mailbox_peers[r] = box.data_ptr<uint8_t>();
  }
  args.replay_generation = epoch.data_ptr<int32_t>();
  args.generation_salt = 4 + 2 * layer;
  args.rank = rank;
  args.watchdog_limit_ns = watchdog_ns;
  args.watchdog_flag = reinterpret_cast<unsigned*>(watchdog.data_ptr<int32_t>());
  args.rms_epsilon = epsilon;
  C10_CUDA_CHECK(d4::launch_mlp_rows8(args, at::cuda::getCurrentCUDAStream(device)));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure", &configure);
  m.def("mlp", &mlp);
  m.def("launch_shape", [] { return std::vector<int>{d4::mlp_rows8_ctas(),
      d4::mlp_rows8_threads(), d4::mlp_rows8_shared_bytes()}; });
}
