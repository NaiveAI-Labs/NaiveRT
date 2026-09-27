// Model-operation extension bindings.
#include "naivert/entry/expert_down_domain.hpp"
#include "naivert/common/expert_activation.hpp"
#include "naivert/common/model_geometry.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "naivert/common/prepared_mlp_input_layout.hpp"
#include "naivert/kernels/model_ops/extension/gen_rmsnorm_expert_proj_prepared_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_rmsnorm_expert_proj_prepared_packed_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_expert_select_up_gate_silu_rows8_prepared_bindings.hpp"
#include "naivert/entry/tp8_argmax_rows_launch.hpp"
#include "naivert/kernels/model_ops/extension/gen_expert_select_up_gate_silu_rows8_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_rmsnorm_expert_proj_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_rmsnorm_head_proj_bindings.hpp"
#include "naivert/entry/unproj_o_allreduce_accepted_row.hpp"
#include "naivert/kernels/model_ops/extension/gen_unproj_o_allreduce_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_rmsnorm_up_gate_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_down_allreduce_bindings.hpp"
#include "naivert/kernels/model_ops/extension/gen_expert_select_up_gate_silu_rows1_bindings.hpp"

namespace naivert::checkpoint_bf16 { void bind(pybind11::module_& m); }
namespace naivert::embed_gather { void bind(pybind11::module_& m); }

namespace naivert_naive_n0p5_model_ops {

// The ExpertDown domain writer is shared with the MTP draft-ops extension
// (naivert/entry/expert_down_domain.cu); these keep the exported binding names.
inline std::size_t expert_down_domain_size() {
  return naivert_expert_down_domain::domain_size();
}
inline int expert_down_descriptor_capacity() {
  return naivert_expert_down_domain::descriptor_capacity();
}

namespace {

void check_cuda_contiguous(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

int cuda_device_index(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "expected CUDA tensor");
  return static_cast<int>(t.get_device());
}

void enable_tp8_peer_access() {
  int restore_device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&restore_device));
  for (int rank = 0; rank < 8; ++rank) {
    C10_CUDA_CHECK(cudaSetDevice(rank));
    for (int peer = 0; peer < 8; ++peer) {
      if (peer == rank) {
        continue;
      }
      const cudaError_t status =
          cudaDeviceEnablePeerAccess(peer, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
      } else {
        C10_CUDA_CHECK(status);
      }
    }
  }
  C10_CUDA_CHECK(cudaSetDevice(restore_device));
}

void launch_unproj_allreduce_accepted_row(
    at::Tensor proj_o_rows, at::Tensor weight, at::Tensor scales,
    at::Tensor residual_rows, at::Tensor accepted_count,
    at::Tensor output, at::Tensor replay_iteration,
    const std::vector<at::Tensor>& mailbox, int64_t rank,
    int64_t mailbox_tag) {
  constexpr int kRanks = 8;
  TORCH_CHECK(rank >= 0 && rank < kRanks,
              "rank must be in [0, 8)");
  TORCH_CHECK(mailbox_tag > 0 &&
                  mailbox_tag <= std::numeric_limits<std::uint32_t>::max(),
              "mailbox_tag must be a nonzero uint32");
  TORCH_CHECK(mailbox.size() == kRanks,
              "selected UnprojO requires eight destination mailboxes");
  for (const auto& item : {
           proj_o_rows, weight, scales, residual_rows, accepted_count,
           output, replay_iteration}) {
    check_cuda_contiguous(item, "selected UnprojO tensor");
  }
  const int device = cuda_device_index(proj_o_rows);
  TORCH_CHECK(
      device == rank &&
          cuda_device_index(weight) == device &&
          cuda_device_index(scales) == device &&
          cuda_device_index(residual_rows) == device &&
          cuda_device_index(accepted_count) == device &&
          cuda_device_index(output) == device &&
          cuda_device_index(replay_iteration) == device,
      "selected UnprojO tensors must share their rank device");
  TORCH_CHECK(
      proj_o_rows.scalar_type() == at::kBFloat16 &&
          proj_o_rows.numel() >= 4 * 2048 &&
          residual_rows.scalar_type() == at::kBFloat16 &&
          residual_rows.numel() >= 4 * naivert::geometry::kHidden &&
          accepted_count.scalar_type() == at::kInt &&
          accepted_count.numel() >= 1 &&
          output.scalar_type() == at::kBFloat16 &&
          output.numel() >= naivert::geometry::kHidden,
      "selected UnprojO candidate rows are incomplete");
  TORCH_CHECK(
      replay_iteration.scalar_type() == at::kByte &&
          replay_iteration.nbytes() >= sizeof(std::uint64_t),
      "selected UnprojO replay iteration must cover uint64");
  std::array<const void*, kRanks> mailbox_ptrs{};
  for (int peer = 0; peer < kRanks; ++peer) {
    check_cuda_contiguous(mailbox[peer], "mailbox");
    TORCH_CHECK(
        cuda_device_index(mailbox[peer]) == peer &&
            mailbox[peer].nbytes() >= 0x70000,
        "selected UnprojO destination mailbox is invalid");
    mailbox_ptrs[peer] = mailbox[peer].data_ptr();
  }
  c10::cuda::CUDAGuard guard(proj_o_rows.device());
  C10_CUDA_CHECK(
      naivert_replica_unproj_o_allreduce::
          launch_unproj_o_allreduce_selected_rank(
              proj_o_rows.data_ptr(), weight.data_ptr(),
              scales.data_ptr(), residual_rows.data_ptr(),
              reinterpret_cast<const std::uint32_t*>(
                  accepted_count.data_ptr<std::int32_t>()),
              output.data_ptr(), mailbox_ptrs.data(),
              static_cast<int>(rank),
              static_cast<std::uint32_t>(mailbox_tag),
              reinterpret_cast<const std::uint64_t*>(
                  replay_iteration.data_ptr()),
              device,
              at::cuda::getCurrentCUDAStream(device).stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_expert_down_replay_domain_impl(
    at::Tensor domain, at::Tensor up_gate, at::Tensor w_down,
    at::Tensor w_down_scale, at::Tensor sel_probs,
    at::Tensor sel_indices, at::Tensor exp_out, at::Tensor residual,
    const std::vector<at::Tensor>& mailbox, int64_t mailbox_tag,
    int64_t descriptor_index, int64_t rank,
    const std::uint64_t* replay_iteration) {
  check_cuda_contiguous(domain, "domain");
  TORCH_CHECK(domain.scalar_type() == at::kByte,
              "domain must be uint8");
  TORCH_CHECK(
      domain.nbytes() >= expert_down_domain_size(),
      "domain tensor is smaller than ExpertDownRuntimeDomain");
  check_cuda_contiguous(up_gate, "up_gate");
  TORCH_CHECK(up_gate.scalar_type() ==
                  (naivert::kExpertActivationFp32 ? at::kFloat : at::kBFloat16),
              "up_gate has the wrong dtype for this build");
  TORCH_CHECK(up_gate.numel() >= naivert_expert_down_domain::activation_row_elements(),
              "up_gate is too small for one token");
  check_cuda_contiguous(w_down, "w_down");
  check_cuda_contiguous(w_down_scale, "w_down_scale");
  TORCH_CHECK(w_down_scale.scalar_type() ==
      (naivert::kExpertActivationFp32 ? at::kFloat : at::kBFloat16),
      "w_down_scale has the wrong dtype for this build");
  check_cuda_contiguous(sel_probs, "sel_probs");
  check_cuda_contiguous(sel_indices, "sel_indices");
  check_cuda_contiguous(exp_out, "exp_out");
  check_cuda_contiguous(residual, "residual");
  TORCH_CHECK(mailbox.size() == 8,
              "ExpertDown source ABI requires eight mailbox tensors");
  TORCH_CHECK(
      mailbox_tag > 0 &&
          mailbox_tag <= std::numeric_limits<std::uint32_t>::max(),
      "ExpertDown source mailbox tag must be a nonzero uint32");
  TORCH_CHECK(
      descriptor_index >= 0 &&
          descriptor_index < expert_down_descriptor_capacity(),
      "ExpertDown descriptor index exceeds the constant table");
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be in [0, 8)");
  c10::cuda::CUDAGuard guard(domain.device());
  const int device = cuda_device_index(domain);
  TORCH_CHECK(device == rank, "domain device must match rank");
  TORCH_CHECK(
      cuda_device_index(up_gate) == device &&
          cuda_device_index(w_down) == device &&
          cuda_device_index(w_down_scale) == device &&
          cuda_device_index(sel_probs) == device &&
          cuda_device_index(sel_indices) == device &&
          cuda_device_index(exp_out) == device &&
          cuda_device_index(residual) == device,
      "ExpertDown source tensors must reside on their rank device");
  std::array<const void*, 8> mailbox_ptrs{};
  for (int peer = 0; peer < 8; ++peer) {
    check_cuda_contiguous(mailbox[peer], "mailbox");
    TORCH_CHECK(
        cuda_device_index(mailbox[peer]) == peer,
        "ExpertDown mailbox destination must reside on its rank");
    TORCH_CHECK(
        mailbox[peer].nbytes() >= naivert_expert_down_domain::mailbox_row_bytes(),
        "ExpertDown source mailbox allocation is too small");
    mailbox_ptrs[peer] = mailbox[peer].data_ptr();
  }
  C10_CUDA_CHECK(naivert_expert_down_domain::prepare_source_domain(
      domain.data_ptr(), domain.nbytes(), up_gate.data_ptr(),
      w_down.data_ptr(), w_down_scale.data_ptr(),
      sel_probs.data_ptr<float>(), sel_indices.data_ptr<int>(),
      exp_out.data_ptr(), residual.data_ptr(), mailbox_ptrs.data(),
      replay_iteration, static_cast<std::uint32_t>(mailbox_tag),
      static_cast<int>(descriptor_index),
      static_cast<int>(rank), device,
      at::cuda::getCurrentCUDAStream(device).stream()));
}

// Tag-only preparation is used by isolated kernel checks. Serving uses the
// epoch variant below so repeated layers advance the mailbox generation.
void prepare_expert_down_replay_domain(
    at::Tensor domain, at::Tensor up_gate, at::Tensor w_down,
    at::Tensor w_down_scale, at::Tensor sel_probs,
    at::Tensor sel_indices, at::Tensor exp_out, at::Tensor residual,
    const std::vector<at::Tensor>& mailbox, int64_t mailbox_tag,
    int64_t descriptor_index, int64_t rank) {
  prepare_expert_down_replay_domain_impl(
      domain, up_gate, w_down, w_down_scale, sel_probs, sel_indices, exp_out,
      residual, mailbox, mailbox_tag, descriptor_index, rank, nullptr);
}

void prepare_expert_down_replay_domain_epoch(
    at::Tensor domain, at::Tensor up_gate, at::Tensor w_down,
    at::Tensor w_down_scale, at::Tensor sel_probs,
    at::Tensor sel_indices, at::Tensor exp_out, at::Tensor residual,
    const std::vector<at::Tensor>& mailbox, int64_t mailbox_tag,
    int64_t descriptor_index, int64_t rank, at::Tensor replay_iteration) {
  check_cuda_contiguous(replay_iteration, "replay_iteration");
  TORCH_CHECK(replay_iteration.nbytes() >= sizeof(std::uint64_t),
              "replay_iteration must hold one uint64");
  TORCH_CHECK(cuda_device_index(replay_iteration) == cuda_device_index(domain),
              "replay_iteration must live on the domain's device");
  prepare_expert_down_replay_domain_impl(
      domain, up_gate, w_down, w_down_scale, sel_probs, sel_indices, exp_out,
      residual, mailbox, mailbox_tag, descriptor_index, rank,
      reinterpret_cast<const std::uint64_t*>(replay_iteration.data_ptr()));
}

void launch_expert_down_from_replay_domain(
    at::Tensor domain, int64_t descriptor_index, int64_t rank) {
  check_cuda_contiguous(domain, "domain");
  TORCH_CHECK(domain.scalar_type() == at::kByte,
              "domain must be uint8");
  TORCH_CHECK(
      domain.nbytes() >= expert_down_domain_size(),
      "domain tensor is smaller than ExpertDownRuntimeDomain");
  TORCH_CHECK(
      descriptor_index >= 0 &&
          descriptor_index < expert_down_descriptor_capacity(),
      "ExpertDown descriptor index exceeds the constant table");
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be in [0, 8)");
  c10::cuda::CUDAGuard guard(domain.device());
  const int device = cuda_device_index(domain);
  TORCH_CHECK(device == rank, "domain device must match rank");
  naivert_expert_down_domain::launch_source_domain(
      domain.data_ptr(), static_cast<int>(descriptor_index),
      static_cast<int>(rank), device,
      at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_expert_down_rows_from_replay_domain(
    at::Tensor domains, int64_t descriptor_index, int64_t rank,
    int64_t active_rows) {
  check_cuda_contiguous(domains, "domains");
  TORCH_CHECK(domains.scalar_type() == at::kByte,
              "domains must be uint8");
  TORCH_CHECK(active_rows >= 1 && active_rows <= 8,
              "ExpertDown supports 1 to 8 active rows (one mailbox slab per row)");
  TORCH_CHECK(
      domains.nbytes() >=
          active_rows * static_cast<int64_t>(expert_down_domain_size()),
      "domains tensor does not cover all active rows");
  TORCH_CHECK(
      descriptor_index >= 0 &&
          descriptor_index < expert_down_descriptor_capacity(),
      "ExpertDown descriptor index exceeds the constant table");
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be in [0, 8)");
  c10::cuda::CUDAGuard guard(domains.device());
  const int device = cuda_device_index(domains);
  TORCH_CHECK(device == rank, "domains device must match rank");
  naivert_expert_down_domain::launch_source_domain_rows(
      domains.data_ptr(), static_cast<int>(descriptor_index),
      static_cast<int>(rank), static_cast<int>(active_rows), device,
      at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_expert_down_rows_from_replay_domain_pdl_consumer(
    at::Tensor domains, int64_t descriptor_index, int64_t rank,
    int64_t active_rows) {
  check_cuda_contiguous(domains, "domains");
  TORCH_CHECK(domains.scalar_type() == at::kByte,
              "domains must be uint8");
  TORCH_CHECK(active_rows >= 2 && active_rows <= 8,
              "PDL ExpertDown supports 2 to 8 active rows (one mailbox slab per row)");
  TORCH_CHECK(
      domains.nbytes() >=
          active_rows * static_cast<int64_t>(expert_down_domain_size()),
      "domains tensor does not cover all active rows");
  TORCH_CHECK(
      descriptor_index >= 0 &&
          descriptor_index < expert_down_descriptor_capacity(),
      "ExpertDown descriptor index exceeds the constant table");
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be in [0, 8)");
  c10::cuda::CUDAGuard guard(domains.device());
  const int device = cuda_device_index(domains);
  TORCH_CHECK(device == rank, "domains device must match rank");
  naivert_expert_down_domain::launch_source_domain_rows_pdl_secondary(
      domains.data_ptr(), static_cast<int>(descriptor_index),
      static_cast<int>(rank), static_cast<int>(active_rows), device,
      at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tp8_argmax_rows(
    at::Tensor logits, at::Tensor token_out,
    at::Tensor top_p_scores,
    const std::vector<at::Tensor>& sampler_mailboxes,
    int64_t rank) {
  const int64_t rows = token_out.numel();
  TORCH_CHECK(rows >= 1 && rows <= 8, "ExecuteTop1 supports 1..8 rows");
  constexpr int64_t kMailboxBytes = 128;
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be in [0,8)");
  check_cuda_contiguous(logits, "logits");
  check_cuda_contiguous(token_out, "token_out");
  check_cuda_contiguous(top_p_scores, "top_p_scores");
  const int device = cuda_device_index(token_out);
  TORCH_CHECK(
      device == rank &&
          cuda_device_index(logits) == device &&
          cuda_device_index(top_p_scores) == device,
      "ExecuteTop1 rows output device must match rank");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat &&
          logits.numel() >= rows * naivert::geometry::kVocabShard &&
          token_out.scalar_type() == at::kInt &&
          top_p_scores.scalar_type() == at::kFloat &&
          top_p_scores.numel() >= rows,
      "ExecuteTop1 rows outputs must cover the requested rows");
  TORCH_CHECK(
      sampler_mailboxes.size() == 8,
      "ExecuteTop1 rows requires eight mailboxes");
  naivert_tp8_argmax_rows::Tp8ArgmaxRowsArgs args{};
  args.logits = logits.data_ptr<float>();
  args.token_out = token_out.data_ptr<int>();
  args.top_p_scores = top_p_scores.data_ptr<float>();
  args.rows = static_cast<int>(rows);
  args.vocab = naivert::geometry::kVocabShard;
  args.rank = static_cast<int>(rank);
  for (int peer = 0; peer < 8; ++peer) {
    check_cuda_contiguous(
        sampler_mailboxes[peer], "sampler_mailbox");
    TORCH_CHECK(
        cuda_device_index(sampler_mailboxes[peer]) == peer &&
            sampler_mailboxes[peer].nbytes() >=
                rows * kMailboxBytes,
        "ExecuteTop1 rows mailbox must cover the requested rank-local rows");
    args.mailbox_peers[peer] =
        reinterpret_cast<const std::uint8_t*>(
            sampler_mailboxes[peer].data_ptr());
  }
  c10::cuda::CUDAGuard guard(token_out.device());
  naivert_tp8_argmax_rows::launch_tp8_argmax_rows(
      args, at::cuda::getCurrentCUDAStream(device).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  naivert::checkpoint_bf16::bind(m);
  naivert::embed_gather::bind(m);
  m.def("enable_tp8_peer_access", &enable_tp8_peer_access);
  m.def("launch_rmsnorm_up_gate", &launch_rmsnorm_up_gate);
  m.def("launch_rmsnorm_up_gate_pdl_consumer", &launch_rmsnorm_up_gate_pdl_consumer);
  m.def("launch_down_allreduce", &launch_down_allreduce);
  m.def("launch_down_allreduce_pdl_consumer", &launch_down_allreduce_pdl_consumer);
  m.def("launch_unproj_allreduce", &launch_unproj_allreduce);
  m.def(
      "launch_unproj_allreduce_replay",
      &launch_unproj_allreduce_replay);
  m.def(
      "launch_unproj_allreduce_pdl_consumer",
      &launch_unproj_allreduce_pdl_consumer);
  m.def(
      "launch_unproj_allreduce_replay_pdl_consumer",
      &launch_unproj_allreduce_replay_pdl_consumer);
  m.def(
      "launch_unproj_allreduce_accepted_row",
      &launch_unproj_allreduce_accepted_row);
  m.def("expert_down_domain_size", &expert_down_domain_size);
  m.def("expert_activation_is_fp32", [] { return naivert::kExpertActivationFp32; });
  m.def("expert_weight_scales_are_fp32", [] { return naivert::kExpertActivationFp32; });
  m.def("expert_down_mailbox_row_bytes", &naivert_expert_down_domain::mailbox_row_bytes);
  m.def(
      "expert_down_descriptor_capacity",
      &expert_down_descriptor_capacity);
  m.def(
      "prepare_expert_down_replay_domain",
      &prepare_expert_down_replay_domain);
  m.def(
      "prepare_expert_down_replay_domain_epoch",
      &prepare_expert_down_replay_domain_epoch);
  m.def(
      "launch_expert_down_from_replay_domain",
      &launch_expert_down_from_replay_domain);
  m.def(
      "launch_expert_down_rows_from_replay_domain",
      &launch_expert_down_rows_from_replay_domain);
  m.def(
      "launch_expert_down_rows_from_replay_domain_pdl_consumer",
      &launch_expert_down_rows_from_replay_domain_pdl_consumer);
  m.def("prepared_mlp_input_bytes", []() { return naivert::prepared_mlp::kBytes; });
  m.def("launch_rmsnorm_expert_proj_prepared_packed", &launch_rmsnorm_expert_proj_prepared_packed);
  m.def("launch_rmsnorm_expert_proj_prepared_packed_pdl_producer", &launch_rmsnorm_expert_proj_prepared_packed_pdl_producer);
  m.def("launch_rmsnorm_expert_proj_prepared_packed_only", &launch_rmsnorm_expert_proj_prepared_packed_only);
  m.def("launch_rmsnorm_expert_proj_prepared_packed_only_pdl_producer", &launch_rmsnorm_expert_proj_prepared_packed_only_pdl_producer);
  m.def("launch_rmsnorm_expert_proj_prepared", &launch_rmsnorm_expert_proj_prepared);
  m.def("launch_rmsnorm_expert_proj_prepared_only", &launch_rmsnorm_expert_proj_prepared_only);
  m.def("launch_rmsnorm_expert_proj_prepared_only_pdl_producer", &launch_rmsnorm_expert_proj_prepared_only_pdl_producer);
  m.def("launch_expert_select_up_gate_silu_rows8_prepared", &launch_expert_select_up_gate_silu_rows8_prepared);
  m.def("launch_rmsnorm_expert_proj_prepared_pdl_producer", &launch_rmsnorm_expert_proj_prepared_pdl_producer);
  m.def("launch_expert_select_up_gate_silu_rows8_prepared_pdl_producer", &launch_expert_select_up_gate_silu_rows8_prepared_pdl_producer);
  m.def("launch_rmsnorm_expert_proj", &launch_rmsnorm_expert_proj);
  m.def(
      "launch_rmsnorm_expert_proj_pdl_producer",
      &launch_rmsnorm_expert_proj_pdl_producer);
  m.def(
      "launch_expert_select_up_gate_silu_rows8",
      &launch_expert_select_up_gate_silu_rows8);
  m.def(
      "launch_expert_select_up_gate_silu_rows8_pdl_producer",
      &launch_expert_select_up_gate_silu_rows8_pdl_producer);
  m.def(
      "launch_expert_select_up_gate_silu_rows1",
      &launch_expert_select_up_gate_silu_rows1);
  m.def(
      "launch_expert_select_up_gate_silu_rows1_pdl_consumer",
      &launch_expert_select_up_gate_silu_rows1_pdl_consumer);
  m.def("launch_tp8_argmax_rows", &launch_tp8_argmax_rows);
  m.def(
      "launch_rmsnorm_head_proj_rows1",
      &launch_rmsnorm_head_proj_rows1);
  m.def(
      "launch_rmsnorm_head_proj_rows1_pdl_consumer",
      &launch_rmsnorm_head_proj_rows1_pdl_consumer);
}

}  // namespace naivert_naive_n0p5_model_ops
