#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <optional>
#include "naivert/entry/dspark_head_markov_launch.hpp"
#include "naivert/body/dspark_common/diagnostics.hpp"
namespace {
using naivert_dspark_ops::kExportDiagnostics;
void check(const at::Tensor& t, const char* name, at::ScalarType dtype, int64_t elements, int device) {
  TORCH_CHECK(t.is_cuda() && t.get_device() == device && t.is_contiguous() &&
      t.scalar_type() == dtype && t.numel() == elements, "invalid D5/D6 tensor: ", name);
}
template<class Args>
void bind_watchdog(Args& args, const at::Tensor& flag, int64_t limit, int device) {
  TORCH_CHECK(limit >= 0, "negative watchdog bound");
  check(flag, "watchdog", at::kInt, 1, device);
  args.watchdog_limit_ns = static_cast<unsigned long long>(limit);
  args.watchdog_flag = reinterpret_cast<unsigned*>(flag.data_ptr<int32_t>());
}
void configure(int device) {
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::configure_head_markov(device));
}
template<bool Fused>
naivert_dspark_ops::HeadArgs make_head_args(at::Tensor resid_in, at::Tensor norm_gamma, at::Tensor lm_head,
                at::Tensor norm_out, at::Tensor logits, at::Tensor watchdog_flag,
                int64_t watchdog_limit_ns, double rms_epsilon, int64_t first_logit_row) {
  TORCH_CHECK(resid_in.is_cuda(), "D5 input must be CUDA");
  const int device = resid_in.get_device();
  const int64_t window = 8, rows = 7, hidden = 4096;
  const int64_t vocab = naivert_dspark_ops::head_rows8_vocab();
  check(resid_in, "resid_in", at::kBFloat16, window * hidden, device);
  check(norm_gamma, "norm_gamma", at::kFloat, hidden, device);
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(norm_gamma.data_ptr()) % 16 == 0, "D5 gamma must be 16-byte aligned");
  check(lm_head, "lm_head", at::kBFloat16, vocab * hidden, device);
  TORCH_CHECK(lm_head.sizes().vec() == naivert_dspark_ops::head_rows8_weight_shape(),
              "lm_head must use the exported packed weight shape");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(lm_head.data_ptr()) % 16 == 0,
              "packed lm_head must be 16-byte aligned for TMA");
  check(norm_out, "norm_out", at::kBFloat16, kExportDiagnostics ? window * hidden : 0, device);
  check(logits, "logits", at::kFloat, !Fused || kExportDiagnostics ? rows * vocab : 0, device);
  TORCH_CHECK(norm_out.data_ptr() != resid_in.data_ptr(),
              "norm_out must not alias resid_in: every CTA re-reads resid_in for the row norm");

  naivert_dspark_ops::HeadArgs args{};
  // Initialize padding for deterministic raw argument comparisons.
  std::memset(&args, 0, sizeof(args));
  TORCH_CHECK(std::isfinite(rms_epsilon) && rms_epsilon > 0, "invalid RMS epsilon");
  args.rms_epsilon = static_cast<float>(rms_epsilon);
  TORCH_CHECK(first_logit_row == 0 || first_logit_row == 1, "first_logit_row must be 0 or 1");
  args.first_logit_row = static_cast<int>(first_logit_row);
  args.resid_in = reinterpret_cast<const __nv_bfloat16*>(resid_in.data_ptr());
  args.norm_gamma = norm_gamma.data_ptr<float>();
  args.lm_head = reinterpret_cast<const __nv_bfloat16*>(lm_head.data_ptr());
  args.norm_out = reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr());
  args.logits = logits.data_ptr<float>();
  bind_watchdog(args, watchdog_flag, watchdog_limit_ns, device);

  return args;
}
void head_rows8(at::Tensor resid_in, at::Tensor norm_gamma, at::Tensor lm_head,
                at::Tensor norm_out, at::Tensor logits, at::Tensor watchdog_flag,
                int64_t watchdog_limit_ns, double rms_epsilon, int64_t first_logit_row) {
  auto args = make_head_args<false>(resid_in, norm_gamma, lm_head, norm_out, logits, watchdog_flag, watchdog_limit_ns, rms_epsilon, first_logit_row);
  const int device = resid_in.get_device();
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::launch_head_rows8(args,
      at::cuda::getCurrentCUDAStream(device).stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// D6: one persistent launch, replicated W1 lookup and sharded W2/logits.
// Candidate tags 20..26 retain bounded cross-rank waits. The former latent
// tags are reserved; --no-markov skips lookup/bias but keeps candidate reduction.
void run_markov(at::Tensor markov_w1, at::Tensor markov_w2, at::Tensor logits,
                  at::Tensor block_ids, at::Tensor draft_tokens, at::Tensor prev_token,
                  at::Tensor lat_r, at::Tensor latent, at::Tensor step_bias,
                  at::Tensor step_logits, at::Tensor cand_val, at::Tensor cand_idx,
                  at::Tensor lat_trace, at::Tensor bias_trace, at::Tensor logit_trace,
                  at::Tensor cta_val, at::Tensor cta_idx, at::Tensor barrier, std::vector<at::Tensor> peers,
                  at::Tensor replay_generation, int64_t rank, bool apply_markov,
                  at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor bias_table, const naivert_dspark_ops::HeadArgs* head, unsigned long long* control, std::optional<at::Tensor> sampling_state,
                  std::optional<at::Tensor> sampling_params) {
  TORCH_CHECK(block_ids.is_cuda(), "D6 block IDs must be CUDA");
  const int device = block_ids.get_device();
  const bool fused = head != nullptr;
  const int64_t window = 8, rows = 7, ranks = 8;
  const int64_t vocab = naivert_dspark_ops::head_rows8_vocab();
  const int64_t latent_w = naivert_dspark_ops::markov_block_latent();
  const int64_t scratch = naivert_dspark_ops::markov_block_scratch_elements();
  TORCH_CHECK(rank >= 0 && rank < ranks, "rank must be inside TP8");
  check(markov_w1, "markov_w1", at::kBFloat16, kExportDiagnostics && apply_markov ? ranks * vocab * latent_w : 0, device);
  check(markov_w2, "markov_w2", at::kBFloat16, 0, device);  // Only bias preparation reads W2.
  check(bias_table, "bias_table", at::kBFloat16, apply_markov ? naivert_dspark_ops::markov_bias_table_elements() : 0, device);
  check(logits, "logits", at::kFloat, !fused || kExportDiagnostics ? rows * vocab : 0, device);
  check(block_ids, "block_ids", at::kLong, window, device);
  check(draft_tokens, "draft_tokens", at::kLong, rows, device);
  check(prev_token, "prev_token", at::kLong, 1, device);
  check(lat_r, "lat_r", at::kBFloat16, kExportDiagnostics ? latent_w : 0, device);
  check(latent, "latent", at::kBFloat16, kExportDiagnostics ? latent_w : 0, device);
  check(step_bias, "step_bias", at::kBFloat16, kExportDiagnostics ? vocab : 0, device);
  check(step_logits, "step_logits", at::kFloat, kExportDiagnostics ? vocab : 0, device);
  check(cand_val, "cand_val", at::kFloat, kExportDiagnostics ? ranks : 0, device);
  check(cand_idx, "cand_idx", at::kLong, kExportDiagnostics ? ranks : 0, device);
  check(lat_trace, "lat_trace", at::kBFloat16, kExportDiagnostics ? rows * latent_w : 0, device);
  check(bias_trace, "bias_trace", at::kBFloat16, kExportDiagnostics ? rows * vocab : 0, device);
  check(logit_trace, "logit_trace", at::kFloat, kExportDiagnostics || sampling_state ? rows * vocab : 0, device);
  check(cta_val, "cta_val", at::kFloat, fused ? 0 : scratch, device);
  check(cta_idx, "cta_idx", at::kInt, fused ? 0 : scratch, device);
  check(barrier, "barrier", at::kInt, fused ? 0 : 2, device);
  check(replay_generation, "replay_generation", at::kInt, 1, device);
  // The chain reads block_ids[0] and writes block_ids[1..7]; `draft_tokens` is a separate copy of
  // the same seven values, so the two must not be the same allocation.
  TORCH_CHECK(block_ids.data_ptr() != draft_tokens.data_ptr(),
              "block_ids must not alias draft_tokens");
  TORCH_CHECK(logit_trace.numel() == 0 || logits.numel() == 0 || logit_trace.data_ptr() != logits.data_ptr(),
              "logit_trace must not alias logits: the chain re-reads logits[j] at every step");
  TORCH_CHECK(peers.size() == 8, "D6 requires eight peer mailboxes");

  naivert_dspark_ops::MarkovArgs args{};
  // Initialize padding for deterministic raw argument comparisons.
  std::memset(&args, 0, sizeof(args));
  args.markov_w1 = reinterpret_cast<const __nv_bfloat16*>(markov_w1.data_ptr());
  args.markov_w2 = reinterpret_cast<const __nv_bfloat16*>(markov_w2.data_ptr());
  args.bias_table = reinterpret_cast<const __nv_bfloat16*>(bias_table.data_ptr());
  args.logits = logits.data_ptr<float>();
  args.block_ids = block_ids.data_ptr<int64_t>();
  args.draft_tokens = draft_tokens.data_ptr<int64_t>();
  args.prev_token = prev_token.data_ptr<int64_t>();
  args.lat_r = reinterpret_cast<__nv_bfloat16*>(lat_r.data_ptr());
  args.latent = reinterpret_cast<__nv_bfloat16*>(latent.data_ptr());
  args.step_bias = reinterpret_cast<__nv_bfloat16*>(step_bias.data_ptr());
  args.step_logits = step_logits.data_ptr<float>();
  args.cand_val = cand_val.data_ptr<float>();
  args.cand_idx = cand_idx.data_ptr<int64_t>();
  args.lat_trace = reinterpret_cast<__nv_bfloat16*>(lat_trace.data_ptr());
  args.bias_trace = reinterpret_cast<__nv_bfloat16*>(bias_trace.data_ptr());
  args.logit_trace = logit_trace.data_ptr<float>();
  args.cta_val = cta_val.data_ptr<float>();
  args.cta_idx = cta_idx.data_ptr<int32_t>();
  for (int i = 0; i < 8; ++i) {
    const auto& box = peers[i];
    TORCH_CHECK(box.is_cuda() && box.is_contiguous() && box.scalar_type() == at::kByte &&
        box.numel() >= naivert_dspark_ops::head_markov_mailbox_bytes(), "invalid D6 peer mailbox");
    if (i == rank) TORCH_CHECK(box.get_device() == device, "D6 local mailbox device mismatch");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(box.get_device() != peers[j].get_device(), "D6 peers must use distinct GPUs");
    args.mailbox_peers[i] = box.data_ptr<uint8_t>();
  }
  args.barrier = reinterpret_cast<unsigned*>(barrier.data_ptr<int32_t>());
  args.replay_generation = replay_generation.data_ptr<int32_t>();
  args.rank = static_cast<int>(rank);
  args.apply_markov = apply_markov ? 1 : 0;
  if (sampling_state) {
    check(*sampling_state, "sampling_state", at::kLong, 2, device);
    args.sampling_state = sampling_state->data_ptr<int64_t>();
  }
  if (sampling_params) {
    check(*sampling_params, "sampling_params", at::kFloat, 8, device);
    args.sampling_params = sampling_params->data_ptr<float>();
  }
  bind_watchdog(args, watchdog_flag, watchdog_limit_ns, device);

  c10::cuda::CUDAGuard guard(device);
  if (head) C10_CUDA_CHECK(naivert_dspark_ops::launch_head_markov(
      *head, args, control, at::cuda::getCurrentCUDAStream(device).stream()));
  else C10_CUDA_CHECK(naivert_dspark_ops::launch_markov_block(args,
      at::cuda::getCurrentCUDAStream(device).stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void markov_block(at::Tensor markov_w1, at::Tensor markov_w2, at::Tensor logits,
                  at::Tensor block_ids, at::Tensor draft_tokens, at::Tensor prev_token,
                  at::Tensor lat_r, at::Tensor latent, at::Tensor step_bias,
                  at::Tensor step_logits, at::Tensor cand_val, at::Tensor cand_idx,
                  at::Tensor lat_trace, at::Tensor bias_trace, at::Tensor logit_trace,
                  at::Tensor cta_val, at::Tensor cta_idx, at::Tensor barrier, std::vector<at::Tensor> peers,
                  at::Tensor replay_generation, int64_t rank, bool apply_markov,
                  at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor bias_table, std::optional<at::Tensor> sampling_state) {
  run_markov(markov_w1, markov_w2, logits, block_ids, draft_tokens, prev_token, lat_r, latent, step_bias, step_logits, cand_val, cand_idx, lat_trace, bias_trace, logit_trace, cta_val, cta_idx, barrier, peers, replay_generation, rank, apply_markov, watchdog_flag, watchdog_limit_ns, bias_table, nullptr, nullptr, sampling_state, std::nullopt);
}
void head_markov(at::Tensor markov_w1, at::Tensor markov_w2, at::Tensor logits,
                  at::Tensor block_ids, at::Tensor draft_tokens, at::Tensor prev_token,
                  at::Tensor lat_r, at::Tensor latent, at::Tensor step_bias,
                  at::Tensor step_logits, at::Tensor cand_val, at::Tensor cand_idx,
                  at::Tensor lat_trace, at::Tensor bias_trace, at::Tensor logit_trace,
                  at::Tensor cta_val, at::Tensor cta_idx, at::Tensor barrier, std::vector<at::Tensor> peers,
                  at::Tensor replay_generation, int64_t rank, bool apply_markov,
                  at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor bias_table, at::Tensor resid_in, at::Tensor norm_gamma, at::Tensor lm_head, at::Tensor norm_out, at::Tensor control, double rms_epsilon, int64_t first_logit_row, std::optional<at::Tensor> sampling_state,
                  std::optional<at::Tensor> sampling_params) {
  auto head = make_head_args<true>(resid_in, norm_gamma, lm_head, norm_out, logits, watchdog_flag, watchdog_limit_ns, rms_epsilon, first_logit_row);
  TORCH_CHECK(resid_in.get_device() == logits.get_device(), "head and Markov must share a device");
  check(control, "D56 control", at::kLong, naivert_dspark_ops::head_markov_control_elements(), logits.get_device());
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(control.data_ptr()) % 16 == 0, "D56 control must be 16-byte aligned");
  run_markov(markov_w1, markov_w2, logits, block_ids, draft_tokens, prev_token, lat_r, latent, step_bias, step_logits, cand_val, cand_idx, lat_trace, bias_trace, logit_trace, cta_val, cta_idx, barrier, peers, replay_generation, rank, apply_markov, watchdog_flag, watchdog_limit_ns, bias_table, &head, reinterpret_cast<unsigned long long*>(control.data_ptr<int64_t>()), sampling_state, sampling_params);
}

void prepare_markov_bias(at::Tensor w1, at::Tensor w2, at::Tensor output) {
  const int device = w1.get_device();
  const auto vocab = naivert_dspark_ops::head_rows8_vocab();
  const auto width = naivert_dspark_ops::markov_block_latent();
  check(w1,"w1",at::kBFloat16,8 * vocab * width,device);
  check(w2,"w2",at::kBFloat16,vocab * width,device);
  check(output,"output",at::kBFloat16,naivert_dspark_ops::markov_bias_table_elements(),device);
  TORCH_CHECK(output.data_ptr()!=w1.data_ptr() && output.data_ptr()!=w2.data_ptr(),"bias table must not alias weights");
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::launch_precompute_bias_table(
      reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      at::cuda::getCurrentCUDAStream(device).stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure", &configure);
  m.def("head_rows8", &head_rows8);
  m.def("markov_block", &markov_block);
  m.def("head_markov", &head_markov,pybind11::arg("markov_w1"),pybind11::arg("markov_w2"),pybind11::arg("logits"),pybind11::arg("block_ids"),pybind11::arg("draft_tokens"),pybind11::arg("prev_token"),pybind11::arg("lat_r"),pybind11::arg("latent"),pybind11::arg("step_bias"),pybind11::arg("step_logits"),pybind11::arg("cand_val"),pybind11::arg("cand_idx"),pybind11::arg("lat_trace"),pybind11::arg("bias_trace"),pybind11::arg("logit_trace"),pybind11::arg("cta_val"),pybind11::arg("cta_idx"),pybind11::arg("barrier"),pybind11::arg("peers"),pybind11::arg("replay_generation"),pybind11::arg("rank"),pybind11::arg("apply_markov"),pybind11::arg("watchdog_flag"),pybind11::arg("watchdog_limit_ns"),pybind11::arg("bias_table"),pybind11::arg("resid_in"),pybind11::arg("norm_gamma"),pybind11::arg("lm_head"),pybind11::arg("norm_out"),pybind11::arg("control"),pybind11::arg("rms_epsilon"),pybind11::arg("first_logit_row"),pybind11::arg("sampling_state"),pybind11::arg("sampling_params")=std::nullopt);
  m.def("prepare_markov_bias", &prepare_markov_bias);
  m.def("head_rows8_weight_shape", &naivert_dspark_ops::head_rows8_weight_shape);
  m.def("head_norm_gamma_layout", &naivert_dspark_ops::head_norm_gamma_layout);
  m.def("markov_bias_table_elements", &naivert_dspark_ops::markov_bias_table_elements);
  m.def("markov_block_scratch_elements", &naivert_dspark_ops::markov_block_scratch_elements);
  m.def("head_markov_control_elements", &naivert_dspark_ops::head_markov_control_elements);
  m.def("mailbox_bytes", &naivert_dspark_ops::head_markov_mailbox_bytes);
  m.def("launch_shape", [](int op) {
    TORCH_CHECK(op >= 0 && op <= 2, "operation must be D5=0, D6=1 or D56=2");
    return std::vector<int>{op == 1 ? naivert_dspark_ops::markov_block_ctas() : naivert_dspark_ops::head_rows8_ctas(),
        naivert_dspark_ops::head_markov_threads(op), naivert_dspark_ops::head_markov_shared_bytes(op)};
  });
}
