#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include "naivert/entry/dspark_context_launch.hpp"
#include "naivert/body/dspark_common/diagnostics.hpp"

namespace {
using naivert_dspark_ops::kExportDiagnostics;
void check(const at::Tensor& t, const char* name, at::ScalarType dtype,
           int64_t elements, int device) {
  TORCH_CHECK(t.is_cuda() && t.get_device() == device && t.is_contiguous() &&
              t.scalar_type() == dtype && t.numel() == elements, "invalid context tensor: ", name);
}
template<class Args>
void bind_watchdog(Args& args, const at::Tensor& flag, int64_t limit, int device) {
  TORCH_CHECK(limit >= 0, "negative D1 watchdog bound");
  check(flag, "watchdog", at::kInt, 1, device);
  args.watchdog_limit_ns = static_cast<unsigned long long>(limit);
  args.watchdog_flag = reinterpret_cast<unsigned*>(flag.data_ptr<int32_t>());
}
void configure(int device) {
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::configure_context(device));
}
template<bool Fused>
naivert_dspark_ops::FcCtxArgs make_fc_ctx(at::Tensor target_hidden, at::Tensor fc, at::Tensor hidden_norm, at::Tensor ctx_valid,
            at::Tensor fc_partial, at::Tensor ctx_sum, at::Tensor ctx, at::Tensor ssq,
            at::Tensor barrier, std::vector<at::Tensor> peers, at::Tensor replay_generation,
            int64_t rank, at::Tensor watchdog_flag,
            int64_t watchdog_limit_ns, double rms_epsilon) {
  TORCH_CHECK(target_hidden.is_cuda(), "D1 input must be CUDA");
  const int device = target_hidden.get_device();
  const int64_t window = 8, hidden = 4096;
  check(target_hidden, "target_hidden", at::kBFloat16, window * hidden, device);
  check(fc, "fc", at::kBFloat16, hidden * hidden, device);
  check(hidden_norm, "hidden_norm", at::kFloat, hidden, device);
  check(ctx_valid, "ctx_valid", at::kBFloat16, window, device);
  check(fc_partial, "fc_partial", at::kBFloat16, kExportDiagnostics ? window * hidden : 0, device);
  check(ctx_sum, "ctx_sum", at::kBFloat16, Fused || kExportDiagnostics ? window * hidden : 0, device);
  check(ctx, "ctx", at::kBFloat16, !Fused || kExportDiagnostics ? window * hidden : 0, device);
  check(ssq, "ssq", at::kFloat, window * naivert_dspark_ops::context_ctas(), device);
  check(barrier, "barrier", at::kInt, 2, device);
  TORCH_CHECK(peers.size() == 8, "D1 requires eight peer mailboxes");
  check(replay_generation, "replay_generation", at::kInt, 1, device);
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be inside TP8");

  naivert_dspark_ops::FcCtxArgs args{};
  // Initialize padding as well, so raw kernel-argument comparisons are deterministic.
  std::memset(&args, 0, sizeof(args));
  TORCH_CHECK(std::isfinite(rms_epsilon) && rms_epsilon > 0, "invalid RMS epsilon");
  args.rms_epsilon = static_cast<float>(rms_epsilon);
  args.target_hidden = reinterpret_cast<const __nv_bfloat16*>(target_hidden.data_ptr());
  args.fc = reinterpret_cast<const __nv_bfloat16*>(fc.data_ptr());
  args.hidden_norm = hidden_norm.data_ptr<float>();
  args.ctx_valid = reinterpret_cast<const __nv_bfloat16*>(ctx_valid.data_ptr());
  args.fc_partial = reinterpret_cast<__nv_bfloat16*>(fc_partial.data_ptr());
  args.ctx_sum = reinterpret_cast<__nv_bfloat16*>(ctx_sum.data_ptr());
  args.ctx = reinterpret_cast<__nv_bfloat16*>(ctx.data_ptr());
  args.ssq = ssq.data_ptr<float>();
  args.barrier = reinterpret_cast<unsigned*>(barrier.data_ptr<int32_t>());
  for (int i = 0; i < 8; ++i) {
    const auto& box = peers[i];
    TORCH_CHECK(box.is_cuda() && box.is_contiguous() && box.scalar_type() == at::kByte &&
                box.numel() >= naivert_dspark_ops::context_mailbox_bytes(), "invalid D1 peer mailbox");
    if (i == rank) TORCH_CHECK(box.get_device() == device, "D1 local mailbox device mismatch");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(box.get_device() != peers[j].get_device(), "D1 peers must use distinct GPUs");
    args.mailbox_peers[i] = box.data_ptr<uint8_t>();
  }
  args.replay_generation = reinterpret_cast<std::int32_t*>(replay_generation.data_ptr());
  args.generation_salt = 2;
  args.rank = static_cast<int>(rank);
  bind_watchdog(args, watchdog_flag, watchdog_limit_ns, device);

  return args;
}

void fc_ctx(at::Tensor target_hidden, at::Tensor fc, at::Tensor hidden_norm, at::Tensor ctx_valid,
            at::Tensor fc_partial, at::Tensor ctx_sum, at::Tensor ctx, at::Tensor ssq,
            at::Tensor barrier, std::vector<at::Tensor> peers, at::Tensor replay_generation,
            int64_t rank, at::Tensor watchdog_flag,
            int64_t watchdog_limit_ns, double rms_epsilon) {
  auto args = make_fc_ctx<false>(target_hidden, fc, hidden_norm, ctx_valid, fc_partial, ctx_sum, ctx, ssq, barrier, peers, replay_generation, rank, watchdog_flag, watchdog_limit_ns, rms_epsilon);
  const int device = target_hidden.get_device();
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::launch_fc_ctx(args, at::cuda::getCurrentCUDAStream(device).stream()));
}

// D2 needs no mailbox: every rank projects the reduced context into its KV shard.
template<bool Fused>
naivert_dspark_ops::CtxKvArgs make_ctx_kv(at::Tensor ctx, std::vector<at::Tensor> k_w, std::vector<at::Tensor> v_w,
            std::vector<at::Tensor> k_norm, at::Tensor rope_cos, at::Tensor rope_sin,
            at::Tensor ctx_pos, at::Tensor kv_dst, at::Tensor kv_raw, at::Tensor kv_k,
            at::Tensor kv_v, at::Tensor barrier, double rms_epsilon) {
  TORCH_CHECK(ctx.is_cuda(), "D2 context must be CUDA");
  const int device = ctx.get_device();
  const int64_t window = 8, hidden = 4096, head = 128, layers = 5;
  check(ctx, "ctx", at::kBFloat16, !Fused || kExportDiagnostics ? window * hidden : 0, device);
  TORCH_CHECK(int64_t(k_w.size()) == layers && int64_t(v_w.size()) == layers &&
                  int64_t(k_norm.size()) == layers,
              "k_w / v_w / k_norm must each hold one tensor per draft layer");
  // The caller must bound every device-owned ctx_pos and kv_dst, including masked
  // rows. Reading those values here would add a host synchronization and break
  // capture; this binding checks storage metadata only.
  TORCH_CHECK(rope_cos.numel() > 0 && rope_cos.numel() % head == 0 &&
                  rope_cos.numel() == rope_sin.numel(),
              "rope tables must be a non-empty [rope_rows, 128] pair of the same size");
  check(rope_cos, "rope_cos", at::kFloat, rope_cos.numel(), device);
  check(rope_sin, "rope_sin", at::kFloat, rope_cos.numel(), device);
  check(ctx_pos, "ctx_pos", at::kLong, window, device);
  check(kv_dst, "kv_dst", at::kLong, window, device);
  check(kv_raw, "kv_raw", at::kBFloat16, !Fused || kExportDiagnostics ? layers * 2 * window * head : 0, device);
  check(barrier, "barrier", at::kInt, 2, device);
  // kv_k / kv_v are [layers, cache_rows, 1, 128]; cache_rows is derived rather than passed so the
  // two cannot disagree with the tensor the kernel actually writes into.
  TORCH_CHECK(kv_k.is_cuda() && kv_k.is_contiguous() && kv_k.scalar_type() == at::kBFloat16 &&
                  kv_k.sizes() == kv_v.sizes() && kv_k.dim() == 4 && kv_k.size(0) == layers &&
                  kv_k.size(2) == 1 && kv_k.size(3) == head,
              "kv_k / kv_v must be a contiguous bf16 [5, cache_rows, 1, 128] pair");
  const int64_t cache_rows = kv_k.size(1);
  TORCH_CHECK(cache_rows > 0 && cache_rows <= 1048577, "invalid D2 cache row count");   // capacity + 1, the draft pin
  check(kv_k, "kv_k", at::kBFloat16, kv_k.numel(), device);
  check(kv_v, "kv_v", at::kBFloat16, kv_k.numel(), device);

  naivert_dspark_ops::CtxKvArgs args{};
  // Initialize padding as well, so raw kernel-argument comparisons are deterministic.
  std::memset(&args, 0, sizeof(args));
  TORCH_CHECK(std::isfinite(rms_epsilon) && rms_epsilon > 0, "invalid RMS epsilon");
  args.rms_epsilon = static_cast<float>(rms_epsilon);
  args.ctx = reinterpret_cast<const __nv_bfloat16*>(ctx.data_ptr());
  for (int L = 0; L < layers; ++L) {
    check(k_w[L], "k_w", at::kBFloat16, head * hidden, device);
    check(v_w[L], "v_w", at::kBFloat16, head * hidden, device);
    check(k_norm[L], "k_norm", at::kFloat, head, device);
    args.k_w[L] = reinterpret_cast<const __nv_bfloat16*>(k_w[L].data_ptr());
    args.v_w[L] = reinterpret_cast<const __nv_bfloat16*>(v_w[L].data_ptr());
    args.k_norm[L] = k_norm[L].data_ptr<float>();
  }
  args.rope_cos = rope_cos.data_ptr<float>();
  args.rope_sin = rope_sin.data_ptr<float>();
  args.ctx_pos = ctx_pos.data_ptr<int64_t>();
  args.kv_dst = kv_dst.data_ptr<int64_t>();
  args.kv_raw = reinterpret_cast<__nv_bfloat16*>(kv_raw.data_ptr());
  args.kv_k = reinterpret_cast<__nv_bfloat16*>(kv_k.data_ptr());
  args.kv_v = reinterpret_cast<__nv_bfloat16*>(kv_v.data_ptr());
  args.barrier = reinterpret_cast<unsigned*>(barrier.data_ptr<int32_t>());
  args.cache_rows = static_cast<int>(cache_rows);

  return args;
}

void ctx_kv(at::Tensor ctx, std::vector<at::Tensor> k_w, std::vector<at::Tensor> v_w,
            std::vector<at::Tensor> k_norm, at::Tensor rope_cos, at::Tensor rope_sin,
            at::Tensor ctx_pos, at::Tensor kv_dst, at::Tensor kv_raw, at::Tensor kv_k,
            at::Tensor kv_v, at::Tensor barrier, double rms_epsilon) {
  auto args = make_ctx_kv<false>(ctx, k_w, v_w, k_norm, rope_cos, rope_sin, ctx_pos, kv_dst, kv_raw, kv_k, kv_v, barrier, rms_epsilon);
  const int device = ctx.get_device();
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::launch_ctx_kv(args, at::cuda::getCurrentCUDAStream(device).stream()));
}


void fc_ctx_kv(at::Tensor target_hidden, at::Tensor fc, at::Tensor hidden_norm,
    at::Tensor ctx_valid, at::Tensor fc_partial, at::Tensor ctx_sum, at::Tensor ctx,
    at::Tensor ssq, at::Tensor barrier, std::vector<at::Tensor> peers, at::Tensor replay_generation,
    int64_t rank, at::Tensor watchdog_flag,
    int64_t watchdog_limit_ns, std::vector<at::Tensor> k_w, std::vector<at::Tensor> v_w,
    std::vector<at::Tensor> k_norm, at::Tensor rope_cos, at::Tensor rope_sin,
    at::Tensor ctx_pos, at::Tensor kv_dst, at::Tensor kv_raw, at::Tensor kv_k,
    at::Tensor kv_v, at::Tensor kv_partials, double rms_epsilon) {
  TORCH_CHECK(ctx.device() == target_hidden.device(), "D12 buffers must share a device");
  naivert_dspark_ops::FcCtxKvArgs args{};
  std::memset(&args, 0, sizeof(args));
  args.fc = make_fc_ctx<true>(target_hidden, fc, hidden_norm, ctx_valid, fc_partial,
      ctx_sum, ctx, ssq, barrier, peers, replay_generation, rank,
      watchdog_flag, watchdog_limit_ns, rms_epsilon);
  args.kv = make_ctx_kv<true>(ctx, k_w, v_w, k_norm, rope_cos, rope_sin, ctx_pos,
      kv_dst, kv_raw, kv_k, kv_v, barrier, rms_epsilon);
  const int device = target_hidden.get_device();
  check(kv_partials, "kv_partials", at::kFloat, naivert_dspark_ops::fc_ctx_kv_partial_elements(), device);
  args.kv_partials = kv_partials.data_ptr<float>();
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::launch_fc_ctx_kv(args,
      at::cuda::getCurrentCUDAStream(device).stream()));
}

}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure", &configure);
  m.def("fc_ctx", &fc_ctx);
  m.def("ctx_kv", &ctx_kv);
  m.def("fc_ctx_kv", &fc_ctx_kv);
  m.def("partial_elements", &naivert_dspark_ops::fc_ctx_kv_partial_elements);
  m.def("mailbox_bytes", &naivert_dspark_ops::context_mailbox_bytes);
  m.def("launch_shape", [](int operation) { return std::vector<int>{
      naivert_dspark_ops::context_ctas(), naivert_dspark_ops::context_threads(),
      naivert_dspark_ops::context_shared_bytes(operation)}; });
}
