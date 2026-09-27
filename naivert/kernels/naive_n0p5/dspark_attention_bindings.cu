#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <type_traits>
#include "naivert/entry/dspark_attention_launch.hpp"
#include "naivert/body/dspark_common/diagnostics.hpp"
#include "naivert/body/dspark_common/mailbox_layout.hpp"

namespace {
using naivert_dspark_ops::kExportDiagnostics;
void check(const at::Tensor& tensor, const char* name, at::ScalarType dtype,
           int64_t elements, int device) {
  TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device &&
              tensor.scalar_type() == dtype && tensor.is_contiguous() &&
              tensor.numel() == elements, "invalid D3 tensor: ", name);
}
template<class Args>
void bind_watchdog(Args& args, const at::Tensor& flag, int64_t limit, int device) {
  TORCH_CHECK(limit >= 0, "negative D3 watchdog bound");
  check(flag, "watchdog", at::kInt, 1, device);
  args.watchdog_limit_ns = static_cast<unsigned long long>(limit);
  args.watchdog_flag = reinterpret_cast<unsigned*>(flag.data_ptr<int32_t>());
}
void configure(int device) {
  c10::cuda::CUDAGuard guard(device);
  C10_CUDA_CHECK(naivert_dspark_ops::configure_attn_rows8(device));
}
template<bool Sliding>
void attn_rows8_impl(at::Tensor resid_in, at::Tensor input_ln, at::Tensor q_w, at::Tensor k_w,
                at::Tensor v_w, at::Tensor q_norm, at::Tensor k_norm, at::Tensor o_w,
                at::Tensor rope_cos, at::Tensor rope_sin, at::Tensor blk_pos, at::Tensor key_add,
                at::Tensor kv_k, at::Tensor kv_v, at::Tensor qk_raw, at::Tensor q_out,
                at::Tensor k_blk, at::Tensor v_blk, at::Tensor scores, at::Tensor attn_out,
                at::Tensor o_partial, at::Tensor hid_a, at::Tensor barrier, std::vector<at::Tensor> peers,
                at::Tensor replay_generation, int64_t layer, int64_t rank,
                at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor split_scratch, int64_t window_size, double rms_epsilon, at::Tensor x_norm) {
  TORCH_CHECK(resid_in.is_cuda(), "D3 input must be CUDA");
  const int device = resid_in.get_device();
  const int64_t window = 8, hidden = 4096, head = 128, qrows = 512, layers = 5, qheads = 4;
  TORCH_CHECK(layer >= 0 && layer < layers, "layer must be inside the draft's five");
  TORCH_CHECK(rank >= 0 && rank < 8, "rank must be inside TP8");
  check(resid_in, "resid_in", at::kBFloat16, window * hidden, device);
  check(input_ln, "input_ln", at::kFloat, hidden, device);
  check(q_w, "q_w", at::kBFloat16, qrows * hidden, device);
  check(k_w, "k_w", at::kBFloat16, head * hidden, device);
  check(v_w, "v_w", at::kBFloat16, head * hidden, device);
  check(q_norm, "q_norm", at::kFloat, head, device);
  check(k_norm, "k_norm", at::kFloat, head, device);
  check(o_w, "o_w", at::kBFloat16, hidden * qrows, device);
  check(blk_pos, "blk_pos", at::kLong, window, device);
  check(qk_raw, "qk_raw", at::kBFloat16,
        naivert_dspark_ops::attn_rows8_qk_raw_elems(), device);
  check(q_out, "q_out", at::kBFloat16, kExportDiagnostics ? window * qrows : 0, device);
  check(k_blk, "k_blk", at::kBFloat16, kExportDiagnostics ? window * head : 0, device);
  check(v_blk, "v_blk", at::kBFloat16, window * head, device);
  check(attn_out, "attn_out", at::kBFloat16, window * qrows, device);
  check(o_partial, "o_partial", at::kBFloat16, kExportDiagnostics ? window * hidden : 0, device);
  check(hid_a, "hid_a", at::kBFloat16, window * hidden, device);
  check(barrier, "barrier", at::kInt, 2, device);
  check(replay_generation, "replay_generation", at::kInt, 1, device);
  TORCH_CHECK(hid_a.data_ptr() != resid_in.data_ptr(),
              "hid_a must not alias resid_in: the residual add reads resid[L] after the reduce");
  TORCH_CHECK(peers.size() == 8, "D3 requires eight peer mailboxes");
  // kv_k / kv_v are [5, cache_rows, 1, 128]; `cache_rows` is derived from the tensor rather than
  // passed, so the stride the kernel walks and the tensor it walks cannot disagree.
  TORCH_CHECK(kv_k.is_cuda() && kv_k.is_contiguous() && kv_k.scalar_type() == at::kBFloat16 &&
                  kv_k.sizes() == kv_v.sizes() && kv_k.dim() == 4 && kv_k.size(0) == layers &&
                  kv_k.size(2) == 1 && kv_k.size(3) == head,
              "kv_k / kv_v must be a contiguous bf16 [5, cache_rows, 1, 128] pair");
  check(kv_k, "kv_k", at::kBFloat16, kv_k.numel(), device);
  check(kv_v, "kv_v", at::kBFloat16, kv_k.numel(), device);
  const int64_t cache_rows = kv_k.size(1);
  // The trash row D2 scatters its padded rows into is the LAST row of the cache and is not a key
  // column; `ncache` is therefore one short of the allocation, exactly as `d3_compute`'s
  // `ncache = b.max_seq_len` is.
  const int64_t capacity = cache_rows - 1;
  if constexpr (Sliding) {
    TORCH_CHECK(window_size == 1024, "SWA D3 requires window_size=1024");
    TORCH_CHECK(std::isfinite(rms_epsilon) && rms_epsilon > 0, "invalid D3 RMS epsilon");
    TORCH_CHECK(capacity >= window && capacity <= 1048576 && capacity % 4 == 0,
                "SWA D3 cache capacity must be a multiple of four in [8,1048576]");
    check(x_norm, "x_norm", at::kBFloat16, kExportDiagnostics ? window * hidden : 0, device);
  }
  const int64_t ncache = Sliding ? ((window_size - 1 + 15) / 16) * 16 : capacity;
  const int64_t total = ncache + window;
  const int64_t padded = (total + 255) / 256 * 256;
  TORCH_CHECK(ncache > 0 && ncache % 4 == 0,
              "ncache must be a positive multiple of 4: the probs.values loop steps by four and "
              "must not straddle the cache/block boundary");
  TORCH_CHECK(padded <= naivert_dspark_ops::attn_rows8_max_padded_cols(),
              "a row of ", padded, " padded score columns does not fit the shared-memory scratch "
              "(max ", naivert_dspark_ops::attn_rows8_max_padded_cols(), ")");
  check(key_add, "key_add", at::kFloat, capacity + window, device);
  check(scores, "scores", at::kFloat, kExportDiagnostics ? qheads * window * total : 0, device);
  TORCH_CHECK(rope_cos.numel() == rope_sin.numel() && rope_cos.numel() >= (capacity + window) * head,
              "rope tables must be a matching pair covering at least ", total, " positions");
  check(rope_cos, "rope_cos", at::kFloat, rope_cos.numel(), device);
  check(rope_sin, "rope_sin", at::kFloat, rope_cos.numel(), device);

  check(split_scratch, "split_scratch", at::kFloat,
        naivert_dspark_ops::attn_rows8_split_elems(), device);
  using Args = std::conditional_t<Sliding, naivert_dspark_ops::SwaAttnArgs, naivert_dspark_ops::AttnArgs>;
  Args args{};
  // Padding too: these bytes are hashed by gate1b's argument comparison (cuFuncGetParamInfo
  // reads the whole parameter), and `{}` on an aggregate leaves padding unspecified.
  std::memset(&args, 0, sizeof(args));
  if constexpr (Sliding) {
    args.window_left = static_cast<int>(window_size - 1);
    args.rms_epsilon = static_cast<float>(rms_epsilon);
    args.x_norm = reinterpret_cast<__nv_bfloat16*>(x_norm.data_ptr());
  }
  args.split_scratch = split_scratch.data_ptr<float>();
  args.resid_in = reinterpret_cast<const __nv_bfloat16*>(resid_in.data_ptr());
  args.input_ln = input_ln.data_ptr<float>();
  args.q_w = reinterpret_cast<const __nv_bfloat16*>(q_w.data_ptr());
  args.k_w = reinterpret_cast<const __nv_bfloat16*>(k_w.data_ptr());
  args.v_w = reinterpret_cast<const __nv_bfloat16*>(v_w.data_ptr());
  args.q_norm = q_norm.data_ptr<float>();
  args.k_norm = k_norm.data_ptr<float>();
  args.o_w = reinterpret_cast<const __nv_bfloat16*>(o_w.data_ptr());
  args.rope_cos = rope_cos.data_ptr<float>();
  args.rope_sin = rope_sin.data_ptr<float>();
  args.blk_pos = blk_pos.data_ptr<int64_t>();
  args.key_add = key_add.data_ptr<float>();
  args.kv_k = reinterpret_cast<const __nv_bfloat16*>(kv_k.data_ptr());
  args.kv_v = reinterpret_cast<const __nv_bfloat16*>(kv_v.data_ptr());
  args.qk_raw = reinterpret_cast<__nv_bfloat16*>(qk_raw.data_ptr());
  args.q_out = reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr());
  args.k_blk = reinterpret_cast<__nv_bfloat16*>(k_blk.data_ptr());
  args.v_blk = reinterpret_cast<__nv_bfloat16*>(v_blk.data_ptr());
  args.scores = scores.data_ptr<float>();
  args.attn_out = reinterpret_cast<__nv_bfloat16*>(attn_out.data_ptr());
  args.o_partial = reinterpret_cast<__nv_bfloat16*>(o_partial.data_ptr());
  args.hid_a = reinterpret_cast<__nv_bfloat16*>(hid_a.data_ptr());
  args.barrier = reinterpret_cast<unsigned*>(barrier.data_ptr<int32_t>());
  for (int i = 0; i < 8; ++i) {
    const auto& box = peers[i];
    TORCH_CHECK(box.is_cuda() && box.is_contiguous() && box.scalar_type() == at::kByte &&
                box.numel() >= naivert_dspark_ops::kRowSlabs * naivert::kMailboxSlabStride,
                "invalid D3 peer mailbox");
    if (i == rank) TORCH_CHECK(box.get_device() == device, "D3 local mailbox device mismatch");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(box.get_device() != peers[j].get_device(), "D3 peers must use distinct GPUs");
    args.mailbox_peers[i] = box.data_ptr<uint8_t>();
  }
  args.replay_generation = reinterpret_cast<const std::int32_t*>(replay_generation.data_ptr());
  args.generation_salt =
      static_cast<std::uint32_t>(naivert_dspark_ops::attn_rows8_salt(static_cast<int>(layer)));
  // Computed HERE, in double, and passed down: `1.0f / sqrtf(128.0f)` in device code is not
  // guaranteed to be the same float as the stand-in's `1.0 / math.sqrt(128)` narrowed to fp32,
  // and the difference would be a silent one-ulp drift through every score.
  args.attn_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head)));
  args.layer = static_cast<int>(layer);
  args.cache_rows = static_cast<int>(cache_rows);
  args.ncache = static_cast<int>(ncache);
  args.total_cols = static_cast<int>(total);
  args.padded_cols = static_cast<int>(padded);
  args.rank = static_cast<int>(rank);
  bind_watchdog(args, watchdog_flag, watchdog_limit_ns, device);

  c10::cuda::CUDAGuard guard(device);
  if constexpr (Sliding) {
  C10_CUDA_CHECK(naivert_dspark_ops::launch_attn_rows8_swa(
      args, at::cuda::getCurrentCUDAStream(device).stream()));
  } else {
  C10_CUDA_CHECK(naivert_dspark_ops::launch_attn_rows8(
      args, at::cuda::getCurrentCUDAStream(device).stream()));
  }
}

void attn_rows8(at::Tensor resid_in, at::Tensor input_ln, at::Tensor q_w, at::Tensor k_w,
                at::Tensor v_w, at::Tensor q_norm, at::Tensor k_norm, at::Tensor o_w,
                at::Tensor rope_cos, at::Tensor rope_sin, at::Tensor blk_pos, at::Tensor key_add,
                at::Tensor kv_k, at::Tensor kv_v, at::Tensor qk_raw, at::Tensor q_out,
                at::Tensor k_blk, at::Tensor v_blk, at::Tensor scores, at::Tensor attn_out,
                at::Tensor o_partial, at::Tensor hid_a, at::Tensor barrier, std::vector<at::Tensor> peers,
                at::Tensor replay_generation, int64_t layer, int64_t rank,
                at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor split_scratch) {
  attn_rows8_impl<false>(resid_in, input_ln, q_w, k_w, v_w, q_norm, k_norm, o_w, rope_cos, rope_sin, blk_pos, key_add, kv_k, kv_v, qk_raw, q_out, k_blk, v_blk, scores, attn_out, o_partial, hid_a, barrier, peers, replay_generation, layer, rank, watchdog_flag, watchdog_limit_ns, split_scratch, 0, 1e-6, at::Tensor{});
}

void attn_rows8_swa(at::Tensor resid_in, at::Tensor input_ln, at::Tensor q_w, at::Tensor k_w,
                at::Tensor v_w, at::Tensor q_norm, at::Tensor k_norm, at::Tensor o_w,
                at::Tensor rope_cos, at::Tensor rope_sin, at::Tensor blk_pos, at::Tensor key_add,
                at::Tensor kv_k, at::Tensor kv_v, at::Tensor qk_raw, at::Tensor q_out,
                at::Tensor k_blk, at::Tensor v_blk, at::Tensor scores, at::Tensor attn_out,
                at::Tensor o_partial, at::Tensor hid_a, at::Tensor barrier, std::vector<at::Tensor> peers,
                at::Tensor replay_generation, int64_t layer, int64_t rank,
                at::Tensor watchdog_flag, int64_t watchdog_limit_ns, at::Tensor split_scratch, int64_t window_size, double rms_epsilon, at::Tensor x_norm) {
  attn_rows8_impl<true>(resid_in, input_ln, q_w, k_w, v_w, q_norm, k_norm, o_w, rope_cos, rope_sin, blk_pos, key_add, kv_k, kv_v, qk_raw, q_out, k_blk, v_blk, scores, attn_out, o_partial, hid_a, barrier, peers, replay_generation, layer, rank, watchdog_flag, watchdog_limit_ns, split_scratch, window_size, rms_epsilon, x_norm);
}

}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure", &configure);
  m.def("attn_rows8", &attn_rows8);
  m.def("attn_rows8_swa", &attn_rows8_swa);
  m.def("split_elems", &naivert_dspark_ops::attn_rows8_split_elems);
  m.def("launch_shape", [] { return std::vector<int>{naivert_dspark_ops::attn_rows8_ctas(),
      naivert_dspark_ops::attn_rows8_threads(), naivert_dspark_ops::attn_rows8_shared_bytes()}; });
}
