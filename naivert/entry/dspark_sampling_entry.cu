#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#if NAIVERT_DSPARK_PERSISTENT
#include "naivert/body/dspark_sampling/persistent.cuh"
#else
#include "naivert/body/dspark_sampling/sharded.cuh"
#endif

namespace {
#if NAIVERT_DSPARK_PERSISTENT
namespace sampling = naivert_dspark_persistent;
#else
namespace sampling = naivert_dspark_sharded;
#endif
void check(const at::Tensor& t, at::ScalarType type, int64_t count,
           int device) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == type &&
                  t.numel() == count && t.get_device() == device,
              "invalid sampling tensor");
}
void sample(at::Tensor logits, at::Tensor draft_logits,
            std::vector<int64_t> flags, std::vector<int64_t> mailboxes,
            at::Tensor p, at::Tensor q, at::Tensor tokens, at::Tensor proposals,
            at::Tensor rng, at::Tensor scratch, at::Tensor watchdog,
            at::Tensor rep_count, at::Tensor sampling_params, int rank,
            bool anchor, bool defer_consumed) {
  TORCH_CHECK(!anchor || !defer_consumed,
              "anchor must complete its own rendezvous");
  TORCH_CHECK(
      flags.size() == 8 && mailboxes.size() == 8 && rank >= 0 && rank < 8,
      "TP8 required");
  int rows = anchor ? 1 : 8, device = p.get_device();
  c10::cuda::CUDAGuard guard(device);
  check(logits, at::kFloat, rows * sampling::vocabulary, device);
  check(draft_logits, at::kFloat, anchor ? 0 : 7 * sampling::vocabulary,
        device);
  check(p, at::kFloat, rows * sampling::vocabulary, device);
  check(q, at::kFloat, anchor ? 0 : 7 * sampling::vocabulary, device);
  check(tokens, at::kInt, rows, device);
  check(proposals, at::kLong, 7, device);
  check(rng, at::kLong, 2, device);
  check(scratch, at::kByte, sizeof(sampling::Scratch), device);
  check(watchdog, at::kInt, 1, device);
  check(rep_count, at::kInt, 8 * sampling::vocabulary, device);
  check(sampling_params, at::kFloat, 8, device);
  sampling::Args a{};
  a.rep_count = rep_count.data_ptr<int32_t>();
  a.sampling_params = sampling_params.data_ptr<float>();
  a.logits = logits.data_ptr<float>();
  a.draft_logits = draft_logits.data_ptr<float>();
  a.p = p.data_ptr<float>();
  a.q = q.data_ptr<float>();
  a.tokens = tokens.data_ptr<int32_t>();
  a.proposals = proposals.data_ptr<int64_t>();
  a.rng = rng.data_ptr<int64_t>();
  a.watchdog = reinterpret_cast<unsigned*>(watchdog.data_ptr<int32_t>());
  a.rank = rank;
  a.anchor = anchor;
  for (int r = 0; r < 8; ++r) {
    TORCH_CHECK(flags[r] && mailboxes[r], "null sampling peer");
    a.flags[r] = reinterpret_cast<unsigned long long*>(flags[r]);
    a.peers[r] = reinterpret_cast<sampling::Mailbox*>(mailboxes[r]);
  }
  const auto stream = at::cuda::getCurrentCUDAStream(device);
#if NAIVERT_DSPARK_PERSISTENT
  C10_CUDA_CHECK(sampling::run(
      a, reinterpret_cast<sampling::Scratch*>(scratch.data_ptr<uint8_t>()),
      defer_consumed, stream));
#else
  sampling::run(
      a, reinterpret_cast<sampling::Scratch*>(scratch.data_ptr<uint8_t>()),
      defer_consumed, stream);
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void naivert_dspark_uniforms_kernel(const int64_t* rng, float* out, int domain,
                                int row, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = naivert_dspark_ops::sampling_uniform(rng, domain, row, i);
}
void uniforms(at::Tensor rng, at::Tensor out, int domain, int row) {
  int device = out.get_device();
  c10::cuda::CUDAGuard guard(device);
  check(rng, at::kLong, 2, device);
  check(out, at::kFloat, out.numel(), device);
  TORCH_CHECK(out.numel() > 0 && out.numel() <= sampling::full_vocabulary &&
                  domain >= 0 && domain <= 3 && row >= 0 && row < 8,
              "invalid RNG domain/row/size");
  naivert_dspark_uniforms_kernel<<<(out.numel() + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream(device)>>>(
      rng.data_ptr<int64_t>(), out.data_ptr<float>(), domain, row, out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#if NAIVERT_DSPARK_PERSISTENT
// Diagnostic: the tagged survivor-count words of this rank's inbox ([max_rows][8] u64, tag << 32 | count;
// tag bit 31 set = second-stage gather; count == cand_cap + 1 = that source overflowed).
at::Tensor cand_counts(at::Tensor mailbox) {
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kByte &&
              mailbox.numel() >= int64_t(sizeof(sampling::Mailbox)), "mailbox tensor");
  c10::cuda::CUDAGuard guard(mailbox.get_device());
  auto out = torch::empty({sampling::max_rows * 8}, torch::dtype(torch::kInt64));
  const auto* mb = reinterpret_cast<const sampling::Mailbox*>(mailbox.data_ptr<uint8_t>());
  C10_CUDA_CHECK(cudaMemcpy(out.data_ptr(), &mb->inbox.cand_count[0][0], out.numel() * sizeof(int64_t),
                            cudaMemcpyDeviceToHost));
  return out;
}
#endif
#if NAIVERT_DSPARK_PERSISTENT && defined(PERSISTENT_STAMPS)
// The last launch's per-CTA %globaltimer stamps on `device` (stamp_points x stamp_ctas, ns).
at::Tensor stamps(int device) {
  c10::cuda::CUDAGuard guard(device);
  auto out = torch::empty({sampling::stamp_points * sampling::stamp_ctas},
                          torch::dtype(torch::kInt64));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(out.data_ptr(), sampling::g_stamps,
                                      out.numel() * sizeof(int64_t)));
  return out;
}
#endif
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &sample);
  m.def("uniforms", &uniforms);
#if NAIVERT_DSPARK_PERSISTENT
  m.def("cand_counts", &cand_counts);
  m.def("cand_cap", [] { return sampling::cand_cap; });
  m.def("cand_shared", [] { return sampling::cand_shared; });
#endif
#if NAIVERT_DSPARK_PERSISTENT && defined(PERSISTENT_STAMPS)
  m.def("cand_history", [](int device) {
    c10::cuda::CUDAGuard guard(device);
    auto out = torch::empty({sampling::cand_hist_words + 1}, torch::dtype(torch::kInt64));
    unsigned next = 0;
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(&next, sampling::g_cand_hist_next, sizeof(next)));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(out.data_ptr(), sampling::g_cand_hist, sampling::cand_hist_words * sizeof(int64_t)));
    out[sampling::cand_hist_words] = int64_t(next);
    return out;
  });
  m.def("stamps", &stamps);
  m.def("stamp_points", [] { return sampling::stamp_points; });
  m.def("stamp_ctas", [] { return sampling::stamp_ctas; });
#endif
  m.def("scratch_size", [] { return sizeof(sampling::Scratch); });
  m.def("mailbox_size", [] { return sizeof(sampling::Mailbox); });
}
