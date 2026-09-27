#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <optional>
#include <tuple>
#include "naivert/entry/dspark_tail_launch.hpp"
namespace naivert_dspark_tail {
#define FIELD_NAME(name, type, dtype, count) #name,
const std::vector<std::string> geometry_fields{NAIVERT_DSPARK_GEOMETRY_FIELDS(FIELD_NAME)};
const std::vector<std::string> accept_fields{NAIVERT_DSPARK_ACCEPT_FIELDS(FIELD_NAME)};
#undef FIELD_NAME

// Allocation metadata and launch validation derive from the same field lists.
auto layouts(int max_seq_len) {
  TORCH_CHECK(max_seq_len >= 8 && max_seq_len <= 1048576, "tail layout capacity");
  const int max_kv = max_seq_len + kRows;
  using Layout = std::vector<std::tuple<std::string, std::string, std::int64_t>>;
#define LAYOUT(name, type, dtype, count) {#name, #dtype, count},
  Layout geometry{NAIVERT_DSPARK_GEOMETRY_FIELDS(LAYOUT)};
  Layout accept{NAIVERT_DSPARK_ACCEPT_FIELDS(LAYOUT)};
#undef LAYOUT
  return std::make_pair(geometry, accept);
}

static void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                         std::int64_t count, int device, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(tensor.get_device() == device && tensor.scalar_type() == dtype,
              name, " has the wrong device or dtype");
  TORCH_CHECK(count < 0 ? tensor.numel() >= kHidden && tensor.numel() % kHidden == 0
                        : tensor.numel() == count, name, " has the wrong size");
}
#define ASSIGN(name, type, dtype, count) \
  check_tensor(values[index], at::k##dtype, count, device, #name); \
  a.name = reinterpret_cast<type*>(values[index++].data_ptr());

void geometry(const std::vector<at::Tensor>& values, int max_seq_len) {
  TORCH_CHECK(values.size() == geometry_fields.size(), "geometry field count");
  TORCH_CHECK(max_seq_len > 0 &&
              max_seq_len <= 1048576,
              "cache bound overflows launch geometry");
  const int max_kv = max_seq_len + kRows;
  const int device = values.front().get_device();
  c10::cuda::CUDAGuard guard(values.front().device());
  GeometryArgs a{};
  int index = 0;
  NAIVERT_DSPARK_GEOMETRY_FIELDS(ASSIGN)
  a.max_seq_len = max_seq_len;
  a.max_kv = max_kv;
  C10_CUDA_CHECK(launch_geometry(a, at::cuda::getCurrentCUDAStream(device)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void accept(const std::vector<at::Tensor>& values, std::int64_t mask_token,
            std::optional<at::Tensor> sampling_state, std::optional<at::Tensor> verify_counts) {
  TORCH_CHECK(values.size() == accept_fields.size(), "accept field count");
  const int device = values.front().get_device();
  c10::cuda::CUDAGuard guard(values.front().device());
  AcceptArgs a{};
  int index = 0;
  NAIVERT_DSPARK_ACCEPT_FIELDS(ASSIGN)
  a.mask_token = mask_token;
  if (sampling_state) {
    check_tensor(*sampling_state, at::kLong, 2, device, "sampling_state");
    a.sampling_state = sampling_state->data_ptr<int64_t>();
  }
  if (verify_counts) {
    check_tensor(*verify_counts, at::kLong, 10, device, "verify_counts");
    a.verify_counts = verify_counts->data_ptr<int64_t>();
  }
  C10_CUDA_CHECK(launch_accept(a, at::cuda::getCurrentCUDAStream(device)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void epilogue(const std::vector<at::Tensor>& geometry_values,
              const std::vector<at::Tensor>& accept_values, int max_seq_len,
              std::int64_t mask_token, const std::vector<at::Tensor>& state,
              const std::vector<std::int64_t>& flags, at::Tensor host_ring,
              at::Tensor host_flags, at::Tensor cancel, int capacity, int rank, bool chain, bool staged,
              std::optional<at::Tensor> rep_count) {
  TORCH_CHECK(geometry_values.size() == geometry_fields.size() &&
              accept_values.size() == accept_fields.size() && state.size() == (staged ? 15 : 11) &&
              flags.size() == 8 && rank >= 0 && rank < 8, "epilogue argument count/rank");
  const int device = geometry_values.front().get_device();
  c10::cuda::CUDAGuard guard(geometry_values.front().device());
  TORCH_CHECK(max_seq_len > 0 && max_seq_len <= 1048576, "epilogue capacity");
  const int max_kv = max_seq_len + kRows;
  EpilogueArgs e{};
  {
    auto& a = e.g;
    const auto& values = geometry_values;
    int index = 0;
    NAIVERT_DSPARK_GEOMETRY_FIELDS(ASSIGN)
    a.max_seq_len = max_seq_len;
    a.max_kv = max_kv;
  }
  {
    auto& a = e.a;
    const auto& values = accept_values;
    int index = 0;
    NAIVERT_DSPARK_ACCEPT_FIELDS(ASSIGN)
    a.mask_token = mask_token;
  }
  TORCH_CHECK(e.g.ctx_rows == e.a.ctx_rows && e.g.cache_len == e.a.cache_len,
              "geometry and acceptance must share cache/row counters");
  const at::ScalarType types[] = {at::kBFloat16, at::kInt, at::kInt, at::kLong,
      at::kInt, at::kInt, at::kLong, at::kLong, at::kByte, at::kLong, at::kInt};
  const std::int64_t counts[] = {kRows*kHidden, 1, 1, 1, 1, kRows, 2, 10,
      sizeof(naivert_runtime_speculative::DecodeControlState), 1, 1};
  for (int i = 0; i < 11; ++i) check_tensor(state[i], types[i], counts[i], device, "epilogue state");
  e.context = reinterpret_cast<const __nv_bfloat16*>(state[0].data_ptr());
  e.pos = state[1].data_ptr<std::int32_t>();
  e.generation = state[2].data_ptr<std::int32_t>();
  e.epoch = state[3].data_ptr<std::int64_t>();
  e.arrivals = reinterpret_cast<unsigned*>(state[4].data_ptr());
  e.a.sampling_state = state[6].data_ptr<std::int64_t>();
  e.a.verify_counts = state[7].data_ptr<std::int64_t>();
  e.watchdog = reinterpret_cast<unsigned*>(state[10].data_ptr());
  e.rank = rank;
  if (staged) {
  for (int i=11; i<13; ++i) check_tensor(state[i],at::kBFloat16,5*kRows*128,device,"staged KV");
  for (int i=13; i<15; ++i) check_tensor(state[i],at::kBFloat16,5*(max_seq_len+1)*128,device,"cache KV");
  e.staged_k=reinterpret_cast<const __nv_bfloat16*>(state[11].data_ptr());
  e.staged_v=reinterpret_cast<const __nv_bfloat16*>(state[12].data_ptr());
  e.cache_k=reinterpret_cast<__nv_bfloat16*>(state[13].data_ptr());
  e.cache_v=reinterpret_cast<__nv_bfloat16*>(state[14].data_ptr());
  e.cache_rows=max_seq_len+1;
  }
  // Repeat-penalty counts: optional, so the pinned reference tail (which predates them) keeps its
  // positional signature; absent means the commit leaves the counts alone.
  if (rep_count) {
    check_tensor(*rep_count, at::kInt, 152576, device, "rep_count");
    e.commit.rep_count = rep_count->data_ptr<std::int32_t>();
  }
  for (int r = 0; r < 8; ++r) {
    TORCH_CHECK(flags[r], "null sampler peer");
    e.flags[r] = reinterpret_cast<unsigned long long*>(flags[r]);
  }
  auto& c = e.commit;
  c.control = reinterpret_cast<naivert_runtime_speculative::DecodeControlState*>(state[8].data_ptr());
  c.token_out = state[5].data_ptr<std::int32_t>();
  c.previous_complete_ns = reinterpret_cast<std::uint64_t*>(state[9].data_ptr());
  TORCH_CHECK(capacity > 0 && cancel.device().is_cpu() && cancel.is_pinned() &&
              cancel.is_contiguous() && cancel.scalar_type() == at::kInt && cancel.numel() == 2,
              "invalid commit capacity/cancel");
  c.publish = rank == 0;
  if (c.publish) {
    TORCH_CHECK(host_ring.device().is_cpu() && host_ring.is_pinned() && host_ring.is_contiguous() &&
                host_ring.scalar_type() == at::kByte &&
                host_ring.numel() >= capacity * sizeof(naivert_runtime_persistent_sequencer::SequencerOutputRecord) &&
                host_flags.device().is_cpu() && host_flags.is_pinned() && host_flags.is_contiguous() &&
                host_flags.scalar_type() == at::kInt && host_flags.numel() >= capacity, "invalid output ring");
    c.host_output_ring = reinterpret_cast<naivert_runtime_persistent_sequencer::SequencerOutputRecord*>(host_ring.data_ptr());
    c.host_flags_ring = reinterpret_cast<std::uint32_t*>(host_flags.data_ptr());
  }
  c.output_ring_capacity = capacity;
  c.cancel = reinterpret_cast<std::uint32_t*>(cancel.data_ptr());
  c.accepted_count = reinterpret_cast<const std::uint32_t*>(e.a.accepted_count);
  c.token_row_stride = sizeof(std::int32_t);
  c.token_out_rows = kRows;
  auto stream = at::cuda::getCurrentCUDAStream(device);
  C10_CUDA_CHECK(launch_tail(e, stream, chain, staged));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef ASSIGN
}  // namespace naivert_dspark_tail
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace naivert_dspark_tail;
  m.attr("geometry_fields")=geometry_fields;
  m.attr("accept_fields")=accept_fields;
  m.def("geometry",&geometry);
  m.def("accept",&accept,pybind11::arg("values"),pybind11::arg("mask_token"),
        pybind11::arg("sampling_state")=std::nullopt,pybind11::arg("verify_counts")=std::nullopt);
  m.def("epilogue",&epilogue,pybind11::arg("geometry_values"),pybind11::arg("accept_values"),pybind11::arg("max_seq_len"),
        pybind11::arg("mask_token"),pybind11::arg("state"),pybind11::arg("flags"),pybind11::arg("host_ring"),
        pybind11::arg("host_flags"),pybind11::arg("cancel"),pybind11::arg("capacity"),pybind11::arg("rank"),
        pybind11::arg("chain"),pybind11::arg("staged"),pybind11::arg("rep_count")=std::nullopt);
  m.def("argument_bytes",[]{return std::vector<int>{sizeof(GeometryArgs),sizeof(AcceptArgs),sizeof(EpilogueArgs)};});
  m.def("layouts", &layouts);
}
