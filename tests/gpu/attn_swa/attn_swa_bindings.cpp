// Torch bindings for the standalone attn_swa check (tests/gpu/attn_swa).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/extension.h>

#include "naivert/body/attn_swa/attn_swa_args.hpp"

namespace naivert::attn_swa {
cudaError_t launch_attn_swa_rows1(const AttnSwaArgs& args, int grid,
                                  cudaStream_t stream);
}

namespace {

using naivert::attn_swa::AttnSwaArgs;
using Scratch = naivert::attn_swa::AttnSwaScratchLayout;

template <class T>
T* pointer(const torch::Tensor& tensor, const char* name,
           c10::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  return reinterpret_cast<T*>(tensor.data_ptr());
}

int64_t workspace_bytes() { return int64_t(Scratch::kBytes); }
int64_t window_rows() { return int64_t(Scratch::kWindowRows); }

// Tensor maps refer to the original row-major BF16 matrices. The view permutes
// the row and K-block dimensions into StagedTile's shared-memory order.
void prepare_maps(torch::Tensor workspace, torch::Tensor wq, torch::Tensor wk, torch::Tensor wv) {
  if constexpr (!Scratch::kUseProjectionTma) return;
  static_assert(sizeof(CUtensorMap) == Scratch::kProjectionMapBytes);
  auto* out = pointer<std::uint8_t>(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(workspace.numel() >= int64_t(Scratch::kBytes), "workspace too small");
  const c10::cuda::CUDAGuard guard(workspace.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  TORCH_CHECK(cudaStreamIsCapturing(stream.stream(), &capture) == cudaSuccess &&
      capture == cudaStreamCaptureStatusNone, "prepare_maps must run before graph capture");
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
      const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
      CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult query;
  TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
      cudaEnableDefault, &query) == cudaSuccess && fn != nullptr, "tensor map API unavailable");
  alignas(64) CUtensorMap maps[Scratch::kProjectionMapCount]{};
  torch::Tensor weights[Scratch::kProjectionMapCount] = {wq, wk, wv};
  const int rows[Scratch::kProjectionMapCount] = {Scratch::kQ, Scratch::kHeadDim, Scratch::kVDim};
  for (int i = 0; i < Scratch::kProjectionMapCount; ++i) {
    auto* source = pointer<__nv_bfloat16>(weights[i], "weight", torch::kBFloat16);
    TORCH_CHECK(weights[i].device() == workspace.device(), "weights and workspace must share a device");
    TORCH_CHECK(weights[i].dim() == 2 && weights[i].size(0) == rows[i] &&
        weights[i].size(1) == Scratch::kDim, "weight geometry mismatch");
    const cuuint64_t dims[4] = {64,16,64,cuuint64_t(rows[i]/16)};
    const cuuint64_t strides[3] = {8192,128,131072};
    const cuuint32_t box[4] = {64,16,64,1}, step[4] = {1,1,1,1};
    const auto result = reinterpret_cast<Encode>(fn)(maps+i, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4, source, dims, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "projection tensor-map encode failed: ", int(result));
  }
  TORCH_CHECK(cudaMemcpyAsync(out + Scratch::kProjectionMapsOffset, maps, sizeof(maps),
      cudaMemcpyHostToDevice, stream.stream()) == cudaSuccess, "tensor-map copy failed");
  TORCH_CHECK(cudaStreamSynchronize(stream.stream()) == cudaSuccess, "tensor-map preparation failed");
}

py::dict layout() {
  py::dict out;
  out["fused"] = std::make_pair(int64_t(Scratch::kFusedOffset), int64_t(Scratch::kFusedBytes));
  out["q_rot"] = std::make_pair(int64_t(Scratch::kQRotOffset), int64_t(Scratch::kQRotBytes));
  out["fused_bases"] = py::make_tuple(Scratch::kQBase, Scratch::kKBase, Scratch::kVBase, Scratch::kFused);
  out["window"] = int64_t(Scratch::kWindow);
  return out;
}

struct TailSpec {
  torch::Tensor o_weight;
  torch::Tensor o_scales;
  torch::Tensor hidden_out;
  std::vector<torch::Tensor> mailboxes;
  torch::Tensor epoch;
  int64_t rank;
  int64_t tag;
};

void run_impl(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq,
              torch::Tensor wk, torch::Tensor wv, torch::Tensor rope_freqs,
              torch::Tensor sink_bias, torch::Tensor k_window, torch::Tensor v_window,
              torch::Tensor proj_o, torch::Tensor workspace, torch::Tensor cur_pos,
              int64_t max_seq_len, int64_t grid, const TailSpec* tail, float value_scale = 1.0f) {
  AttnSwaArgs args{};
  args.value_scale = value_scale;
  args.residual = pointer<const __nv_bfloat16>(residual, "residual", torch::kBFloat16);
  args.input_norm = pointer<const float>(input_norm, "input_norm", torch::kFloat32);
  args.wq = pointer<const __nv_bfloat16>(wq, "wq", torch::kBFloat16);
  args.wk = pointer<const __nv_bfloat16>(wk, "wk", torch::kBFloat16);
  args.wv = pointer<const __nv_bfloat16>(wv, "wv", torch::kBFloat16);
  args.rope_freqs = pointer<const float>(rope_freqs, "rope_freqs", torch::kFloat32);
  args.sink_bias = pointer<const float>(sink_bias, "sink_bias", torch::kFloat32);
  args.k_window = pointer<__nv_bfloat16>(k_window, "k_window", torch::kBFloat16);
  args.v_window = pointer<__nv_bfloat16>(v_window, "v_window", torch::kBFloat16);
  args.proj_o = pointer<__nv_bfloat16>(proj_o, "proj_o", torch::kBFloat16);
  args.workspace = pointer<std::uint8_t>(workspace, "workspace", torch::kUInt8);
  args.cur_pos = pointer<const int>(cur_pos, "cur_pos", torch::kInt32);
  TORCH_CHECK(workspace.numel() >= int64_t(Scratch::kBytes), "workspace too small");
  TORCH_CHECK(k_window.size(0) >= Scratch::kWindow && v_window.size(0) >= Scratch::kWindow,
              "window caches need at least kWindow rows");
  TORCH_CHECK(rope_freqs.size(0) >= max_seq_len, "rope table shorter than max_seq_len");
  TORCH_CHECK(sink_bias.numel() >= Scratch::kLocalHeads, "sink_bias needs one value per local head");
  args.max_seq_len = int(max_seq_len);
  int device_grid = int(grid);
  if (device_grid <= 0) {
    // cudaDeviceGetAttribute, not torch's cudaDeviceProp: its layout differs between the CUDA
    // 12.x build of torch and the CUDA 13.x headers these bindings compile against.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&device_grid, cudaDevAttrMultiProcessorCount, device);
  }
  if (tail != nullptr) {
    TORCH_CHECK(tail->mailboxes.size() == 8, "eight mailbox tensors (one per rank)");
    if (tail->o_weight.scalar_type() == torch::kBFloat16) {
      TORCH_CHECK(tail->o_weight.dim() == 2 && tail->o_weight.size(0) == Scratch::kDim &&
                  tail->o_weight.size(1) == Scratch::kLocalHeads * Scratch::kVDim,
                  "BF16 o_weight must be [hidden, local attention width]");
      args.unproj.o_weight_bf16 = pointer<const __nv_bfloat16>(tail->o_weight, "o_weight", torch::kBFloat16);
      args.unproj.o_weight = reinterpret_cast<const __nv_fp8_e4m3*>(args.unproj.o_weight_bf16);
    } else {
    args.unproj.o_weight = pointer<const __nv_fp8_e4m3>(tail->o_weight, "o_weight", torch::kFloat8_e4m3fn);
    }
    args.unproj.o_scales = pointer<const __nv_bfloat16>(tail->o_scales, "o_scales", torch::kBFloat16);
    args.unproj.hidden_out = pointer<__nv_bfloat16>(tail->hidden_out, "hidden_out", torch::kBFloat16);
    for (int i = 0; i < 8; ++i) {
      args.unproj.mailbox_peers[i] = pointer<const std::uint8_t>(tail->mailboxes[i], "mailbox", torch::kUInt8);
    }
    args.unproj.replay_epoch = tail->epoch.numel() > 0
        ? pointer<const std::uint64_t>(tail->epoch, "epoch", torch::kInt64) : nullptr;
    args.unproj.rank = int(tail->rank);
    args.unproj.mailbox_tag = std::uint32_t(tail->tag);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = naivert::attn_swa::launch_attn_swa_rows1(args, device_grid, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "attn_swa launch failed: ", cudaGetErrorString(err));
}

void run(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
         torch::Tensor wv, torch::Tensor rope_freqs, torch::Tensor sink_bias, torch::Tensor k_window,
         torch::Tensor v_window, torch::Tensor proj_o, torch::Tensor workspace, torch::Tensor cur_pos,
         int64_t max_seq_len, int64_t grid) {
  run_impl(residual, input_norm, wq, wk, wv, rope_freqs, sink_bias, k_window, v_window, proj_o,
           workspace, cur_pos, max_seq_len, grid, nullptr);
}

// Attention + fused o-projection / TP8 all-reduce: hidden_out = residual + allreduce(o_proj(attn)).
void run_fused(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
               torch::Tensor wv, torch::Tensor rope_freqs, torch::Tensor sink_bias,
               torch::Tensor k_window, torch::Tensor v_window, torch::Tensor proj_o,
               torch::Tensor workspace, torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid,
               torch::Tensor o_weight, torch::Tensor o_scales, torch::Tensor hidden_out,
               std::vector<torch::Tensor> mailboxes, torch::Tensor epoch, int64_t rank, int64_t tag) {
  TailSpec tail{o_weight, o_scales, hidden_out, std::move(mailboxes), epoch, rank, tag};
  run_impl(residual, input_norm, wq, wk, wv, rope_freqs, sink_bias, k_window, v_window, proj_o,
           workspace, cur_pos, max_seq_len, grid, &tail);
}

void run_scaled(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
         torch::Tensor wv, torch::Tensor rope_freqs, torch::Tensor sink_bias, torch::Tensor k_window,
         torch::Tensor v_window, torch::Tensor proj_o, torch::Tensor workspace, torch::Tensor cur_pos,
         int64_t max_seq_len, int64_t grid, double value_scale) {
  run_impl(residual, input_norm, wq, wk, wv, rope_freqs, sink_bias, k_window, v_window, proj_o,
           workspace, cur_pos, max_seq_len, grid, nullptr, float(value_scale));
}

// Attention + fused o-projection / TP8 all-reduce: hidden_out = residual + allreduce(o_proj(attn)).
void run_fused_scaled(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
               torch::Tensor wv, torch::Tensor rope_freqs, torch::Tensor sink_bias,
               torch::Tensor k_window, torch::Tensor v_window, torch::Tensor proj_o,
               torch::Tensor workspace, torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid,
               torch::Tensor o_weight, torch::Tensor o_scales, torch::Tensor hidden_out,
               std::vector<torch::Tensor> mailboxes, torch::Tensor epoch, int64_t rank, int64_t tag, double value_scale) {
  TailSpec tail{o_weight, o_scales, hidden_out, std::move(mailboxes), epoch, rank, tag};
  run_impl(residual, input_norm, wq, wk, wv, rope_freqs, sink_bias, k_window, v_window, proj_o,
           workspace, cur_pos, max_seq_len, grid, &tail, float(value_scale));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &workspace_bytes);
  m.def("prepare_maps", &prepare_maps);
  m.def("window_rows", &window_rows);
  m.def("layout", &layout);
  m.def("run", &run);
  m.def("run_scaled", &run_scaled);
  m.def("run_fused_scaled", &run_fused_scaled);
  m.def("run_fused", &run_fused);
}
