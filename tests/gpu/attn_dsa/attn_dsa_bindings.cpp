// Torch bindings for the standalone attn_dsa check (tests/gpu/attn_dsa).
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "naivert/body/attn_dsa/attn_dsa_args.hpp"

namespace naivert::attn_dsa {
cudaError_t launch_attn_dsa_rows1(const AttnDsaArgs& args, int grid,
                                  cudaStream_t stream);
}

namespace {

using naivert::attn_dsa::AttnDsaArgs;
using Scratch = naivert::attn_dsa::AttnDsaScratchLayout;

template <class T>
T* pointer(const torch::Tensor& tensor, const char* name,
           c10::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  return reinterpret_cast<T*>(tensor.data_ptr());
}

int64_t workspace_bytes() { return int64_t(Scratch::kBytes); }

// Describe the natural BF16 matrices in the staged projection's K-block/row/K-inner order.
// Descriptors and their weight/workspace pointers must remain valid throughout replay.
void prepare_maps(torch::Tensor workspace, torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wik) {
  if constexpr (!Scratch::kUseProjectionTma) return;
  static_assert(sizeof(CUtensorMap) == Scratch::kProjectionMapBytes);
  auto* out = pointer<std::uint8_t>(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(workspace.numel() >= workspace_bytes(), "workspace too small");
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
  torch::Tensor weights[Scratch::kProjectionMapCount] = {wq, wk, wv, wik};
  const int rows[Scratch::kProjectionMapCount] = {Scratch::kQ, Scratch::kHeadDim, Scratch::kVDim, Scratch::kIndexDim};
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
  out["index_heads"] = Scratch::kIndexHeads;
  out["query_rows"] = Scratch::kQueryRows;
  out["fused"] = std::make_pair(int64_t(Scratch::kFusedOffset), int64_t(Scratch::kFusedBytes));
  out["q_rot"] = std::make_pair(int64_t(Scratch::kQRotOffset), int64_t(Scratch::kQRotBytes));
  out["iq_packed"] = std::make_pair(int64_t(Scratch::kIqPackedOffset), int64_t(Scratch::kIqPackedBytes));
  out["index_weights"] = std::make_pair(int64_t(Scratch::kIndexWeightsOffset), int64_t(Scratch::kIndexWeightsBytes));
  out["index_weights_scaled"] = std::make_pair(int64_t(Scratch::kIndexWeightsScaledOffset), int64_t(Scratch::kIndexWeightsScaledBytes));
  out["logits"] = std::make_pair(int64_t(Scratch::kIndexLogitsOffset), int64_t(Scratch::kIndexLogitsBytes));
  out["selections"] = std::make_pair(int64_t(Scratch::kIndexSelectsOffset), int64_t(Scratch::kIndexSelectsBytes));
  out["phase_timers"] = std::make_pair(
      int64_t(Scratch::kAttentionTempOffset + Scratch::kPhaseTimerFloatOffset * sizeof(float)),
      int64_t(Scratch::kPhaseTimerCount * sizeof(unsigned long long)));
  out["fused_bases"] = py::make_tuple(Scratch::kQBase, Scratch::kKBase, Scratch::kVBase,
                                      Scratch::kIkBase, Scratch::kIqBase, Scratch::kFused);
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
              torch::Tensor wk, torch::Tensor wv, torch::Tensor wiq, torch::Tensor wiq_scale, torch::Tensor wik,
              torch::Tensor ik_norm_weight, torch::Tensor ik_norm_bias, torch::Tensor wis,
              torch::Tensor rope_freqs, torch::Tensor k, torch::Tensor v, torch::Tensor ki,
              torch::Tensor ki_scale, torch::Tensor proj_o, torch::Tensor workspace,
              torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid, const TailSpec* tail, float value_scale = 1.0f) {
  AttnDsaArgs args{};
  args.value_scale = value_scale;
  args.residual = pointer<const __nv_bfloat16>(residual, "residual", torch::kBFloat16);
  args.input_norm = pointer<const float>(input_norm, "input_norm", torch::kFloat32);
  args.wq = pointer<const __nv_bfloat16>(wq, "wq", torch::kBFloat16);
  args.wk = pointer<const __nv_bfloat16>(wk, "wk", torch::kBFloat16);
  args.wv = pointer<const __nv_bfloat16>(wv, "wv", torch::kBFloat16);
  if (wiq.scalar_type() == torch::kBFloat16)
    args.wiq_bf16 = pointer<const __nv_bfloat16>(wiq, "wiq", torch::kBFloat16);
  else
  args.wiq = pointer<const __nv_fp8_e4m3>(wiq, "wiq", torch::kFloat8_e4m3fn);
  args.wiq_scale = pointer<const float>(wiq_scale, "wiq_scale", torch::kFloat32);
  TORCH_CHECK(wiq.dim() == 2 && wiq.size(0) == Scratch::kIqRows && wiq.size(1) == Scratch::kDim,
              "wiq must match the compiled index head count: ", Scratch::kIndexHeads);
  TORCH_CHECK(wiq_scale.dim() == 2 && wiq_scale.size(0) == Scratch::kIndexHeads && wiq_scale.size(1) == 32,
              "wiq_scale must be [index_heads, 32]");
  args.wik = pointer<const __nv_bfloat16>(wik, "wik", torch::kBFloat16);
  args.ik_norm_weight = pointer<const float>(ik_norm_weight, "ik_norm_weight", torch::kFloat32);
  args.ik_norm_bias = pointer<const float>(ik_norm_bias, "ik_norm_bias", torch::kFloat32);
  args.wis = pointer<const __nv_bfloat16>(wis, "wis", torch::kBFloat16);
  TORCH_CHECK(wis.dim() == 2 && wis.size(0) == Scratch::kIndexHeads && wis.size(1) == Scratch::kDim,
              "wis must be [index_heads, 4096]");
  TORCH_CHECK(residual.numel() == Scratch::kQueryRows * Scratch::kDim &&
              proj_o.numel() == Scratch::kQueryRows * Scratch::kLocalHeads * Scratch::kVDim,
              "input and output rows must match the compiled query row count");
  args.rope_freqs = pointer<const float>(rope_freqs, "rope_freqs", torch::kFloat32);
  args.k = pointer<__nv_bfloat16>(k, "k", torch::kBFloat16);
  args.v = pointer<__nv_bfloat16>(v, "v", torch::kBFloat16);
  args.ki = pointer<__nv_fp8_e4m3>(ki, "ki", torch::kFloat8_e4m3fn);
  args.ki_scale = pointer<float>(ki_scale, "ki_scale", torch::kFloat32);
  TORCH_CHECK(ki_scale.size(0) == max_seq_len, "ki_scale needs max_seq_len entries");
  args.proj_o = pointer<__nv_bfloat16>(proj_o, "proj_o", torch::kBFloat16);
  args.workspace = pointer<std::uint8_t>(workspace, "workspace", torch::kUInt8);
  args.cur_pos = pointer<const int>(cur_pos, "cur_pos", torch::kInt32);
  TORCH_CHECK(workspace.numel() >= int64_t(Scratch::kBytes), "workspace too small");
  TORCH_CHECK(k.size(0) == max_seq_len && v.size(0) == max_seq_len && ki.size(0) == max_seq_len,
              "caches must have max_seq_len rows");
  TORCH_CHECK(rope_freqs.size(0) >= max_seq_len, "rope table shorter than max_seq_len");
  args.max_seq_len = int(max_seq_len);
  args.attention_scale_multiplier = 0.0f;
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
  cudaError_t err = naivert::attn_dsa::launch_attn_dsa_rows1(args, device_grid, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "attn_dsa launch failed: ", cudaGetErrorString(err));
}

void run(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
         torch::Tensor wv, torch::Tensor wiq, torch::Tensor wiq_scale, torch::Tensor wik, torch::Tensor ik_norm_weight,
         torch::Tensor ik_norm_bias, torch::Tensor wis, torch::Tensor rope_freqs, torch::Tensor k,
         torch::Tensor v, torch::Tensor ki, torch::Tensor ki_scale, torch::Tensor proj_o,
         torch::Tensor workspace,
         torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid) {
  run_impl(residual, input_norm, wq, wk, wv, wiq, wiq_scale, wik, ik_norm_weight, ik_norm_bias, wis, rope_freqs,
           k, v, ki, ki_scale, proj_o, workspace, cur_pos, max_seq_len, grid, nullptr);
}

// Attention + fused o-projection / TP8 all-reduce: hidden_out = residual + allreduce(o_proj(attn)).
void run_fused(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
               torch::Tensor wv, torch::Tensor wiq, torch::Tensor wiq_scale, torch::Tensor wik, torch::Tensor ik_norm_weight,
               torch::Tensor ik_norm_bias, torch::Tensor wis, torch::Tensor rope_freqs, torch::Tensor k,
               torch::Tensor v, torch::Tensor ki, torch::Tensor ki_scale, torch::Tensor proj_o,
         torch::Tensor workspace,
               torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid, torch::Tensor o_weight,
               torch::Tensor o_scales, torch::Tensor hidden_out, std::vector<torch::Tensor> mailboxes,
               torch::Tensor epoch, int64_t rank, int64_t tag) {
  TailSpec tail{o_weight, o_scales, hidden_out, std::move(mailboxes), epoch, rank, tag};
  run_impl(residual, input_norm, wq, wk, wv, wiq, wiq_scale, wik, ik_norm_weight, ik_norm_bias, wis, rope_freqs,
           k, v, ki, ki_scale, proj_o, workspace, cur_pos, max_seq_len, grid, &tail);
}

void run_scaled(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
         torch::Tensor wv, torch::Tensor wiq, torch::Tensor wiq_scale, torch::Tensor wik, torch::Tensor ik_norm_weight,
         torch::Tensor ik_norm_bias, torch::Tensor wis, torch::Tensor rope_freqs, torch::Tensor k,
         torch::Tensor v, torch::Tensor ki, torch::Tensor ki_scale, torch::Tensor proj_o,
         torch::Tensor workspace,
         torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid, double value_scale) {
  run_impl(residual, input_norm, wq, wk, wv, wiq, wiq_scale, wik, ik_norm_weight, ik_norm_bias, wis, rope_freqs,
           k, v, ki, ki_scale, proj_o, workspace, cur_pos, max_seq_len, grid, nullptr, float(value_scale));
}

// Attention + fused o-projection / TP8 all-reduce: hidden_out = residual + allreduce(o_proj(attn)).
void run_fused_scaled(torch::Tensor residual, torch::Tensor input_norm, torch::Tensor wq, torch::Tensor wk,
               torch::Tensor wv, torch::Tensor wiq, torch::Tensor wiq_scale, torch::Tensor wik, torch::Tensor ik_norm_weight,
               torch::Tensor ik_norm_bias, torch::Tensor wis, torch::Tensor rope_freqs, torch::Tensor k,
               torch::Tensor v, torch::Tensor ki, torch::Tensor ki_scale, torch::Tensor proj_o,
         torch::Tensor workspace,
               torch::Tensor cur_pos, int64_t max_seq_len, int64_t grid, torch::Tensor o_weight,
               torch::Tensor o_scales, torch::Tensor hidden_out, std::vector<torch::Tensor> mailboxes,
               torch::Tensor epoch, int64_t rank, int64_t tag, double value_scale) {
  TailSpec tail{o_weight, o_scales, hidden_out, std::move(mailboxes), epoch, rank, tag};
  run_impl(residual, input_norm, wq, wk, wv, wiq, wiq_scale, wik, ik_norm_weight, ik_norm_bias, wis, rope_freqs,
           k, v, ki, ki_scale, proj_o, workspace, cur_pos, max_seq_len, grid, &tail, float(value_scale));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &workspace_bytes);
  m.def("prepare_maps", &prepare_maps);
  m.def("layout", &layout);
  m.def("run", &run);
  m.def("run_scaled", &run_scaled);
  m.def("run_fused_scaled", &run_fused_scaled);
  m.def("run_fused", &run_fused);
}
