// Native BF16 checkpoint linears. Shared tensor-core projection and TP mailboxes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "naivert/body/attn_common/bf16_linear_allreduce.cuh"

namespace naivert::checkpoint_bf16 {
using Tensor = at::Tensor;
using TailArgs = naivert::attn_unproj::UnprojTailArgs;
constexpr int H = 4096, I = 2048;

template<int Rows, int K>
__global__ __launch_bounds__(256, 1) void linear_kernel(
    const __grid_constant__ TailArgs args, const __nv_bfloat16* x,
    const __nv_bfloat16* residual) {
  extern __shared__ __align__(128) unsigned char shared[];
  naivert::attn_unproj::Bf16LinearAllreduce<Rows, H, K>::run(
      args, args.o_weight_bf16, x, residual, shared, int(blockIdx.x));
}

template<int Rows>
__global__ __launch_bounds__(256, 1) void swiglu_kernel(
    const __nv_bfloat16* x, const float* gamma, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* out) {
  extern __shared__ __align__(128) unsigned char shared[];
  auto* norm = reinterpret_cast<__nv_bfloat16*>(shared);
  auto* partial = reinterpret_cast<float*>(shared + Rows * H * 2);
  auto* projected = partial + 8 * 128;
  naivert::attn_rows::normalize_rows<Rows, H>(x, gamma, norm, 1e-5f);
  auto row_ptr = [&](int row) {
    const int index = (row / 32) * 16 + row % 16;
    return (row % 32 < 16 ? gate : up) + std::size_t(index) * H;
  };
  auto emit = [&](int row, int query, float value) { projected[(row % 32) * Rows + query] = value; };
  naivert::attn_rows::project_rows_mma<Rows, H, 4, 2, 8, 128>(0, I * 2 / 16, row_ptr, norm, partial, emit);
  for (int item = int(threadIdx.x); item < 16 * Rows; item += int(blockDim.x)) {
    const int row = item / Rows, query = item % Rows;
    const float g = projected[row * Rows + query];
    const float u = projected[(row + 16) * Rows + query];
    out[query * I + int(blockIdx.x) * 16 + row] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
  }
}

void check(const Tensor& t, at::ScalarType type, int device, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == type &&
              t.get_device() == device, name, " must be contiguous, on the input CUDA device, and have the required dtype");
}

template<int Rows, int K>
void linear_launch(const TailArgs& args, const Tensor& x, const Tensor& residual, cudaStream_t stream) {
  constexpr int bytes = naivert::attn_unproj::Bf16LinearAllreduce<Rows, H, K>::kSharedBytes;
  C10_CUDA_CHECK(cudaFuncSetAttribute(linear_kernel<Rows, K>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  linear_kernel<Rows, K><<<H/64, 256, bytes, stream>>>(args,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()));
}

void linear(Tensor x, Tensor weight, Tensor residual, Tensor out,
            std::vector<Tensor> mailboxes, Tensor epoch, int64_t rank, int64_t tag) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be a CUDA matrix");
  const int device = x.get_device(), rows = x.size(0), k = x.size(1);
  c10::cuda::CUDAGuard guard(x.device());
  for (auto& t : {x, weight, residual, out}) check(t, at::kBFloat16, device, "BF16 linear tensor");
  check(epoch, at::kLong, device, "epoch");
  TORCH_CHECK((rows == 1 || rows == 8) && (k == 1024 || k == I), "supported shapes: rows 1/8, K 1024/2048");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({H,k}) && residual.numel() == rows*H && out.numel() == rows*H,
              "BF16 linear shapes do not match");
  TORCH_CHECK(mailboxes.size() == 8 && rank >= 0 && rank < 8 && tag > 0 && tag < 256 && epoch.numel() == 1,
              "invalid TP mailbox identity");
  TailArgs args{};
  args.o_weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  args.hidden_out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.replay_epoch = reinterpret_cast<const uint64_t*>(epoch.data_ptr());
  args.rank = int(rank); args.mailbox_tag = unsigned(tag);
  for (int peer=0; peer<8; ++peer) {
    TORCH_CHECK(mailboxes[peer].is_cuda() && mailboxes[peer].is_contiguous() && mailboxes[peer].scalar_type() == at::kByte &&
                mailboxes[peer].nbytes() >= rows * naivert::kMailboxSlabStride, "invalid peer mailbox");
    args.mailbox_peers[peer] = reinterpret_cast<const uint8_t*>(mailboxes[peer].data_ptr());
  }
  auto stream = at::cuda::getCurrentCUDAStream(device).stream();
#define LAUNCH(R) if (k == 1024) linear_launch<R,1024>(args,x,residual,stream); else linear_launch<R,I>(args,x,residual,stream)
  if (rows==1) { LAUNCH(1); } else { LAUNCH(8); }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Rows>
void swiglu_launch(const Tensor& x, const Tensor& gamma, const Tensor& gate, const Tensor& up, const Tensor& out,
                   cudaStream_t stream) {
  constexpr int bytes = Rows * H * 2 + 8*128*4 + 32*Rows*4;
  C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_kernel<Rows>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  swiglu_kernel<Rows><<<I/16,256,bytes,stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      gamma.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
}
void swiglu(Tensor x, Tensor gamma, Tensor gate, Tensor up, Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.size(1)==H, "x must be [rows,4096]");
  const int device=x.get_device(), rows=x.size(0);
  c10::cuda::CUDAGuard guard(x.device());
  for (auto& t : {x,gate,up,out}) check(t,at::kBFloat16,device,"BF16 SwiGLU tensor");
  check(gamma,at::kFloat,device,"gamma");
  TORCH_CHECK(gate.sizes()==at::IntArrayRef({I,H}) && up.sizes()==gate.sizes() && gamma.numel()==H && out.numel()==rows*I,
              "BF16 SwiGLU shapes do not match");
  TORCH_CHECK(rows==1 || rows==8,"supported row counts are 1/8");
  auto stream=at::cuda::getCurrentCUDAStream(device).stream();
  if (rows==1) swiglu_launch<1>(x,gamma,gate,up,out,stream);
  else swiglu_launch<8>(x,gamma,gate,up,out,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void bind(pybind11::module_& m) {
  m.def("launch_checkpoint_bf16_linear", &linear);
  m.def("launch_checkpoint_bf16_swiglu", &swiglu);
}
}  // namespace naivert::checkpoint_bf16
