#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace naivert_torch_extension {

inline void require_cuda_contiguous(
    const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void require_rank_local(
    const at::Tensor& tensor, int device, const char* name) {
  require_cuda_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.get_device() == device, name, " must be rank-local");
}

inline void require_cuda_contiguous(
    const at::Tensor& tensor,
    at::ScalarType dtype,
    int device,
    const char* name) {
  require_cuda_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(
      tensor.get_device() == device, name, " must be rank-local");
}

inline at::Tensor alias_device_pointer(
    std::uint64_t address,
    std::int64_t elements,
    const std::string& dtype,
    std::int64_t device,
    bool allow_float32) {
  TORCH_CHECK(address != 0, "device alias pointer must be nonzero");
  TORCH_CHECK(elements > 0, "device alias must contain elements");
  TORCH_CHECK(
      device >= 0 && device < 8,
      "device alias rank must be inside TP8");
  at::ScalarType scalar_type = at::kByte;
  if (dtype == "uint8") {
    scalar_type = at::kByte;
  } else if (dtype == "int32") {
    scalar_type = at::kInt;
  } else if (dtype == "bfloat16") {
    scalar_type = at::kBFloat16;
  } else if (allow_float32 && dtype == "float32") {
    scalar_type = at::kFloat;
  } else {
    TORCH_CHECK(false, "unsupported device alias dtype");
  }
  c10::cuda::CUDAGuard guard(
      c10::Device(c10::DeviceType::CUDA, device));
  return at::from_blob(
      reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(address)),
      {elements},
      [](void*) {},
      at::TensorOptions()
          .dtype(scalar_type)
          .device(c10::DeviceType::CUDA, device));
}

}  // namespace naivert_torch_extension
