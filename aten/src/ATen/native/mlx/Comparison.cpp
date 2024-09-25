#include <ATen/native/mlx/Comparison.h>
#include <ATen/native/mlx/Convert.h>
#include <ATen/ScalarOps.h>
#include <mlx/ops.h>
#include <ATen/Scalar.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/stack.h>
#include "c10/core/Allocator.h"
#endif

namespace at::native {
TORCH_IMPL_FUNC(eq_tensor_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  eq_out_mlx_impl(self, mat2, result);
}

static Tensor wrapped_scalar_tensor_mlx(const Scalar& scalar, const Device device) {
  // Copied and modified from aten/stc/ATen/ScalarOps.h
  // as MPS doesn't support float64 tensor.
  Tensor tensor;
  if (scalar.isFloatingPoint()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kFloat));
  } else if (scalar.isBoolean()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kBool));
  } else if (scalar.isComplex()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kComplexDouble));
  } else {
    TORCH_INTERNAL_ASSERT(scalar.isIntegral(false));
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kLong));
  }
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

TORCH_IMPL_FUNC(ne_scalar_out_mlx)(const Tensor & self, const Scalar & other, const Tensor & result) {
  std::cout << "The type is: " << (int)other.type() << std::endl;
  Tensor mat2 = wrapped_scalar_tensor_mlx(other, DeviceType::MLX);

  auto sizes = self.sizes();
  for (uint64_t item: sizes) {
    std::cout << "Ne sizes: " << item << std::endl;
  }
  ne_out_mlx_impl(self, mat2, result);
}

TORCH_IMPL_FUNC(ne_tensor_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ne_out_mlx_impl(self, mat2, result);
}

void eq_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array mat2_mlx = mlx::convert::tensor_to_mlx(mat2);

  auto sizes = self.sizes();
  for (uint64_t item: sizes) {
    std::cout << "Eq input sizes: " << item << std::endl;
  }
  ::mlx::core::array result_mlx = ::mlx::core::equal(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();
  auto out_sizes = result.sizes();
  for (uint64_t item: sizes) {
    std::cout << "Eq result sizes: " << item << std::endl;
  }
  mlx::convert::set_tensor_result(result_mlx, result);
}

void ne_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array mat2_mlx = mlx::convert::tensor_to_mlx(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::not_equal(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();
  mlx::convert::set_tensor_result(result_mlx, result);
}

Tensor & abs_out_mlx(const Tensor & self, Tensor & output) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);

  ::mlx::core::array result_mlx = ::mlx::core::abs(self_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();
  mlx::convert::set_tensor_result(result_mlx, output);

  if (!output.is_same_size(self)) {
    if (self.is_contiguous()) {
      output.unsafeGetTensorImpl()->set_sizes_contiguous(self.sizes());
    } else {
      output.unsafeGetTensorImpl()->set_sizes_and_strides(self.sizes(), self.strides());
    }
  }

  auto sizes = output.sizes();
  return output;
}

Tensor & fill_scalar_mlx(Tensor &self, const Scalar &value) {
  if (self.numel() == 0) {
    return self;
  }

  // TODO: This is repeated code
  std::vector<int> mlx_shape;
  auto self_sizes = self.sizes();
  mlx_shape.resize(self_sizes.size());
  // TODO: Can this be optimized?
  for (size_t i=0; i<self_sizes.size(); i++) {
    std::cout << "Fill size: " << self_sizes[i] << std::endl;
    mlx_shape[i] = static_cast<int>(self_sizes[i]);
  }

  ::mlx::core::array result = {};
  if (value.isFloatingPoint()) {
    float32_t val = value.toFloat();
    std::cout << "The float is: " << val << std::endl;
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::float32);
  } else if (value.isBoolean()) {
    bool val = value.toBool();
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::bool_);
  } else if (value.isComplex()) {
    c10::complex<float> c10_val = value.toComplexFloat();
    std::complex<float> val = {c10_val.real(), c10_val.imag()};
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::complex64);
  } else {
    TORCH_INTERNAL_ASSERT(value.isIntegral(false));
    uint64_t val = value.toUInt64();
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::uint64);
  }

  std::cout << "Calling eval" << std::endl;
  result.eval();
  mlx::convert::set_tensor_result(result, self);
  std::cout << "Finished fill scalar" << std::endl;
  return self;
}

}
