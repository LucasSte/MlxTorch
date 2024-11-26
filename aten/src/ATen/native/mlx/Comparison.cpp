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
  Tensor mat2 = wrapped_scalar_tensor_mlx(other, DeviceType::MLX);
  ne_out_mlx_impl(self, mat2, result);
}

TORCH_IMPL_FUNC(ne_tensor_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ne_out_mlx_impl(self, mat2, result);
}

void eq_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& mat2_mlx = mlx::convert::retrieve_array(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::equal(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);

  mlx::convert::introduce_result(std::move(result_mlx), result);
}

void ne_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& mat2_mlx = mlx::convert::retrieve_array(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::not_equal(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);

  mlx::convert::introduce_result(std::move(result_mlx), result);
}

Tensor & abs_out_mlx(const Tensor & self, Tensor & output) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);

  ::mlx::core::array result_mlx = ::mlx::core::abs(self_mlx, ::mlx::core::Device::gpu);

  mlx::convert::introduce_result(std::move(result_mlx), output);

  if (!output.is_same_size(self)) {
    // TODO: Set storage size nbytes here!
    if (self.is_contiguous()) {
      output.unsafeGetTensorImpl()->set_sizes_contiguous(self.sizes());
    } else {
      output.unsafeGetTensorImpl()->set_sizes_and_strides(self.sizes(), self.strides());
    }
  }

  return output;
}

}
