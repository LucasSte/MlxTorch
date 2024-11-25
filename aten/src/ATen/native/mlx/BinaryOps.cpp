#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/BinaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/lerp_native.h>
#include <ATen/ops/result_type.h>
#endif

#include <mlx/array.h>
#include <ATen/native/mlx/Convert.h>
#include <mlx/ops.h>

namespace at::native {
TORCH_IMPL_FUNC(lerp_scalar_mlx)(const Tensor& self, const Tensor& end, const Scalar& weight, const Tensor& out) {
  if (weight.toDouble() == 0) {
    if (!self.is_alias_of(out)) {
      out.copy_(self);
    }
  }

  const bool weight_has_value = weight.toDouble() != 1.0;
  if (weight_has_value) {
    auto commonDtype = at::result_type(self, end);
    at::native::alpha_check(commonDtype, weight);
  }

  if (!weight_has_value) {
    if (!self.is_alias_of(out)) { // if inplace, no-op
      out.copy_(end);
    }
    return;
  }
  ::mlx::core::array& start_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& end_mlx = mlx::convert::retrieve_array(end);
  // TODO: Not only float is allowed here!
  ::mlx::core::array weight_mlx = mlx::convert::scalar_to_mlx(weight);

  ::mlx::core::array sub = ::mlx::core::subtract(end_mlx, start_mlx, ::mlx::core::Device::gpu);
  ::mlx::core::array mult = ::mlx::core::multiply(sub, weight_mlx, ::mlx::core::Device::gpu);
  ::mlx::core::array out_mlx = ::mlx::core::add(start_mlx, mult);

  out_mlx.eval();
  mlx::convert::introduce_result(out_mlx, out);
}

TORCH_IMPL_FUNC(div_out_mlx)(const Tensor& self, const Tensor& other, const Tensor& output) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& other_mlx = mlx::convert::retrieve_array(other);
  ::mlx::core::array result_mlx = ::mlx::core::divide(self_mlx, other_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, output);
}

TORCH_IMPL_FUNC(add_out_mlx)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  ::mlx::core::array& other_mlx = mlx::convert::retrieve_array(other);
  ::mlx::core::array alpha_mlx = mlx::convert::scalar_to_mlx(alpha);

  ::mlx::core::array mul = ::mlx::core::multiply(other_mlx, alpha_mlx, ::mlx::core::Device::gpu);
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array result_mlx = ::mlx::core::add(self_mlx, mul, ::mlx::core::Device::gpu);
  result_mlx.eval();
  mlx::convert::introduce_result(result_mlx, output);
}

}
