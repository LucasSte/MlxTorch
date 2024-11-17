#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorMeta.h>
#include <ATen/Scalar.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <ATen/native/mlx/Convert.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/round_native.h>
#endif


namespace at::native {
TORCH_IMPL_FUNC(sqrt_out_mlx)(const Tensor& self, const Tensor& output) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array result = ::mlx::core::sqrt(self_mlx, ::mlx::core::Device::gpu);
  result.eval();
  mlx::convert::introduce_result(result, output);
}

TORCH_IMPL_FUNC(round_out_mlx)(const Tensor& self, const Tensor& output) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array result = ::mlx::core::round(self_mlx, 0, ::mlx::core::Device::gpu);
  result.eval();
  mlx::convert::introduce_result(result, output);
}
}
