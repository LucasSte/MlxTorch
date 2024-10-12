#include <ATen/native/mlx/Activation.h>
#include <ATen/native/mlx/Convert.h>
#include <mlx/ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sigmoid_native.h>
#endif

namespace at::native {
Tensor relu_mlx(const Tensor& self) {
  ::mlx::core::array array = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array zero = ::mlx::core::array(0.0);
  ::mlx::core::array result = ::mlx::core::maximum(array, zero, ::mlx::core::Device::gpu);
  result.eval();

  return mlx::convert::new_from_mlx(result);
}

TORCH_IMPL_FUNC(sigmoid_out_mlx)(const Tensor& self, const Tensor& output) {
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array result_mlx = ::mlx::core::sigmoid(self_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::set_tensor_result(result_mlx, output);
}

}
