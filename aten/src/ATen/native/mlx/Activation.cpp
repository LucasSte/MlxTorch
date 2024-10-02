#include <ATen/native/mlx/Activation.h>
#include <ATen/native/mlx/Convert.h>
#include <mlx/ops.h>

namespace at::native {
Tensor relu_mlx(const Tensor& self) {
  ::mlx::core::array array = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array zero = ::mlx::core::array(0.0);
  ::mlx::core::array result = ::mlx::core::maximum(array, zero, ::mlx::core::Device::gpu);
  result.eval();

  return mlx::convert::new_from_mlx(result);
}
}
