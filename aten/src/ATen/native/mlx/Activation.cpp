#include <ATen/native/mlx/Activation.h>
#include <ATen/native/mlx/Convert.h>
#include <mlx/ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sigmoid_native.h>
#include <ATen/ops/sigmoid_backward_native.h>
#endif

namespace at::native {
Tensor relu_mlx(const Tensor& self) {
  ::mlx::core::array array = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array zero = ::mlx::core::array(0.0, mlx::convert::convert_type(self));
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

TORCH_IMPL_FUNC(sigmoid_backward_out_mlx)(const Tensor &grad_output, const Tensor &output, const Tensor &grad_input) {
  if (grad_output.numel() == 0) {
    return;
  }

  ::mlx::core::array output_mlx = mlx::convert::tensor_to_mlx(output);
  ::mlx::core::array grad_output_mlx = mlx::convert::tensor_to_mlx(grad_output);
  ::mlx::core::array unit_arr = ::mlx::core::array(1.0, mlx::convert::convert_type(grad_output));
  ::mlx::core::array one_minus_sigmoid = ::mlx::core::subtract(unit_arr, output_mlx, ::mlx::core::Device::gpu);
  ::mlx::core::array times_tensor = ::mlx::core::multiply(one_minus_sigmoid, output_mlx, ::mlx::core::Device::gpu);
  ::mlx::core::array grad_input_mlx = ::mlx::core::multiply(grad_output_mlx, times_tensor, ::mlx::core::Device::gpu);

  grad_input_mlx.eval();

  mlx::convert::set_tensor_result(grad_input_mlx, grad_input);
}

}
