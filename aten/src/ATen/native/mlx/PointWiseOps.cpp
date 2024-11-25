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
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#endif

namespace at::native {
TORCH_IMPL_FUNC(addcmul_out_mlx)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& output) {

  ::mlx::core::array& t1 = mlx::convert::retrieve_array(tensor1);
  ::mlx::core::array& t2 = mlx::convert::retrieve_array(tensor2);
  ::mlx::core::array val = mlx::convert::scalar_to_mlx(value);

  ::mlx::core::array mul1 = ::mlx::core::multiply(t1, t2, ::mlx::core::Device::gpu);
  ::mlx::core::array mul2 = ::mlx::core::multiply(val, mul1, ::mlx::core::Device::gpu);
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);

  ::mlx::core::array result_mlx = ::mlx::core::add(self_mlx, mul2, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, output);
}

TORCH_IMPL_FUNC(addcdiv_out_mlx)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& output) {
  ::mlx::core::array& t1 = mlx::convert::retrieve_array(tensor1);
  ::mlx::core::array& t2 = mlx::convert::retrieve_array(tensor2);
  ::mlx::core::array val = mlx::convert::scalar_to_mlx(value);

  ::mlx::core::array div = ::mlx::core::divide(t1, t2, ::mlx::core::Device::gpu);
  ::mlx::core::array mul = ::mlx::core::multiply(val, div, ::mlx::core::Device::gpu);
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);

  ::mlx::core::array result_mlx = ::mlx::core::add(self_mlx, mul, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, output);
}
}
