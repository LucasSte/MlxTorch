#include <ATen/TensorMeta.h>
#include <ATen/native/mlx/LinearAlgebra.h>
#include <mlx/array.h>
#include <mlx/ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/bitwise_and_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/sigmoid_native.h>
#include <ATen/ops/stack.h>
#endif
#include <ATen/native/DispatchStub.h>
#include "ATen/native/IndexKernel.h"
#include <ATen/native/mlx/Convert.h>
#include "c10/core/Allocator.h"


namespace at::native {

TORCH_IMPL_FUNC(mm_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  mm_out_mlx_impl(self, mat2, result);
}

TORCH_IMPL_FUNC(mul_out_mlx)(const Tensor& self, const Tensor& mat2, const Tensor& output) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& mat2_mlx = mlx::convert::retrieve_array(mat2);
  ::mlx::core::array result_mlx = ::mlx::core::multiply(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, output);
}


    void mm_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  // Ensure both tensors are in MLX or CPU!

  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& mat2_mlx = mlx::convert::retrieve_array(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::matmul(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  // Do I need to evaluate it here?
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, result);

// Just to remember: (register_dispatch_key.py -> create_out)

// 1. Reduce operations overhead (creating mlx tensors)
// 2. Only evaluate when needed. If I use MPS operations (try to avoid them), evaluate MLX and sync MPS.
// 3. Ensure all operations are correctly implemented (write mlx evaluation tests)
// 4. Understand how much work is needed to benchmark things (only do it if it is going to be quick)
// 5. Release
}

TORCH_IMPL_FUNC(addmm_out_mlx)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
 ::mlx::core::array& bias = mlx::convert::retrieve_array(self);
 ::mlx::core::array& input = mlx::convert::retrieve_array(mat1);
 ::mlx::core::array& weight = mlx::convert::retrieve_array(mat2);

 float fbeta = beta.toFloat();
 float falpha = alpha.toFloat();
 ::mlx::core::array result_mlx = ::mlx::core::addmm(bias, input, weight, falpha, fbeta, ::mlx::core::Device::gpu);
 result_mlx.eval();
 mlx::convert::introduce_result(result_mlx, const_cast<Tensor&>(result));
}
// TODO: Put the following in another file

TORCH_IMPL_FUNC(bitwise_and_out_mlx)(const Tensor& self, const Tensor& mat2, const Tensor& output) {
  ::mlx::core::array& self_mlx = mlx::convert::retrieve_array(self);
  ::mlx::core::array& mat2_mlx = mlx::convert::retrieve_array(mat2);
  ::mlx::core::array result_mlx = ::mlx::core::bitwise_and(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  result_mlx.eval();

  mlx::convert::introduce_result(result_mlx, output);
}

}
