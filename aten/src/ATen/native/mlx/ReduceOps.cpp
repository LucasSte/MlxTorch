#include <ATen/TensorMeta.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>
#include <ATen/native/mlx/Convert.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sum_native.h>
#endif

namespace at::native {

TENSOR_IMPL_FUNC(sum_out_mlx)
(const Tensor &input,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> dtype,
 const Tensor &output_t) {
  ::mlx::core::array input_mlx = mlx::convert::tensor_to_mlx(input);
  if (dtype && *dtype != input.dtype()) {
    ::mlx::core::Dtype new_type = mlx::convert::convert_scalar_type(*dtype);
    input_mlx = ::mlx::core::astype(input_mlx, new_type, ::mlx::core::Device::gpu);
  }


}

}
