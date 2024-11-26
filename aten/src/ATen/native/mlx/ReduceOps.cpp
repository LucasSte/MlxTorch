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
#include <ATen/ops/sum.h>
#include <ATen/ops/mean_native.h>
#include "ATen/core/ATen_fwd.h"
#endif

namespace at::native {

TORCH_IMPL_FUNC(sum_out_mlx)
(const Tensor &input,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> dtype,
 const Tensor &output_t) {
  ::mlx::core::array& input_mlx = mlx::convert::retrieve_array(input);
  if (dtype && *dtype != input.dtype()) {
    ::mlx::core::Dtype new_type = mlx::convert::convert_scalar_type(*dtype);
    input_mlx = ::mlx::core::astype(input_mlx, new_type, ::mlx::core::Device::gpu);
  }

  ::mlx::core::array result;
  if (opt_dim.has_value()) {
    IntArrayRef dim_ref = opt_dim.value();
    std::vector<int> dims(dim_ref.size());
    for (size_t i=0; i<dim_ref.size(); i++) {
      dims[i] = static_cast<int>(dim_ref[i]);
    }
    result = ::mlx::core::sum(input_mlx, dims, keepdim, ::mlx::core::Device::gpu);
  } else {
    result = ::mlx::core::sum(input_mlx, keepdim, ::mlx::core::Device::gpu);
  }

  mlx::convert::introduce_result(std::move(result), output_t);
}

TORCH_IMPL_FUNC(mean_out_mlx)
(const Tensor& input_t,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> dtype,
 const Tensor& output_t) {
  ::mlx::core::array& input_mlx = mlx::convert::retrieve_array(input_t);
  if (dtype && *dtype != input_t.dtype()) {
    ::mlx::core::Dtype new_type = mlx::convert::convert_scalar_type(*dtype);
    input_mlx = ::mlx::core::astype(input_mlx, new_type, ::mlx::core::Device::gpu);
  }

  ::mlx::core::array result;
  if (opt_dim.has_value() && !opt_dim.value().empty()) {
    IntArrayRef dim_ref = opt_dim.value();
    std::vector<int> dims(dim_ref.size());
    for (size_t i=0; i<dim_ref.size(); i++) {
      dims[i] = static_cast<int>(dim_ref[i]);
    }
    result = ::mlx::core::mean(input_mlx, dims, keepdim, ::mlx::core::Device::gpu);
  } else {
    result = ::mlx::core::mean(input_mlx, keepdim, ::mlx::core::Device::gpu);
  }

  mlx::convert::introduce_result(std::move(result), output_t);
}

}
