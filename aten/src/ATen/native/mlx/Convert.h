#include <ATen/TensorMeta.h>
#include <ATen/Tensor.h>
#include <mlx/dtype.h>
#include <mlx/array.h>
#include <ATen/Scalar.h>

namespace at::native::mlx::convert {
::mlx::core::Dtype convert_type(const Tensor& self);
::mlx::core::Dtype convert_scalar_type(ScalarType t);
::mlx::core::array tensor_to_mlx(const Tensor& self);
::mlx::core::array scalar_to_mlx(const Scalar& scalar);
::mlx::core::array& retrieve_array(const Tensor& self);
void introduce_result(const ::mlx::core::array& mlx_result, const Tensor& tensor_result);
Tensor new_from_mlx(const ::mlx::core::array& input);
}
