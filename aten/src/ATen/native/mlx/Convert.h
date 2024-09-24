#include <ATen/TensorMeta.h>
#include <ATen/Tensor.h>
#include <mlx/dtype.h>
#include <mlx/array.h>

namespace at::native::mlx::convert {
::mlx::core::Dtype convert_type(const Tensor &self);
::mlx::core::array tensor_to_mlx(const Tensor &self);
void set_tensor_result(const ::mlx::core::array & mlx_result, const Tensor & tensor_result, const std::string name = __builtin_FUNCTION());
}
