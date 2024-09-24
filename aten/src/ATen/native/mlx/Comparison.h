#include <ATen/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/Scalar.h>

namespace at::native {
void eq_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result);
void ne_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result);
Tensor & abs_out_mlx(const Tensor & self, Tensor & output);
Tensor & fill_scalar_mlx(Tensor &self, const Scalar &value);
}
