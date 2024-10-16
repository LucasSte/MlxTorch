#include <ATen/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/Scalar.h>

namespace at::native {
Tensor & fill_scalar_mlx(Tensor &self, const Scalar &value);
Tensor & zero_mlx_(Tensor &self);
}
