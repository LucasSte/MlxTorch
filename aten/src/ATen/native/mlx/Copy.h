#include <ATen/Tensor.h>
#include <ATen/TensorMeta.h>

namespace at::native {
Tensor _copy_from_mlx(const Tensor& self, const Tensor& dst, bool non_blocking);
}
