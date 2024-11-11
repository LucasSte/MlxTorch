#include <ATen/Tensor.h>
#include <ATen/TensorMeta.h>

namespace at::native {
Tensor mlx_view(const Tensor& self, IntArrayRef size);
}
