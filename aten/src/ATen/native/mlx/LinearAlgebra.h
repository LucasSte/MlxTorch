#include <ATen/Tensor.h>

namespace at::native {
void mm_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result);

}
