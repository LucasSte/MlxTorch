#include <ATen/TensorMeta.h>
#include <ATen/native/mlx/LinearAlgebra.h>

namespace at::native {

TORCH_IMPL_FUNC(mm_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  mm_out_mlx_impl(self, mat2, result);
}

void mm_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
// 1. Test with two allocations
// 2. If it works, modify pytorch to not allocate a tensor beforehand for MLX. (register_dispatch_key.py -> create_out)
// 3. Check if any other cleanup function is necessary.
}

}
