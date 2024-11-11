#include <ATen/native/mlx/TensorShape.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <ATen/native/mlx/Convert.h>

namespace at::native {
Tensor mlx_view(const Tensor& self, IntArrayRef size) {
  // TODO: This can truly be made more efficient
  // TODO: I believe this can be removed!
  std::vector<int> mlx_shape(size.size());
  for (size_t i=0; i<size.size(); i++) {
    mlx_shape[i] = static_cast<int>(size[i]);
  }
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array result_mlx = ::mlx::core::reshape(self_mlx, std::move(mlx_shape), ::mlx::core::Device::gpu);
  result_mlx.eval();
  return mlx::convert::new_from_mlx(result_mlx);
}
}
