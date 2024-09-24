#include <ATen/TensorMeta.h>
#include <ATen/native/mlx/LinearAlgebra.h>
#include <mlx/array.h>
#include <mlx/ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/stack.h>
#include "c10/core/Allocator.h"
#endif
#include <ATen/native/DispatchStub.h>
#include "ATen/native/IndexKernel.h"
#include <ATen/native/mlx/Convert.h>

namespace at::native {

TORCH_IMPL_FUNC(mm_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  mm_out_mlx_impl(self, mat2, result);
}

TORCH_IMPL_FUNC(mul_out_mlx)(const Tensor& self, const Tensor& other, const Tensor& output) {
  std::cout << "Calculating mul" << std::endl;
}


    void mm_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  // Ensure both tensors are in MLX or CPU!

  std::cout << "Calculating matmul!" << std::endl;
  ::mlx::core::array self_mlx = mlx::convert::tensor_to_mlx(self);
  ::mlx::core::array mat2_mlx = mlx::convert::tensor_to_mlx(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::matmul(self_mlx, mat2_mlx, ::mlx::core::Device::gpu);
  // Do I need to evaluate it here?
  result_mlx.eval();
  std::cout << "Calculated mamtul!" << std::endl;

  mlx::convert::set_tensor_result(result_mlx, result);

//  const at::DataPtr& test_ptr = result.storage().data_ptr();

  // float32_t * ptr = reinterpret_cast<float32_t*>(test_ptr.get());
  // Dispatch stub for MLX is not working. (print(mlx_arr) -> after matmul)

  // Is this needed?
//  auto self_strides = self.strides();
//  std::vector<size_t> mlx_strides;
//  mlx_strides.resize(self_strides.size());
//  for (size_t i=0; i<self_strides.size(); i++) {
//    mlx_strides[i] = static_cast<size_t>(self_strides[i]);
//  }
//

// 1. Test with two allocations
// 2. If it works, modify pytorch to not allocate a tensor beforehand for MLX. (register_dispatch_key.py -> create_out)
// 3. Check if any other cleanup function is necessary.
}

// Create eq tensor out mlx, abs and mul

}
