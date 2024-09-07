#include <ATen/TensorMeta.h>
#include <ATen/native/mlx/LinearAlgebra.h>
#include <ATen/mlx/MLXAllocator.h>
#include <mlx/allocator.h>
#include <mlx/dtype.h>
#include <mlx/array.h>
#include <mlx/ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mm_native.h>
#include <ATen/ops/stack.h>
#include "c10/core/Allocator.h"
#endif

namespace at::native {

TORCH_IMPL_FUNC(mm_out_mlx)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  mm_out_mlx_impl(self, mat2, result);
}

static ::mlx::core::Dtype convert_type(const Tensor &self) {
  switch (self.dtype().toScalarType()) {
    case ScalarType::Byte:
      return ::mlx::core::uint8;
    case ScalarType::Char:
      return ::mlx::core::int8;
    case ScalarType::Short:
      return ::mlx::core::int16;
    case ScalarType::Int:
      return ::mlx::core::int32;
    case ScalarType::Long:
      return ::mlx::core::int64;
    case ScalarType::Half:
      return ::mlx::core::float16;
    case ScalarType::Float:
      return ::mlx::core::float32;
    case ScalarType::ComplexDouble:
      return ::mlx::core::complex64;
    case ScalarType::Bool:
      return ::mlx::core::bool_;
    case ScalarType::BFloat16:
      return ::mlx::core::bfloat16;
    case ScalarType::UInt16:
      return ::mlx::core::uint16;
    case ScalarType::UInt32:
      return ::mlx::core::uint32;
    case ScalarType::UInt64:
      return ::mlx::core::uint64;
    default:
      TORCH_CHECK(false, "Invalid type");
  }
}

static ::mlx::core::array convert_to_mlx(const Tensor &self) {
  auto self_sizes = self.sizes();
  std::vector<int> mlx_shape;
  mlx_shape.resize(self_sizes.size());
  for (size_t i=0; i<self_sizes.size(); i++) {
    mlx_shape[i] = static_cast<int>(self_sizes[i]);
  }

  const at::DataPtr& data_ptr = self.storage().data_ptr();

  float32_t * ptr = reinterpret_cast<float32_t*>(data_ptr.get());
  std::cout << "Data: " << std::endl;
  for(size_t i=0; i<4; i++) {
    std::cout << (*ptr) << " At: " << ptr << std::endl;
    ptr += 1;
  }

  uint8_t* mtl_addr_ptr = (uint8_t*) data_ptr.get() - 8;
  uint64_t mlt_addr = *(uint64_t*)mtl_addr_ptr;
  void* new_raw_ptr = (void*)mlt_addr;

  ::mlx::core::allocator::Buffer buf = {new_raw_ptr};
  ::mlx::core::Dtype mlx_type = convert_type(self);

  ::mlx::core::array self_mlx = ::mlx::core::array(
      std::move(buf),
      std::move(mlx_shape),
      mlx_type,
      ::mlx::core::allocator::free
  );

  return self_mlx;
}

void mm_out_mlx_impl(const Tensor & self, const Tensor & mat2, const Tensor & result) {

  ::mlx::core::array self_mlx = convert_to_mlx(self);
  ::mlx::core::array mat2_mlx = convert_to_mlx(mat2);

  ::mlx::core::array result_mlx = ::mlx::core::matmul(self_mlx, mat2_mlx, ::mlx::core::Device::cpu);
  // Do I need to evaluate it here?
  result_mlx.eval();

  auto data_ptr = result_mlx.data_shared_ptr();
  Allocator *allocator = at::mlx::getMLXAllocator();
  DataPtr pytorch_ptr(data_ptr->buffer.raw_ptr(), data_ptr->buffer.raw_ptr(), allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));

  std::cout << "Calculated mamtul!" << std::endl;
  auto old_ptr = result.storage().set_data_ptr(std::move(pytorch_ptr));

  const at::DataPtr& test_ptr = result.storage().data_ptr();

  float32_t * ptr = reinterpret_cast<float32_t*>(test_ptr.get());
  std::cout << "Result Data: " << std::endl;
  // This is correct!
  for(size_t i=0; i<4; i++) {
    std::cout << (*ptr) << " At: " << ptr << std::endl;
    ptr += 1;
  }

  // eq Tensor out is wrong I believe. Is it?
  // Dispatch stub for MLX is not working.
  // A copy to CPU to print the array isn't working.


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

}
