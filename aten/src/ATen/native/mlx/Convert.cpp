#include <ATen/native/mlx/Convert.h>
#include <c10/core/ScalarType.h>
#include <ATen/mlx/MLXAllocator.h>
#include <mlx/allocator.h>

namespace at::native::mlx::convert {
::mlx::core::Dtype convert_type(const Tensor &self) {
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

::mlx::core::array tensor_to_mlx(const Tensor &self) {
  auto self_sizes = self.sizes();
  std::vector<int> mlx_shape;
  mlx_shape.resize(self_sizes.size());
  // TODO: Can this be optimized?
  for (size_t i=0; i<self_sizes.size(); i++) {
    mlx_shape[i] = static_cast<int>(self_sizes[i]);
  }

  const at::DataPtr& data_ptr = self.storage().data_ptr();

  ::mlx::core::allocator::Buffer buf = {data_ptr.get()};
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::from_buffer(buf);
  ctr_ptr->rc.fetch_add(1);

  ::mlx::core::Dtype mlx_type = convert_type(self);

  ::mlx::core::array self_mlx = ::mlx::core::array(
      std::move(buf),
      std::move(mlx_shape),
      mlx_type,
      ::mlx::core::allocator::free
  );

  return self_mlx;
}

void set_tensor_result(const ::mlx::core::array & mlx_result, const Tensor & tensor_result,
                       const std::string name) {
  auto data_ptr = mlx_result.data_shared_ptr();
  Allocator *allocator = at::mlx::getMLXAllocator();
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::from_buffer(data_ptr->buffer);
  ctr_ptr->rc.fetch_add(1);
  std::cout << "Result ptr: " << data_ptr->buffer.raw_ptr() << " caller: " << name << std::endl;
  DataPtr pytorch_ptr(data_ptr->buffer.ptr(), data_ptr->buffer.ptr(), allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));

  auto old_ptr = tensor_result.storage().set_data_ptr(std::move(pytorch_ptr));
}

}
