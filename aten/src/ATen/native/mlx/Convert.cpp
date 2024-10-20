#include <ATen/native/mlx/Convert.h>
#include <c10/core/ScalarType.h>
#include <ATen/mlx/MLXAllocator.h>
#include <mlx/allocator.h>
#include <mlx/dtype.h>
#include <ATen/Storage.h>
#include <c10/core/SymInt.h>
#include <ATen/core/TensorBase.h>

namespace at::native::mlx::convert {
::mlx::core::Dtype convert_type(const Tensor &self) {
  return convert_scalar_type(self.dtype().toScalarType());
}

::mlx::core::Dtype convert_scalar_type(ScalarType t) {
  switch (t) {
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
    case ScalarType::ComplexFloat:
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
      TORCH_CHECK(false, "Invalid type1");
  }
}

static ScalarType to_tensor_type(const ::mlx::core::array & arr) {
  switch (arr.dtype().val()) {
    case ::mlx::core::Dtype::Val::uint8:
      return ScalarType::Byte;
    case ::mlx::core::Dtype::Val::int8:
      return ScalarType::Char;
    case ::mlx::core::Dtype::Val::int16:
      return ScalarType::Short;
    case ::mlx::core::Dtype::Val::int32:
      return ScalarType::Int;
    case ::mlx::core::Dtype::Val::int64:
      return ScalarType::Long;
    case ::mlx::core::Dtype::Val::float16:
      return ScalarType::Half;
    case ::mlx::core::Dtype::Val::float32:
      return ScalarType::Float;
    case ::mlx::core::Dtype::Val::complex64:
      return ScalarType::ComplexFloat;
    case ::mlx::core::Dtype::Val::bool_:
      return ScalarType::Bool;
    case ::mlx::core::Dtype::Val::bfloat16:
      return ScalarType::BFloat16;
    case ::mlx::core::Dtype::Val::uint16:
      return ScalarType::UInt16;
    case ::mlx::core::Dtype::Val::uint32:
      return ScalarType::UInt32;
    case ::mlx::core::Dtype::Val::uint64:
      return ScalarType::UInt64;
    default:
      TORCH_CHECK(false, "Invalid type2");
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

  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(data_ptr.get());
  ctr_ptr->rc.fetch_add(1);
  ::mlx::core::allocator::Buffer buf = {ctr_ptr->mtl_ptr};

  ::mlx::core::Dtype mlx_type = convert_type(self);

  size_t bytes_offset = static_cast<size_t>(self.storage_offset()) + self.dtype().itemsize();
  ::mlx::core::array self_mlx = ::mlx::core::array(
      std::move(buf),
      std::move(mlx_shape),
      mlx_type,
      ::mlx::core::allocator::free
  );
  self_mlx.set_storage_offset(bytes_offset);

  return self_mlx;
}

void set_tensor_result(const ::mlx::core::array & mlx_result, const Tensor & tensor_result) {
  auto data_ptr = mlx_result.data_shared_ptr();
  Allocator *allocator = at::mlx::getMLXAllocator();
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(data_ptr->buffer.raw_ptr());
  ctr_ptr->rc.fetch_add(1);
  // std::cout << "Result ptr: " << data_ptr->buffer.raw_ptr() << " caller: " << name << std::endl;
  DataPtr pytorch_ptr(data_ptr->buffer.raw_ptr(), data_ptr->buffer.raw_ptr(), allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));

  auto old_ptr = tensor_result.storage().set_data_ptr(std::move(pytorch_ptr));
  size_t bytes_offset = mlx_result.storage_offset();
  size_t original_offset = static_cast<size_t>(tensor_result.storage_offset()) * tensor_result.dtype().itemsize();
  if (bytes_offset > 0 && bytes_offset != original_offset) {
    int64_t offset = static_cast<int64_t>(bytes_offset / tensor_result.dtype().itemsize());
    tensor_result.unsafeGetTensorImpl()->set_storage_offset(offset);
  }
}

Tensor new_from_mlx(const ::mlx::core::array & input) {
  at::Allocator * mlx_allocator = at::mlx::getMLXAllocator();
  // What about using the Data shared ptr for memory management?
  void * raw_ptr = input.data_shared_ptr()->buffer.raw_ptr();
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(raw_ptr);
  ctr_ptr->rc.fetch_add(1);
  size_t size_bytes = ::mlx::core::allocator::MemControl::usable_size(::mlx::core::allocator::Buffer(ctr_ptr->mtl_ptr));
  auto sym_int = SymInt(static_cast<int64_t>(size_bytes));


  DataPtr data_ptr(raw_ptr, raw_ptr, mlx_allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      sym_int,
      std::move(data_ptr),
      mlx_allocator,
      true
      );
  constexpr c10::DispatchKeySet mlx_dks(c10::DispatchKey::MLX);
  ScalarType tensor_type = to_tensor_type(input);
  caffe2::TypeMeta type = caffe2::TypeMeta::fromScalarType(tensor_type);
  auto tensor = at::detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), mlx_dks, type
      );

  auto mlx_shape = input.shape();
  std::vector<int64_t> ref(mlx_shape.size());
  for(size_t i=0; i<mlx_shape.size(); i++) {
    ref[i] = static_cast<int64_t>(mlx_shape[i]);
  }
  auto shape_ref = ArrayRef(ref);

  auto mlx_strides = input.strides();
  std::vector<int64_t> ref2(mlx_strides.size());
  for(size_t i=0; i<mlx_strides.size(); i++) {
    ref2[i] = static_cast<int64_t>(mlx_strides[i]);
  }
  auto strides_ref = ArrayRef(ref2);

  size_t bytes_offset = input.storage_offset();
  std::optional<int64_t> tensor_offset = std::nullopt;
  if (size_bytes > 0) {
    tensor_offset = static_cast<int64_t>(size_bytes / tensor.dtype().itemsize());
  }

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(shape_ref, strides_ref, tensor_offset);
  return tensor;
}

}
