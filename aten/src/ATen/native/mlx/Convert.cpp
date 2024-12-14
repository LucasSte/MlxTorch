#include <ATen/native/mlx/Convert.h>
#include <c10/core/ScalarType.h>
#include <ATen/mlx/MLXAllocator.h>
#include <mlx/allocator.h>
#include <mlx/ops.h>
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

ScalarType to_tensor_type(const ::mlx::core::array & arr) {
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
  std::vector<int> mlx_shape(self_sizes.size());
  // TODO: Can this be optimized?
  for (size_t i=0; i<self_sizes.size(); i++) {
    mlx_shape[i] = static_cast<int>(self_sizes[i]);
  }

  auto self_strides = self.strides();
  std::vector<size_t> mlx_strides(self_strides.size());
  for (size_t i=0; i<self_strides.size(); i++) {
    mlx_strides[i] = static_cast<size_t>(self_strides[i]);
  }

  ::mlx::core::array self_mlx;
  const at::DataPtr& data_ptr = self.storage().data_ptr();
  if (data_ptr.get() == nullptr && !self.storage().unsafeGetStorageImpl()->arr_st.is_null()) {
      TensorImpl * Impl = self.unsafeGetTensorImpl();
      Impl->mlx_arr = self.storage().unsafeGetStorageImpl()->arr_st;
      self_mlx = Impl->mlx_arr;
  } else {
    ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(data_ptr.get());
    ctr_ptr->rc.fetch_add(1);
    ::mlx::core::allocator::Buffer buf = {ctr_ptr->mtl_ptr};

    ::mlx::core::Dtype mlx_type = convert_type(self);

    self_mlx = ::mlx::core::array(
        std::move(buf),
        mlx_shape,
        mlx_type,
        ::mlx::core::allocator::free
    );
  }

  ::mlx::core::array mlx_res = ::mlx::core::as_strided(self_mlx, std::move(mlx_shape), std::move(mlx_strides), static_cast<size_t>(self.storage_offset()));
  // mlx_res.eval();
  return mlx_res;
}

::mlx::core::array& retrieve_array(const Tensor& self) {
  TensorImpl * impl = self.unsafeGetTensorImpl();
  ::mlx::core::array * arr = &impl->mlx_arr;
  if (arr->is_null()) {
    *arr = tensor_to_mlx(self);
    return *arr;
  }
  return *arr;
}

Tensor new_from_mlx_only(::mlx::core::array input) {
  // input.eval();
  at::Allocator * mlx_allocator = at::mlx::getMLXAllocator();
  // What about using the Data shared ptr for memory management?
  auto sym_int = SymInt(static_cast<int64_t>(input.nbytes()));

  DataPtr data_ptr(nullptr, nullptr, mlx_allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));
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
  if (bytes_offset > 0) {
    tensor_offset = static_cast<int64_t>(bytes_offset / tensor.dtype().itemsize());
  }

  TensorImpl * TImpl = tensor.unsafeGetTensorImpl();
  TImpl->set_sizes_and_strides(shape_ref, strides_ref, tensor_offset);
  TImpl->mlx_arr = std::move(input);
  TImpl->storage().unsafeGetStorageImpl()->arr_st = TImpl->mlx_arr;
  return tensor;
}

Tensor new_from_mlx(::mlx::core::array input) {
  input.eval();
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
  if (bytes_offset > 0) {
    tensor_offset = static_cast<int64_t>(bytes_offset / tensor.dtype().itemsize());
  }

  TensorImpl * TImpl = tensor.unsafeGetTensorImpl();
  TImpl->set_sizes_and_strides(shape_ref, strides_ref, tensor_offset);
  TImpl->mlx_arr = std::move(input);
  TImpl->storage().unsafeGetStorageImpl()->arr_st = TImpl->mlx_arr;
  return tensor;
}

::mlx::core::array scalar_to_mlx(const Scalar &scalar) {
  const ScalarType dtype = scalar.type();
  const ::mlx::core::Dtype mlx_type = convert_scalar_type(dtype);

  switch (dtype) {
    case ScalarType::ComplexFloat: {
      c10::complex<float> complex_num = scalar.toComplexFloat();
      std::complex<float> mlx_float = {complex_num.real(), complex_num.imag()};
      return ::mlx::core::array(mlx_float, mlx_type);
    }
    case ScalarType::Float:
      return ::mlx::core::array(scalar.toFloat(), mlx_type);
    case ScalarType::UInt64:
    case ScalarType::Long:
      return ::mlx::core::array(scalar.toUInt64(), mlx_type);
    case ScalarType::Bool:
      return ::mlx::core::array(scalar.toBool(), mlx_type);
    default:
      throw std::runtime_error("Unknown scalar type MLX.");
  }
}

void introduce_mlx_only(::mlx::core::array mlx_result, const Tensor& tensor_result) {
  // mlx_result.eval();
  tensor_result.storage().set_nbytes(c10::SymInt(static_cast<int64_t>(mlx_result.nbytes())));

  TensorImpl * TImpl = tensor_result.unsafeGetTensorImpl();
  //TImpl->set_sizes_and_strides(sizes_ref, strides_ref, storage_offset);
  TImpl->mlx_arr = std::move(mlx_result);
  TImpl->storage().unsafeGetStorageImpl()->arr_st = TImpl->mlx_arr;
}

void introduce_result(::mlx::core::array mlx_result, const Tensor& tensor_result) {
  mlx_result.eval();
  auto data_ptr = mlx_result.data_shared_ptr();
  Allocator *allocator = at::mlx::getMLXAllocator();
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(data_ptr->buffer.raw_ptr());
  ctr_ptr->rc.fetch_add(1);
  DataPtr pytorch_ptr(data_ptr->buffer.raw_ptr(), data_ptr->buffer.raw_ptr(), allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));

  auto old_ptr = tensor_result.storage().set_data_ptr(std::move(pytorch_ptr));
  c10::SymInt size_bytes(static_cast<int64_t>(ctr_ptr->usable_size(::mlx::core::allocator::Buffer(ctr_ptr->mtl_ptr))));

  tensor_result.storage().set_nbytes(std::move(size_bytes));
  size_t bytes_offset = mlx_result.storage_offset();
  size_t original_offset = static_cast<size_t>(tensor_result.storage_offset()) * tensor_result.dtype().itemsize();

  const std::vector<int> &mlx_sizes = mlx_result.shape();
  std::vector<int64_t> torch_sizes(mlx_sizes.size());
  for (size_t i=0; i<mlx_sizes.size(); i++) {
    torch_sizes[i] = static_cast<int64_t>(mlx_sizes[i]);
  }
  IntArrayRef sizes_ref = ArrayRef<int64_t>(torch_sizes);

  const std::vector<size_t> &mlx_strides = mlx_result.strides();
  std::vector<int64_t> torch_strides(mlx_strides.size());
  for (size_t i=0; i<mlx_strides.size(); i++) {
    torch_strides[i] = static_cast<int64_t>(mlx_strides[i]);
  }
  IntArrayRef strides_ref = ArrayRef<int64_t>(torch_strides);

  std::optional<int64_t> storage_offset = std::nullopt;
  if (bytes_offset != original_offset) {
    storage_offset = static_cast<int64_t>(bytes_offset / tensor_result.dtype().itemsize());
  }

  TensorImpl * TImpl = tensor_result.unsafeGetTensorImpl();
  TImpl->set_sizes_and_strides(sizes_ref, strides_ref, storage_offset);
  TImpl->mlx_arr = std::move(mlx_result);
  TImpl->storage().unsafeGetStorageImpl()->arr_st = TImpl->mlx_arr;
  tensor_result.storage().unsafeGetStorageImpl()->set_resizable(true);
}
}
