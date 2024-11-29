#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/mlx/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include "MLXAllocator.h"

namespace at::detail {
// Try following CUDA's implementation. If that does not work,
// try the MPS one.

TensorBase empty_mlx(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto device = device_or_default(device_opt);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::MLX);
    TORCH_CHECK_NOT_IMPLEMENTED(
        layout_or_default(layout_opt) == Layout::Strided,
        "Only strided implemented for MLX"
        );

    check_size_nonnegative(size);

    auto* allocator = at::mlx::getMLXAllocator();
    int64_t n_elements = c10::multiply_integers(size);
    auto dtype = dtype_or_default(dtype_opt);
    TORCH_CHECK_TYPE(dtype != ScalarType::Double, "Not implemented for MLX");

    auto dtype_meta = scalarTypeToTypeMeta(dtype);
    int64_t size_bytes = n_elements * dtype_meta.itemsize();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        true);

    auto tensor = at::detail::make_tensor<TensorImpl>(
        storage_impl, DispatchKey::MLX, dtype_meta
        );

    if (size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    auto memory_format = memory_format_opt.value_or(MemoryFormat::Contiguous);
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
      at::native::fill_empty_deterministic_(tensor);
    }
    return tensor;
}

TensorBase empty_mlx(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_mlx(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  // TODO: Ste: Check is MLX is available
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mlx());
  auto dtype = dtype_or_default(dtype_opt);
  TORCH_CHECK_TYPE(dtype != ScalarType::Double, "Double not supported");
  mlx::MLXAllocator* allocator = at::mlx::getMLXAllocator();
  constexpr c10::DispatchKeySet mlx_dks(c10::DispatchKey::MLX);
  Tensor result = at::detail::empty_strided_generic(
      size, stride, allocator, mlx_dks, dtype
      );
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }

  return result;
}

TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<DeviceType> device_opt) {
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mlx());

  auto *allocator = at::mlx::getMLXAllocator();
  constexpr c10::DispatchKeySet mlx_dks(c10::DispatchKey::MLX);
  Tensor result = at::detail::empty_strided_generic(
      size, stride, allocator, mlx_dks, dtype
      );

  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }

  return result;
}

TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_mlx(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

TensorBase create_null_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  at::Allocator * mlx_allocator = at::mlx::getMLXAllocator();
  DataPtr null_ptr(nullptr, nullptr, mlx_allocator->raw_deleter(), at::Device(at::DeviceType::MLX, 0));
  c10::SymInt size_bytes(0);
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      std::move(null_ptr),
      mlx_allocator,
      false
      );

  constexpr c10::DispatchKey mlx_dks(c10::DispatchKey::MLX);
  auto dtype = dtype_or_default(options.dtype_opt());
  TORCH_CHECK_TYPE(dtype != ScalarType::Double, "Double not supported");
  auto tensor = at::detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), mlx_dks, dtype
      );

  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  } else {
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  }

  return tensor;
}

} // namespace at::detail
