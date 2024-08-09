#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/mlx/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include "MLXAllocator.h"

namespace at::detail {
// Try following CUDA's implementation. If that does not work,
// try the MPS one.
// TODO: Create emtpy mlx as well.

TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<DeviceType> device_opt
) {
  // TODO: Ste: Check is MLX is available
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_mlx());
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
    const TensorOptions &options) {
  return at::native::empty_strided_mlx(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt()
      );
}

} // namespace at::detail
