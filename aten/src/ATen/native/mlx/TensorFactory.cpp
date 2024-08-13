
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mlx/EmptyTensor.h>

namespace at::native {

Tensor empty_mlx(IntArrayRef size,
                 std::optional<ScalarType> dtype_opt,
                 std::optional<Layout> layout_opt,
                 std::optional<Device> device_opt,
                 std::optional<bool> pin_memory_opt,
                 std::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_mlx(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt
      );
  return result;
}

Tensor empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_mlx(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  return result;
}

}

