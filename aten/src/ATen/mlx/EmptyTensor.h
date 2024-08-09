
#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

C10_EXPORT TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<DeviceType> device_opt);

C10_EXPORT TensorBase empty_strided_mlx(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options);

} // namespace at::detail
