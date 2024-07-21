
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace at::mlx {
C10_REGISTER_GUARD_IMPL(MLX, c10::impl::NoOpDeviceGuardImpl<DeviceType::MLX>)
} // namespace at::mlx
