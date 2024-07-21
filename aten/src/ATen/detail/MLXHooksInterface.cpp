// Created by LucasSte on 21/07/24.

#include <ATen/detail/MLXHooksInterface.h>
#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

const MLXHooksInterface& getMLXHooks() {
  static std::unique_ptr<MLXHooksInterface> mlx_hooks;
#if !defined C10_MOBILE
  static c10::once_flag once;
  c10::call_once(once, [] {
    mlx_hooks = MLXHooksRegistry()->Create("MLXHooks", MLXHooksArgs{});
    if (!mlx_hooks) {
      mlx_hooks = std::make_unique<MLXHooksInterface>();
    }
  });
#else
  if (mps_hooks == nullptr) {
    mlx_hooks = std::make_unique<MLXHooksInterface>();
  }
#endif
  return *mlx_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MLXHooksRegistry, MLXHooksInterface, MLXHooksArgs)

} // namespace at
