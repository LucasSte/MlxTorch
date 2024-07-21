// Created by LucasSte on 21/07/24.

#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
namespace at {

struct TORCH_API MLXHooksInterface : AcceleratorHooksInterface {
// this fails the implementation if MLXHooks functions are called, but
// MLX backend is not present.
#define FAIL_MLXHOOKS_FUNC(func) \
    TORCH_CHECK(false, "Cannot execute ", func, "() without MPS backend.");

  ~MLXHooksInterface() override = default;

  virtual bool hasMLX() const {
    return false;
  }

  virtual const Generator& getDefaultMLXGenerator() const {
    FAIL_MLXHOOKS_FUNC(__func__)
  }

  virtual void deviceSynchronize() const {
    FAIL_MLXHOOKS_FUNC(__func__)
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    FAIL_MLXHOOKS_FUNC(__func__)
  }
#undef FAIL_MLXHOOKS_FUNC
};

struct TORCH_API MLXHooksArgs {};

TORCH_DECLARE_REGISTRY(MLXHooksRegistry, MLXHooksInterface, MLXHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MLXHooksInterface& getMLXHooks();
} // namespace detail
} // namespace at
C10_CLANG_DIAGNOSTIC_POP()
