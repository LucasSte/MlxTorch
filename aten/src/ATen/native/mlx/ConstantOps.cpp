#include <ATen/native/mlx/ConstantOps.h>
#include <mlx/ops.h>
#include <mlx/array.h>
#include <ATen/native/mlx/Convert.h>

namespace at::native {
Tensor & fill_scalar_mlx(Tensor &self, const Scalar &value) {
  if (self.numel() == 0) {
    return self;
  }

  // TODO: This is repeated code
  std::vector<int> mlx_shape;
  auto self_sizes = self.sizes();
  mlx_shape.resize(self_sizes.size());
  // TODO: Can this be optimized?
  for (size_t i=0; i<self_sizes.size(); i++) {
    mlx_shape[i] = static_cast<int>(self_sizes[i]);
  }

  ::mlx::core::array result;
  if (value.isFloatingPoint()) {
    float32_t val = value.toFloat();
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::float32);
  } else if (value.isBoolean()) {
    bool val = value.toBool();
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::bool_);
  } else if (value.isComplex()) {
    c10::complex<float> c10_val = value.toComplexFloat();
    std::complex<float> val = {c10_val.real(), c10_val.imag()};
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::complex64);
  } else {
    TORCH_INTERNAL_ASSERT(value.isIntegral(false));
    uint64_t val = value.toUInt64();
    result = ::mlx::core::full(std::move(mlx_shape), val, ::mlx::core::uint64);
  }

  mlx::convert::introduce_result(std::move(result), self);
  return self;
}

Tensor& zero_mlx_(Tensor & self) {
  return fill_scalar_mlx(self, 0.0f);
}
}
