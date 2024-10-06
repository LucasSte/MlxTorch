#include <ATen/native/mlx/Losses.h>
#include <ATen/native/mlx/Convert.h>
#include <ATen/core/Reduction.h>
#include <mlx/array.h>
#include <mlx/ops.h>

namespace at::native {
Tensor binary_cross_entropy_mlx(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  ::mlx::core::array mlx_input = mlx::convert::tensor_to_mlx(input);
  ::mlx::core::array mlx_targets = mlx::convert::tensor_to_mlx(target);

  ::mlx::core::array loged = ::mlx::core::log(mlx_input, ::mlx::core::Device::gpu);
  ::mlx::core::array log_inputs_clip = ::mlx::core::clip(loged, ::mlx::core::array(-100.0), std::nullopt, ::mlx::core::Device::gpu);

  ::mlx::core::array one_minus = ::mlx::core::subtract(::mlx::core::array(1), mlx_input, ::mlx::core::Device::gpu);
  ::mlx::core::array minus_log = ::mlx::core::log(one_minus, ::mlx::core::Device::gpu);
  ::mlx::core::array log_inputs_inv_clip = ::mlx::core::clip(minus_log, ::mlx::core::array(-100.0), std::nullopt, ::mlx::core::Device::gpu);

  ::mlx::core::array lhs = ::mlx::core::multiply(mlx_targets, log_inputs_clip, ::mlx::core::Device::gpu);
  ::mlx::core::array rhs = ::mlx::core::multiply(one_minus, log_inputs_inv_clip,::mlx::core::Device::gpu);

  ::mlx::core::array not_loss = ::mlx::core::add(lhs, rhs, ::mlx::core::Device::gpu);
  ::mlx::core::array loss = ::mlx::core::negative(not_loss, ::mlx::core::Device::gpu);

  if (weight_opt) {
    ::mlx::core::array weights = mlx::convert::tensor_to_mlx(*weight_opt);
    loss = ::mlx::core::multiply(loss, weights, ::mlx::core::Device::gpu);
  }

  switch (reduction) {
    case Reduction::Mean: {
      ::mlx::core::array res = ::mlx::core::mean(loss, ::mlx::core::Device::gpu);
      res.eval();
      return mlx::convert::new_from_mlx(res);
    }
    case Reduction::Sum: {
    ::mlx::core::array res = ::mlx::core::sum(loss, ::mlx::core::Device::gpu);
    res.eval();
    return mlx::convert::new_from_mlx(res);
    }
    case Reduction::None: {
      loss.eval();
      return mlx::convert::new_from_mlx(loss);
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid case!");
  }
}
}
