#include <ATen/native/mlx/Losses.h>
#include <ATen/native/mlx/Convert.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/TensorBase.h>
#include <mlx/array.h>
#include <mlx/ops.h>

namespace at::native {
Tensor binary_cross_entropy_mlx(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  // TODO: Is this necessary?
//  Tensor input_squeezed = input.squeeze();
//  Tensor target_squeezed = target.squeeze();
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor &weight = *weight_maybe_owned;

  ::mlx::core::array& mlx_input = mlx::convert::retrieve_array(input);
  ::mlx::core::array& mlx_targets = mlx::convert::retrieve_array(target);

  ::mlx::core::array loged = ::mlx::core::log(mlx_input, ::mlx::core::Device::gpu);
  ::mlx::core::array log_inputs_clip = ::mlx::core::clip(loged, ::mlx::core::array(-100.0, mlx::convert::convert_type(input)), std::nullopt, ::mlx::core::Device::gpu);

  ::mlx::core::array one_minus = ::mlx::core::subtract(::mlx::core::array(1), mlx_input, ::mlx::core::Device::gpu);
  ::mlx::core::array minus_log = ::mlx::core::log(one_minus, ::mlx::core::Device::gpu);
  ::mlx::core::array log_inputs_inv_clip = ::mlx::core::clip(minus_log, ::mlx::core::array(-100.0, mlx::convert::convert_type(input)), std::nullopt, ::mlx::core::Device::gpu);

  ::mlx::core::array lhs = ::mlx::core::multiply(mlx_targets, log_inputs_clip, ::mlx::core::Device::gpu);
  ::mlx::core::array rhs = ::mlx::core::multiply(one_minus, log_inputs_inv_clip,::mlx::core::Device::gpu);

  ::mlx::core::array not_loss = ::mlx::core::add(lhs, rhs, ::mlx::core::Device::gpu);
  ::mlx::core::array loss = ::mlx::core::negative(not_loss, ::mlx::core::Device::gpu);

  if (weight.defined()) {
    ::mlx::core::array& weights = mlx::convert::retrieve_array(weight);
    loss = ::mlx::core::multiply(loss, weights, ::mlx::core::Device::gpu);
  }

  switch (reduction) {
    case Reduction::Mean: {
      ::mlx::core::array res = ::mlx::core::mean(loss, ::mlx::core::Device::gpu);
      return mlx::convert::new_from_mlx_only(std::move(res));
    }
    case Reduction::Sum: {
    ::mlx::core::array res = ::mlx::core::sum(loss, ::mlx::core::Device::gpu);
    return mlx::convert::new_from_mlx_only(std::move(res));
    }
    case Reduction::None: {
      return mlx::convert::new_from_mlx_only(std::move(loss));
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid case!");
  }
}

Tensor binary_cross_entropy_backward_mlx(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction) {
  // TODO: Is this necessary?
//  Tensor input_squeezed = input.squeeze();
//  Tensor target_squeezed = target.squeeze();
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor &weight = *weight_maybe_owned;

  // d(L)/d(y) = -w (x - y) / (y - y^2)
  ::mlx::core::array& input_mlx = mlx::convert::retrieve_array(input);
  ::mlx::core::array& target_mlx = mlx::convert::retrieve_array(target);
  ::mlx::core::array& grad_mlx = mlx::convert::retrieve_array(grad_output);

  ::mlx::core::array epsilon = ::mlx::core::array(1e-12, mlx::convert::convert_type(input));
  // 1 - y
  ::mlx::core::array one_input = ::mlx::core::subtract(::mlx::core::array(1.0, mlx::convert::convert_type(input)), input_mlx, ::mlx::core::Device::gpu);
  // y * (y - 1)
  ::mlx::core::array input_times = ::mlx::core::multiply(input_mlx, one_input, ::mlx::core::Device::gpu);
  // max(y * (1 - y), epsilon)
  ::mlx::core::array gradDenominator = ::mlx::core::maximum(input_times, epsilon, ::mlx::core::Device::gpu);
  // x - y
  ::mlx::core::array input_target = ::mlx::core::subtract(input_mlx, target_mlx, ::mlx::core::Device::gpu);
  // (x - y) / max(y * (1 - y), epsilon)
  ::mlx::core::array division = ::mlx::core::divide(input_target, gradDenominator, ::mlx::core::Device::gpu);
  // w * ((x - y) / max (y * (1 - y), epsilon))
  ::mlx::core::array bce_loss = ::mlx::core::multiply(grad_mlx, division, ::mlx::core::Device::gpu);

  if (weight.defined()) {
    ::mlx::core::array& weight_mlx = mlx::convert::retrieve_array(weight);
    bce_loss = ::mlx::core::multiply(bce_loss, weight_mlx, ::mlx::core::Device::gpu);
  }

  if (reduction == at::Reduction::Mean) {
    ::mlx::core::array numel = ::mlx::core::array(static_cast<float>(input.numel()), mlx::convert::convert_type(input));
    bce_loss = ::mlx::core::divide(bce_loss, numel, ::mlx::core::Device::gpu);
  }


  return mlx::convert::new_from_mlx_only(std::move(bce_loss));
}

}
