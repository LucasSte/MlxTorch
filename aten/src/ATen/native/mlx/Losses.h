#include <ATen/Tensor.h>
#include <ATen/TensorMeta.h>

namespace at::native {
Tensor binary_cross_entropy_mlx(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction);
Tensor binary_cross_entropy_backward_mlx(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction);
}
