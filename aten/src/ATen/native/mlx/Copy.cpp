#include <ATen/native/mlx/Copy.h>
#include <ATen/native/mlx/Convert.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/allocator.h>
#include <c10/core/Allocator.h>
#include <ATen/mlx/MLXAllocator.h>

namespace at::native {

static Tensor copy_mlx(const Tensor &src, const Tensor &dst, bool sameType) {
  ::mlx::core::array src_mlx = mlx::convert::tensor_to_mlx(src);
  if (!sameType) {
    src_mlx = ::mlx::core::astype(src_mlx, mlx::convert::convert_type(dst), ::mlx::core::Device::gpu);
  }
  ::mlx::core::array result = ::mlx::core::copy(src_mlx, ::mlx::core::Device::gpu);
  result.eval();
  mlx::convert::set_tensor_result(result, dst);
  return dst;
}

static Tensor copy_between_devices(const Tensor &src, const Tensor &dst, const at::Device device) {
  const at::DataPtr & data_ptr = src.storage().data_ptr();
  ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(data_ptr.get());
  ctr_ptr->rc.fetch_add(1);
  at::Allocator * mlx_allocator = at::mlx::getMLXAllocator();
  DataPtr mlx_ptr(data_ptr.get(), data_ptr.get(), mlx_allocator->raw_deleter(), device);
  auto old_ptr = dst.storage().set_data_ptr(std::move(mlx_ptr));
  dst.unsafeGetTensorImpl()->set_sizes_and_strides(src.sizes(), src.strides(), src.storage_offset());
  return dst;
}


Tensor _copy_from_mlx(const Tensor& self, const Tensor& dst, bool non_blocking) {
  // TODO: I'm considering there'll be no change in the memory format
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous(), "Only contiguous memory is supported");
  bool needs_broadcasting = false;

  if (self.numel() == 0 || dst.is_same(self)) {
    return dst;
  }
  if (dst.numel() == 0) {
    dst.resize_as_(self);
  }

  TORCH_CHECK(
      dst.dim() >= self.dim(), "Destination ", self.sym_sizes(), " doesn't match the broadcast shape ", self.sym_sizes());
  if (dst.dim() > self.dim()) {
    needs_broadcasting = true;
  } else {
    const IntArrayRef src_sizes = self.sizes();
    const IntArrayRef dst_sizes = dst.sizes();
    for (const auto j : c10::irange(self.dim())) {
      if (src_sizes[j] == 1 && dst_sizes[j] != 1) {
        needs_broadcasting = true;
        break;
      }
    }
  }

  const bool sameDataType = self.dtype() == dst.dtype() && self.is_conj() == dst.is_conj();
  if (self.device().type() == at::kMLX && dst.device().type() == kMLX) {
    return copy_mlx(needs_broadcasting ? self.expand_as(dst) : self, dst, sameDataType);
  }

  // TODO: I'm assuming we can avoid a copy in these two cases.
  if (self.device().type() == at::kCPU && dst.device().type() == kMLX) {
    Tensor res = copy_between_devices(needs_broadcasting ? self.expand_as(dst) : self, dst, at::Device(at::DeviceType::MLX, 0));
    if (!sameDataType) {
      ::mlx::core::array res_mlx = mlx::convert::tensor_to_mlx(res);
      ::mlx::core::array casted = ::mlx::core::astype(res_mlx, mlx::convert::convert_type(dst), ::mlx::core::Device::gpu);
      casted.eval();
      mlx::convert::set_tensor_result(res_mlx, res);
    }
    return res;
  }

  if (self.device().type() == at::kMLX && dst.device().type() == kCPU) {
    if (!sameDataType) {
      ::mlx::core::array res_mlx = mlx::convert::tensor_to_mlx(self);
      ::mlx::core::array casted = ::mlx::core::astype(res_mlx, mlx::convert::convert_type(dst), ::mlx::core::Device::gpu);
      casted.eval();
      mlx::convert::set_tensor_result(res_mlx, dst);
      return dst;
    }
    return copy_between_devices(needs_broadcasting ? self.expand_as(dst) : self, dst, at::Device(at::DeviceType::CPU, -1));
  }

  TORCH_INTERNAL_ASSERT(self.device().type() == DeviceType::MLX, "mlx_copy_ is implemented only for *->MLX; MLX->*");
  return dst;
}
}
