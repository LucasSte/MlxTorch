
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mlx/EmptyTensor.h>
#include <mlx/allocator.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/Resize.h>

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

inline static void maybe_resize_storage_mlx(TensorImpl * self, uint64_t new_size) {
  if (new_size == 0) {
    return;
  }

  auto storage = self->storage().unsafeGetStorageImpl();
  if (!storage) {
    TORCH_CHECK(false, "Tensor: invalid null storage");
  }
  uint64_t new_size_bytes = (new_size + self->storage_offset()) * self->dtype().itemsize();
  if (new_size_bytes > self->storage().nbytes()) {
    at::DataPtr new_data = storage->allocator()->allocate(new_size_bytes);
    size_t copy_capacity = std::min<size_t>(new_size_bytes, storage->nbytes());
    if (storage->data() && copy_capacity > 0) {
      ::mlx::core::allocator::MemControl * old_ctrl = ::mlx::core::allocator::MemControl::mem_control_ptr(const_cast<void*>(storage->data()));
      ::mlx::core::allocator::Buffer old_buf = {old_ctrl->mtl_ptr};
      ::mlx::core::allocator::MemControl * new_ctrl = ::mlx::core::allocator::MemControl::mem_control_ptr(new_data.get());
      ::mlx::core::allocator::Buffer new_buf = {new_ctrl->mtl_ptr};
      // TODO: Is this the best way to copy?
      std::memcpy(new_buf.raw_ptr(), old_buf.raw_ptr(), copy_capacity);
    }
    // Destructively overwrite data_ptr
    storage->set_data_ptr_noswap(std::move(new_data));
    storage->set_nbytes(new_size_bytes);
  }
}

inline TensorImpl * resize_impl_mlx_(
    TensorImpl* self,
    IntArrayRef size,
    std::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_mlx(self, storage_size);

  return self;
}

Tensor& set_storage_mlx_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);
  //std::cout << "set storage_mps " << storage_offset << " stride " << stride << std::endl;
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  std::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                                                   std::optional<IntArrayRef>(stride) : std::nullopt;
  resize_impl_mlx_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

}

