
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mlx/EmptyTensor.h>

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

//const Tensor & resize_mps_(
//    const Tensor &self,
//    IntArrayRef size,
//    std::optional<MemoryFormat> optional_memory_format
//    ) {
//  auto* self_ = self.unsafeGetTensorImpl();
//  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
//
//  if (self_->sizes() == size) {
//    return self;
//  }
//  self_->set_sizes_contiguous(size);
//  int64_t storage_size = self_->numel();
//
//  auto storage = self_->storage().unsafeGetStorageImpl();
//  uint64_t new_size_bytes = (storage_size + self_->storage_offset()) * self_->dtype().itemsize();
//
//  if (new_size_bytes > self_->storage().nbytes()) {
//    at::DataPtr new_data = storage->allocator()->allocate(new_size_bytes);
//    size_t copy_capacity = std::min<size_t>(new_size_bytes, storage->nbytes());
//
//    if (storage->data() && copy_capacity > 0) {
//      // TODO: This isn't right
//      at::native::mps::copy_blit_mps(new_data.get(), storage->data(), copy_capacity);
//    }
//    // Destructively overwrite data_ptr
//    storage->set_data_ptr_noswap(std::move(new_data));
//    storage->set_nbytes(new_size_bytes);
//  }
//
//  if (optional_memory_format.has_value()) {
//    auto memory_format =
//        optional_memory_format.value();
//    TORCH_CHECK(
//        memory_format != MemoryFormat::Preserve,
//        "Unsupported memory format",
//        memory_format);
//    self_->empty_tensor_restride(memory_format);
//  }
//  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
//    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
//  }
//  return self;
//}

}

