#include "MLXAllocator.h"
#include <mlx/allocator.h>

namespace at::mlx {

void MLXAllocator::Delete(void* ptr) {
    if (ptr) {
      ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::Buffer{ptr};
      ::mlx::core::allocator::free(buf);
    }
}

DataPtr MLXAllocator::allocate(size_t n) {
    ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::malloc_or_wait(n);
    return DataPtr{buf.ptr(), buf.ptr(), &Delete, at::Device(at::DeviceType::MLX, 0)};
}

DeleterFnPtr MLXAllocator::raw_deleter() const {
  return &Delete;
}

void MLXAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  default_copy_data(dest, src, count);
}

MLXAllocator* getMLXAllocator() {
  static MLXAllocator allocator;
  return &allocator;
}

}
