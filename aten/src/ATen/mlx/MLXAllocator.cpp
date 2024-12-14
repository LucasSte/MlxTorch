#include "MLXAllocator.h"
#include <mlx/allocator.h>
#include <iostream>

namespace at::mlx {

void MLXAllocator::Delete(void* ptr) {
    if (ptr) {
        ::mlx::core::allocator::MemControl* ctr_ptr = ::mlx::core::allocator::MemControl::mem_control_ptr(ptr);
        ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::Buffer{ctr_ptr->mtl_ptr};
        ::mlx::core::allocator::free(buf);
    }
}

DataPtr MLXAllocator::allocate(size_t n) {
  // Note: I am still allocating extra eight bytes here.
    ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::malloc_or_wait(n);
    return DataPtr{buf.raw_ptr(), buf.raw_ptr(), &MLXAllocator::Delete, at::Device(at::DeviceType::MLX, 0)};
}

DeleterFnPtr MLXAllocator::raw_deleter() const {
  return &MLXAllocator::Delete;
}

void MLXAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  default_copy_data(dest, src, count);
}

MLXAllocator* getMLXAllocator() {
  static MLXAllocator allocator;
  return &allocator;
}
}
