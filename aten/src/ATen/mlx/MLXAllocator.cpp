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
  std::cout << "Allocating " << n << " bytes" << std::endl;
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

//void MLXCpuAllocator::Delete(void* ptr) {
//  if (ptr) {
//    uint8_t * ctx = (uint8_t*)ptr - 8;
//    uint64_t original = *(uint64_t*)ctx;
//    ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::Buffer{(void*) original};
//    ::mlx::core::allocator::free(buf);
//  }
//}
//
//DataPtr MLXCpuAllocator::allocate(size_t n) {
//  // NOTE: I modified mlx to allocate n+8, and return ptr+8 for the raw ptr
//  ::mlx::core::allocator::Buffer buf = ::mlx::core::allocator::malloc_or_wait(n);
//  return DataPtr{buf.raw_ptr(), buf.raw_ptr(), &MLXCpuAllocator::Delete, at::Device(at::DeviceType::CPU)};
//}
//
//DeleterFnPtr MLXCpuAllocator::raw_deleter() const {
//  return &MLXCpuAllocator::Delete;
//}
//
//void MLXCpuAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
//  default_copy_data(dest, src, count);
//}
//
//MLXCpuAllocator *getMLXCpuAllocator() {
//  static MLXCpuAllocator allocator;
//  return &allocator;
//}

}
