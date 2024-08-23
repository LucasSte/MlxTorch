#include <c10/core/Allocator.h>

namespace at::mlx {

struct TORCH_API MLXAllocator final : public c10::Allocator {
 public:
  DataPtr allocate(size_t n) override;
  DeleterFnPtr raw_deleter() const override;
  void copy_data(void * dest, const void * src, std::size_t count) const override;

 private:
  static void Delete(void *ptr);
};

struct TORCH_API MLXCpuAllocator final : public c10::Allocator {
 public:
  DataPtr allocate(size_t n) override;
  DeleterFnPtr raw_deleter() const override;
  void copy_data(void * dest, const void * src, std::size_t count) const override;

 private:
  static void Delete(void *ptr);
};

MLXAllocator* getMLXAllocator();
MLXCpuAllocator* getMLXCpuAllocator();
} // namespace at::mlx
