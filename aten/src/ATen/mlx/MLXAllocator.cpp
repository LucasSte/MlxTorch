#include <c10/core/Allocator.h>
// TODO: How to find this file?
#include <mlx/allocator.h>

namespace at::mlx {
struct TORCH_API MLXAllocator final : public c10::Allocator {

  DataPtr allocate(size_t n) {

  }
};
}
