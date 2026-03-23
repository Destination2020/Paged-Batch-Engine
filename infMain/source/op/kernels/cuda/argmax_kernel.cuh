// Updated on March 15, 2026
#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cstddef>
#include <cstdint>
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
size_t argmax_kernel_cu_bf16(const uint16_t* input_ptr, size_t size, void* stream);
}
#endif  // ARGMAX_KERNEL_CUH
