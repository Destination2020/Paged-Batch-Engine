// Updated on March 15, 2026
#include "sampler/argmax_sampler.h"
#include <algorithm>
#include <base/bf16.h>
#include <vector>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
size_t ArgmaxSampler::sample(const void* logits, size_t size, base::DataType data_type,
                             void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    if (data_type == base::DataType::kDataTypeFp32) {
      const float* fp32_logits = static_cast<const float*>(logits);
      return std::distance(fp32_logits, std::max_element(fp32_logits, fp32_logits + size));
    }
    const uint16_t* bf16_logits = static_cast<const uint16_t*>(logits);
    size_t best_index = 0;
    float best_value = base::bf16_bits_to_float(bf16_logits[0]);
    for (size_t i = 1; i < size; ++i) {
      const float value = base::bf16_bits_to_float(bf16_logits[i]);
      if (value > best_value) {
        best_value = value;
        best_index = i;
      }
    }
    return best_index;
  } else {
    if (data_type == base::DataType::kDataTypeFp32) {
      return kernel::argmax_kernel_cu(static_cast<const float*>(logits), size, stream);
    }
    return kernel::argmax_kernel_cu_bf16(static_cast<const uint16_t*>(logits), size, stream);
  }
}
}  // namespace sampler
