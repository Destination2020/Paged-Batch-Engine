// Updated on March 15, 2026
//
// Created by fss on 24-6-9.
//

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

  size_t sample(const void* logits, size_t size, base::DataType data_type,
                void* stream) override;
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
