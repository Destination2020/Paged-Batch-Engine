// Updated on March 15, 2026
#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <base/base.h>
#include <cstddef>
#include <cstdint>
namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual size_t sample(const void* logits, size_t size, base::DataType data_type,
                        void* stream = nullptr) = 0;

 protected:
  base::DeviceType device_type_;
};
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H
