// Updated on March 15, 2026
#include "sampler/argmax_sampler.h"
#include "../op/kernels/kernels_interface.h"
namespace sampler {
size_t ArgmaxSampler::sample(const void* logits, size_t size, base::DataType data_type,
                             void* stream) {
  return kernel::get_argmax_logits_kernel(device_type_)(logits, size, data_type, stream);
}
}  // namespace sampler
