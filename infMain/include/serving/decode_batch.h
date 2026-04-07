// Decode batch metadata for multi-sequence batched inference
#ifndef KUIPER_INCLUDE_SERVING_DECODE_BATCH_H_
#define KUIPER_INCLUDE_SERVING_DECODE_BATCH_H_

#include <cstdint>
#include <vector>
#include "base/kv_cache_manager.h"
#include "tensor/tensor.h"

namespace serving {

struct DecodeBatchMetadata {
  DecodeBatchMetadata() {}

  int32_t batch_size = 0;
  int32_t max_blocks_per_seq = 0;

  // All on GPU
  tensor::Tensor token_ids;     // [batch_size] int32
  tensor::Tensor positions;     // [batch_size] int32
  tensor::Tensor seq_lens;      // [batch_size] int32
  tensor::Tensor block_tables;  // [batch_size, max_blocks_per_seq] int32
  tensor::Tensor slot_mapping;  // [batch_size] int32: slot = block_id * block_size + offset

  // CPU-side request tracking
  std::vector<base::RequestId> request_ids;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_DECODE_BATCH_H_
