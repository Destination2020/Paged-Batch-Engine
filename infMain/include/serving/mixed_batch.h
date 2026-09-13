// Mixed batch metadata for unified prefill/decode scheduling
#ifndef KUIPER_INCLUDE_SERVING_MIXED_BATCH_H_
#define KUIPER_INCLUDE_SERVING_MIXED_BATCH_H_

#include <cstdint>
#include <vector>
#include "base/kv_cache_manager.h"
#include "tensor/tensor.h"

namespace serving {

struct SampledTokenView {
  const int32_t* tokens = nullptr;
  int32_t count = 0;

  bool empty() const { return count == 0; }
  int32_t size() const { return count; }
  const int32_t& operator[](int32_t idx) const { return tokens[idx]; }
};

struct MixedBatchMetadata {
  MixedBatchMetadata() {}

  int32_t num_requests = 0;
  int32_t num_tokens = 0;
  int32_t num_decode_tokens = 0;
  int32_t num_prefill_tokens = 0;
  int32_t max_blocks_per_seq = 0;

  // All on GPU
  tensor::Tensor token_ids;       // [num_tokens] int32
  tensor::Tensor positions;       // [num_tokens] int32
  tensor::Tensor mrope_positions; // optional axis-major [3, num_tokens] int32
  tensor::Tensor input_embeddings_override; // optional [num_tokens, hidden_size]
  tensor::Tensor slot_mapping;    // [num_tokens] int32, -1 when not materialized yet
  tensor::Tensor seq_lens;        // [num_requests] int32
  tensor::Tensor seq_start_locs;  // [num_requests + 1] int32
  tensor::Tensor block_tables;    // [num_requests, max_blocks_per_seq] int32
  tensor::Tensor logits_indices;  // [num_logits_tokens] int32

  // CPU-side request tracking
  std::vector<base::RequestId> request_ids;
  std::vector<int32_t> tokens_per_request;
  std::vector<int32_t> decode_row_to_request;
  std::vector<int32_t> sample_row_to_request;
  std::vector<int32_t> logits_row_indices;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_MIXED_BATCH_H_







