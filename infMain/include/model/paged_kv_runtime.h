#ifndef KUIPER_INCLUDE_MODEL_PAGED_KV_RUNTIME_H_
#define KUIPER_INCLUDE_MODEL_PAGED_KV_RUNTIME_H_

#include <memory>
#include "base/block_allocator.h"
#include "base/device_context.h"
#include "serving/mixed_batch.h"
#include "tensor/tensor.h"

namespace model {

struct PagedKVDecodeRuntimeArgs {
  int32_t batch_size = 0;
  int32_t head_num = 0;
  int32_t head_size = 0;
  int32_t kv_mul = 0;
  int32_t max_blocks_per_seq = 0;
  int32_t block_size = 0;
  int32_t num_kv_heads = 0;
  const tensor::Tensor* queries = nullptr;
  tensor::Tensor* outputs = nullptr;
  const tensor::Tensor* block_tables = nullptr;
  const tensor::Tensor* seq_lens = nullptr;
  const tensor::Tensor* partial_out = nullptr;
  const tensor::Tensor* partial_max = nullptr;
  const tensor::Tensor* partial_sum = nullptr;
};

struct PagedKVPrefillRuntimeArgs {
  int32_t batch_size = 0;
  int32_t head_num = 0;
  int32_t head_size = 0;
  int32_t kv_mul = 0;
  int32_t max_blocks_per_seq = 0;
  int32_t max_prefix_blocks = 0;
  int32_t block_size = 0;
  int32_t num_kv_heads = 0;
  const tensor::Tensor* queries = nullptr;
  const tensor::Tensor* chunk_keys = nullptr;
  const tensor::Tensor* chunk_values = nullptr;
  tensor::Tensor* outputs = nullptr;
  const tensor::Tensor* block_tables = nullptr;
  const tensor::Tensor* request_indices = nullptr;
  const tensor::Tensor* base_context_lens = nullptr;
  const tensor::Tensor* chunk_row_starts = nullptr;
  const tensor::Tensor* local_token_offsets = nullptr;
  tensor::Tensor* partial_out = nullptr;
  tensor::Tensor* partial_max = nullptr;
  tensor::Tensor* partial_sum = nullptr;
};

class PagedKVRuntime {
 public:
  virtual ~PagedKVRuntime() = default;

  virtual void scatter(
      const tensor::Tensor& key_tensor,
      const tensor::Tensor& value_tensor,
      const base::KVPoolView& pool,
      const tensor::Tensor& slot_mapping,
      int32_t block_size,
      int32_t num_kv_heads,
      int32_t head_size,
      int32_t batch_tokens) const = 0;

  virtual void scatter_single_token(
      const tensor::Tensor& key_tensor,
      const tensor::Tensor& value_tensor,
      const base::KVPoolView& pool,
      int32_t physical_block_id,
      int32_t offset_in_block,
      int32_t block_size,
      int32_t num_kv_heads,
      int32_t head_size) const = 0;

  virtual bool decode(const base::KVPoolView& pool,
                      const PagedKVDecodeRuntimeArgs& args) const = 0;

  virtual void prefill(const base::KVPoolView& pool,
                       const PagedKVPrefillRuntimeArgs& args) const = 0;
};

std::shared_ptr<PagedKVRuntime> create_paged_kv_runtime(
    base::DeviceType device_type,
    const std::shared_ptr<base::DeviceContext>& device_context);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_PAGED_KV_RUNTIME_H_
