// Builder for GPU/CPU metadata used by mixed prefill+decode serving batches.
#ifndef KUIPER_INCLUDE_SERVING_MIXED_BATCH_BUILDER_H_
#define KUIPER_INCLUDE_SERVING_MIXED_BATCH_BUILDER_H_

#include <memory>
#include <vector>
#include "serving/mixed_batch.h"

namespace base {
class DeviceAllocator;
}  // namespace base

namespace serving {

struct SchedulerOutput;

class MixedBatchBuilder {
 public:
  explicit MixedBatchBuilder(base::KVCacheManager* kv_manager);
  ~MixedBatchBuilder();

  void reserve_metadata_capacity(int32_t token_capacity,
                                 int32_t request_capacity,
                                 int32_t logits_capacity,
                                 int32_t slot_mapping_capacity,
                                 int32_t block_table_entries);

  MixedBatchMetadata build(const SchedulerOutput& output, void* stream);
  MixedBatchMetadata build_decode(const SchedulerOutput& output, void* stream);

 private:
  bool cuda_available() const;
  void ensure_token_capacity(int32_t token_capacity);
  void ensure_request_capacity(int32_t request_capacity);
  void ensure_block_table_capacity(int32_t total_entries);
  void ensure_logits_capacity(int32_t logits_capacity);
  void ensure_slot_mapping_capacity(int32_t slot_capacity);

  void copy_host_to_device(const tensor::Tensor& host_tensor,
                           tensor::Tensor& device_tensor,
                           int32_t count,
                           void* stream) const;

  tensor::Tensor make_int32_view(const tensor::Tensor& storage, int32_t count) const;
  tensor::Tensor make_int32_view(const tensor::Tensor& storage,
                                 int32_t dim0,
                                 int32_t dim1) const;

 private:
  base::KVCacheManager* kv_manager_ = nullptr;
  std::shared_ptr<base::DeviceAllocator> host_alloc_;
  std::shared_ptr<base::DeviceAllocator> device_alloc_;
  bool use_cuda_ = false;

  tensor::Tensor host_token_ids_;
  tensor::Tensor device_token_ids_;
  tensor::Tensor host_positions_;
  tensor::Tensor device_positions_;
  tensor::Tensor host_seq_lens_;
  tensor::Tensor device_seq_lens_;
  tensor::Tensor host_block_tables_;
  tensor::Tensor device_block_tables_;
  tensor::Tensor host_logits_indices_;
  tensor::Tensor device_logits_indices_;
  tensor::Tensor host_slot_mapping_;
  tensor::Tensor device_slot_mapping_;

  std::vector<base::RequestId> request_ids_;
  std::vector<int32_t> tokens_per_request_;
  std::vector<int32_t> decode_row_to_request_;
  std::vector<int32_t> sample_row_to_request_;
  std::vector<int32_t> logits_row_indices_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_MIXED_BATCH_BUILDER_H_
