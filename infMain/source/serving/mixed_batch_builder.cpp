// Builder for mixed serving batch metadata.
#include "serving/mixed_batch_builder.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <glog/logging.h>
#include "base/alloc.h"
#include "serving/scheduler.h"

namespace {

tensor::Tensor reshape_view(const tensor::Tensor& tensor, const std::vector<int32_t>& dims) {
  tensor::Tensor view = tensor;
  view.reshape_no_realloc(dims);
  return view;
}

void ensure_int32_storage(tensor::Tensor& tensor,
                          int32_t capacity,
                          const std::shared_ptr<base::DeviceAllocator>& alloc,
                          base::DeviceType device_type) {
  if (capacity <= 0) {
    return;
  }
  if (!tensor.is_empty() && static_cast<int32_t>(tensor.size()) >= capacity) {
    return;
  }

  int32_t reserve_capacity = std::max(64, capacity);
  if (!tensor.is_empty()) {
    const int32_t current_capacity = static_cast<int32_t>(tensor.size());
    reserve_capacity = std::max(reserve_capacity, current_capacity);
    while (reserve_capacity < capacity &&
           reserve_capacity <= std::numeric_limits<int32_t>::max() / 2) {
      reserve_capacity *= 2;
    }
    reserve_capacity = std::max(reserve_capacity, capacity);
  } else {
    while (reserve_capacity < capacity &&
           reserve_capacity <= std::numeric_limits<int32_t>::max() / 2) {
      reserve_capacity *= 2;
    }
  }

  tensor = tensor::Tensor(base::DataType::kDataTypeInt32, reserve_capacity, true, alloc);
  tensor.set_device_type(device_type);
}

void append_request_block_table(base::KVCacheManager* kv_manager,
                                base::RequestId request_id,
                                int32_t layer_idx,
                                int32_t max_blocks_per_seq,
                                int32_t* dst) {
  CHECK_NE(kv_manager, nullptr);
  CHECK_NE(dst, nullptr);
  std::fill_n(dst, max_blocks_per_seq, -1);
  const auto& block_ids = kv_manager->get_block_ids(request_id, layer_idx);
  for (int32_t block_idx = 0; block_idx < static_cast<int32_t>(block_ids.size()); ++block_idx) {
    dst[block_idx] = block_ids[block_idx];
  }
}

}  // namespace

namespace serving {

MixedBatchBuilder::MixedBatchBuilder(base::KVCacheManager* kv_manager)
    : kv_manager_(kv_manager),
      host_alloc_(base::PinnedCPUDeviceAllocatorFactory::get_instance()),
      device_alloc_(base::CUDADeviceAllocatorFactory::get_instance()) {
  CHECK_NE(kv_manager_, nullptr);
}

MixedBatchBuilder::~MixedBatchBuilder() = default;

void MixedBatchBuilder::reserve_metadata_capacity(int32_t token_capacity,
                                                  int32_t request_capacity,
                                                  int32_t logits_capacity,
                                                  int32_t slot_mapping_capacity,
                                                  int32_t block_table_entries) {
  ensure_token_capacity(token_capacity);
  ensure_request_capacity(request_capacity);
  ensure_logits_capacity(logits_capacity);
  ensure_slot_mapping_capacity(slot_mapping_capacity);
  ensure_block_table_capacity(block_table_entries);
}

MixedBatchMetadata MixedBatchBuilder::build(const SchedulerOutput& output, void* stream) {
  MixedBatchMetadata batch;
  if (output.scheduled_seqs.empty() || output.total_tokens <= 0) {
    return batch;
  }

  batch.num_requests = static_cast<int32_t>(output.scheduled_seqs.size());
  batch.num_tokens = output.total_tokens;
  batch.num_decode_tokens = output.num_decode_seqs;
  batch.num_prefill_tokens = output.total_tokens - output.num_decode_seqs;

  ensure_token_capacity(batch.num_tokens);
  ensure_request_capacity(batch.num_requests);
  ensure_logits_capacity(batch.num_requests);

  request_ids_.resize(batch.num_requests);
  tokens_per_request_.resize(batch.num_requests);
  decode_row_to_request_.clear();
  sample_row_to_request_.clear();
  logits_row_indices_.clear();
  decode_row_to_request_.reserve(batch.num_decode_tokens);
  sample_row_to_request_.reserve(batch.num_requests);
  logits_row_indices_.reserve(batch.num_requests);

  int32_t* token_ids = host_token_ids_.ptr<int32_t>();
  int32_t* positions = host_positions_.ptr<int32_t>();
  int32_t* seq_lens = host_seq_lens_.ptr<int32_t>();
  int32_t* logits_indices = host_logits_indices_.ptr<int32_t>();

  int32_t row_cursor = 0;
  int32_t logits_count = 0;
  int32_t max_blocks = 0;

  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    auto* seq = output.scheduled_seqs[request_idx];
    CHECK_NE(seq, nullptr);

    const int32_t scheduled_tokens = output.num_tokens_per_seq[request_idx];
    const int32_t current_ctx = kv_manager_->get_context_len(seq->request_id);
    const bool is_decode_request = request_idx < output.num_decode_seqs;

    request_ids_[request_idx] = seq->request_id;
    tokens_per_request_[request_idx] = scheduled_tokens;

    int32_t logical_seq_len = current_ctx;
    if (!is_decode_request) {
      logical_seq_len += scheduled_tokens;
    }
    seq_lens[request_idx] = logical_seq_len;

    const auto& block_ids = kv_manager_->get_block_ids(seq->request_id, 0);
    max_blocks = std::max(max_blocks, static_cast<int32_t>(block_ids.size()));

    if (is_decode_request) {
      CHECK_EQ(scheduled_tokens, 1);
      token_ids[row_cursor] = seq->next_token;
      positions[row_cursor] = current_ctx - 1;

      decode_row_to_request_.push_back(request_idx);
      sample_row_to_request_.push_back(request_idx);
      logits_row_indices_.push_back(row_cursor);
      logits_indices[logits_count++] = row_cursor;
    } else {
      const int32_t start_pos = seq->num_prompt_tokens_computed;
      CHECK_LE(start_pos + scheduled_tokens, seq->prefill_target_tokens());

      for (int32_t token_offset = 0; token_offset < scheduled_tokens; ++token_offset) {
        token_ids[row_cursor + token_offset] = seq->prefill_token_at(start_pos + token_offset);
        positions[row_cursor + token_offset] = start_pos + token_offset;
      }

      if (start_pos + scheduled_tokens == seq->prefill_target_tokens()) {
        const int32_t sample_row = row_cursor + scheduled_tokens - 1;
        sample_row_to_request_.push_back(request_idx);
        logits_row_indices_.push_back(sample_row);
        logits_indices[logits_count++] = sample_row;
      }
    }

    row_cursor += scheduled_tokens;
  }

  CHECK_EQ(row_cursor, batch.num_tokens);

  batch.max_blocks_per_seq = std::max(1, max_blocks);
  const int32_t block_table_entries = batch.num_requests * batch.max_blocks_per_seq;
  ensure_block_table_capacity(block_table_entries);

  int32_t* block_tables = host_block_tables_.ptr<int32_t>();
  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    append_request_block_table(kv_manager_, request_ids_[request_idx], 0,
                               batch.max_blocks_per_seq,
                               block_tables + request_idx * batch.max_blocks_per_seq);
  }

  copy_host_to_device(host_token_ids_, device_token_ids_, batch.num_tokens, stream);
  copy_host_to_device(host_positions_, device_positions_, batch.num_tokens, stream);
  copy_host_to_device(host_seq_lens_, device_seq_lens_, batch.num_requests, stream);
  copy_host_to_device(host_block_tables_, device_block_tables_, block_table_entries, stream);
  copy_host_to_device(host_logits_indices_, device_logits_indices_, logits_count, stream);

  batch.token_ids = make_int32_view(device_token_ids_, batch.num_tokens);
  batch.positions = make_int32_view(device_positions_, batch.num_tokens);
  batch.seq_lens = make_int32_view(device_seq_lens_, batch.num_requests);
  batch.block_tables =
      make_int32_view(device_block_tables_, batch.num_requests, batch.max_blocks_per_seq);
  batch.logits_indices = make_int32_view(device_logits_indices_, logits_count);

  batch.request_ids = request_ids_;
  batch.tokens_per_request = tokens_per_request_;
  batch.decode_row_to_request = decode_row_to_request_;
  batch.sample_row_to_request = sample_row_to_request_;
  batch.logits_row_indices = logits_row_indices_;
  return batch;
}

MixedBatchMetadata MixedBatchBuilder::build_decode(const SchedulerOutput& output, void* stream) {
  MixedBatchMetadata batch;
  if (output.num_decode_seqs <= 0) {
    return batch;
  }

  batch.num_requests = output.num_decode_seqs;
  batch.num_tokens = output.num_decode_seqs;
  batch.num_decode_tokens = output.num_decode_seqs;
  batch.num_prefill_tokens = 0;

  ensure_token_capacity(batch.num_tokens);
  ensure_request_capacity(batch.num_requests);
  ensure_logits_capacity(batch.num_requests);
  ensure_slot_mapping_capacity(batch.num_tokens);

  request_ids_.resize(batch.num_requests);
  tokens_per_request_.resize(batch.num_requests);
  decode_row_to_request_.clear();
  sample_row_to_request_.clear();
  logits_row_indices_.clear();
  decode_row_to_request_.reserve(batch.num_requests);
  sample_row_to_request_.reserve(batch.num_requests);
  logits_row_indices_.reserve(batch.num_requests);

  int32_t* token_ids = host_token_ids_.ptr<int32_t>();
  int32_t* positions = host_positions_.ptr<int32_t>();
  int32_t* seq_lens = host_seq_lens_.ptr<int32_t>();
  int32_t* logits_indices = host_logits_indices_.ptr<int32_t>();
  int32_t* slot_mapping = host_slot_mapping_.ptr<int32_t>();

  int32_t max_blocks = 0;
  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    auto* seq = output.scheduled_seqs[request_idx];
    CHECK_NE(seq, nullptr);

    const int32_t current_ctx = kv_manager_->get_context_len(seq->request_id);
    CHECK_GT(current_ctx, 0);

    request_ids_[request_idx] = seq->request_id;
    tokens_per_request_[request_idx] = 1;
    token_ids[request_idx] = seq->next_token;
    positions[request_idx] = current_ctx - 1;
    seq_lens[request_idx] = current_ctx;
    logits_indices[request_idx] = request_idx;

    auto [block_id, offset] = kv_manager_->get_slot(seq->request_id, 0, current_ctx - 1);
    slot_mapping[request_idx] = block_id * kv_manager_->block_size() + offset;

    decode_row_to_request_.push_back(request_idx);
    sample_row_to_request_.push_back(request_idx);
    logits_row_indices_.push_back(request_idx);
    max_blocks =
        std::max(max_blocks, static_cast<int32_t>(kv_manager_->get_block_ids(seq->request_id, 0).size()));
  }

  batch.max_blocks_per_seq = std::max(1, max_blocks);
  const int32_t block_table_entries = batch.num_requests * batch.max_blocks_per_seq;
  ensure_block_table_capacity(block_table_entries);
  int32_t* block_tables = host_block_tables_.ptr<int32_t>();
  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    append_request_block_table(kv_manager_, request_ids_[request_idx], 0,
                               batch.max_blocks_per_seq,
                               block_tables + request_idx * batch.max_blocks_per_seq);
  }

  copy_host_to_device(host_token_ids_, device_token_ids_, batch.num_tokens, stream);
  copy_host_to_device(host_positions_, device_positions_, batch.num_tokens, stream);
  copy_host_to_device(host_seq_lens_, device_seq_lens_, batch.num_requests, stream);
  copy_host_to_device(host_block_tables_, device_block_tables_, block_table_entries, stream);
  copy_host_to_device(host_logits_indices_, device_logits_indices_, batch.num_requests, stream);
  copy_host_to_device(host_slot_mapping_, device_slot_mapping_, batch.num_tokens, stream);

  batch.token_ids = make_int32_view(device_token_ids_, batch.num_tokens);
  batch.positions = make_int32_view(device_positions_, batch.num_tokens);
  batch.seq_lens = make_int32_view(device_seq_lens_, batch.num_requests);
  batch.slot_mapping = make_int32_view(device_slot_mapping_, batch.num_tokens);
  batch.block_tables =
      make_int32_view(device_block_tables_, batch.num_requests, batch.max_blocks_per_seq);
  batch.logits_indices = make_int32_view(device_logits_indices_, batch.num_requests);

  batch.request_ids = request_ids_;
  batch.tokens_per_request = tokens_per_request_;
  batch.decode_row_to_request = decode_row_to_request_;
  batch.sample_row_to_request = sample_row_to_request_;
  batch.logits_row_indices = logits_row_indices_;
  return batch;
}

void MixedBatchBuilder::ensure_token_capacity(int32_t token_capacity) {
  ensure_int32_storage(host_token_ids_, token_capacity, host_alloc_, base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_token_ids_, token_capacity, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
  ensure_int32_storage(host_positions_, token_capacity, host_alloc_,
                       base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_positions_, token_capacity, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
}

void MixedBatchBuilder::ensure_request_capacity(int32_t request_capacity) {
  ensure_int32_storage(host_seq_lens_, request_capacity, host_alloc_, base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_seq_lens_, request_capacity, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
}

void MixedBatchBuilder::ensure_block_table_capacity(int32_t total_entries) {
  ensure_int32_storage(host_block_tables_, total_entries, host_alloc_,
                       base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_block_tables_, total_entries, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
}

void MixedBatchBuilder::ensure_logits_capacity(int32_t logits_capacity) {
  ensure_int32_storage(host_logits_indices_, logits_capacity, host_alloc_,
                       base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_logits_indices_, logits_capacity, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
}

void MixedBatchBuilder::ensure_slot_mapping_capacity(int32_t slot_capacity) {
  ensure_int32_storage(host_slot_mapping_, slot_capacity, host_alloc_,
                       base::DeviceType::kDeviceCPU);
  ensure_int32_storage(device_slot_mapping_, slot_capacity, device_alloc_,
                       base::DeviceType::kDeviceCUDA);
}

void MixedBatchBuilder::copy_host_to_device(const tensor::Tensor& host_tensor,
                                            tensor::Tensor& device_tensor,
                                            int32_t count,
                                            void* stream) const {
  if (count <= 0) {
    return;
  }
  device_alloc_->memcpy(host_tensor.ptr<int32_t>(), device_tensor.ptr<int32_t>(),
                        static_cast<size_t>(count) * sizeof(int32_t),
                        base::MemcpyKind::kMemcpyCPU2CUDA, stream, false);
}

tensor::Tensor MixedBatchBuilder::make_int32_view(const tensor::Tensor& storage,
                                                  int32_t count) const {
  if (count <= 0) {
    return tensor::Tensor(base::DataType::kDataTypeInt32, 0);
  }
  return reshape_view(storage, {count});
}

tensor::Tensor MixedBatchBuilder::make_int32_view(const tensor::Tensor& storage,
                                                  int32_t dim0,
                                                  int32_t dim1) const {
  if (dim0 <= 0 || dim1 <= 0) {
    return tensor::Tensor(base::DataType::kDataTypeInt32, 0);
  }
  return reshape_view(storage, {dim0, dim1});
}

}  // namespace serving
