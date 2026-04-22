// Batched execution path for Qwen2Model.
#include "model/qwen2.h"
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <glog/logging.h>
#include <op/layer.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <sstream>
#include "base/alloc.h"
#include "base/backend_runtime.h"
#include "base/nvtx_utils.h"
#include "model/paged_kv_runtime.h"
#include "../op/kernels/kernels_interface.h"

namespace model {

namespace {

const tensor::Tensor& get_layer_weight0(const std::shared_ptr<op::Layer>& layer) {
  auto param_layer = std::dynamic_pointer_cast<op::LayerParam>(layer);
  CHECK(param_layer != nullptr) << "Layer is not op::LayerParam, cannot access weight";
  return param_layer->get_weight(0);
}

bool batch_diag_enabled() {
  const char* env = std::getenv("KUIPER_BATCH_DIAG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool detailed_nvtx_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("KUIPER_NVTX_DETAILED");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

std::optional<base::nvtx::ScopedRange> make_detailed_range(bool enabled,
                                                           const char* name,
                                                           uint32_t color) {
  if (!enabled) {
    return std::nullopt;
  }
  return std::optional<base::nvtx::ScopedRange>(std::in_place, name, color);
}

const char* data_type_name(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::kDataTypeFp32:
      return "fp32";
    case base::DataType::kDataTypeBf16:
      return "bf16";
    case base::DataType::kDataTypeInt8:
      return "int8";
    case base::DataType::kDataTypeInt32:
      return "int32";
    default:
      return "unknown";
  }
}

std::string dims_string(const tensor::Tensor& tensor) {
  std::ostringstream os;
  os << "[";
  for (int32_t i = 0; i < tensor.dims_size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << tensor.get_dim(i);
  }
  os << "]";
  return os.str();
}

void sync_queue_or_die(const std::shared_ptr<base::DeviceContext>& context,
                       void* queue,
                       const char* when,
                       int32_t layer,
                       const char* op_name) {
  CHECK(context != nullptr && context->runtime != nullptr);
  auto status = context->runtime->synchronize_queue(queue);
  CHECK(status)
      << "[batch-diag] backend sync failure " << when << " layer=" << layer
      << " op=" << op_name << " msg=" << status.get_err_msg();
}

void* compute_queue_or_die(const std::shared_ptr<base::DeviceContext>& context) {
  CHECK(context != nullptr);
  CHECK_NE(context->compute_queue, nullptr);
  return context->compute_queue;
}

void require_blas_context_or_die(
    const std::shared_ptr<base::DeviceContext>& context) {
  CHECK(context != nullptr);
  CHECK_NE(context->blas_handle, nullptr) << "BLAS handle not initialized";
}

void run_batched_matmul_diag(const char* op_name,
                             int32_t layer,
                             const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& output,
                             int32_t batch_tokens,
                             const std::shared_ptr<base::DeviceContext>& context) {
  const bool diag = batch_diag_enabled();
  void* queue = compute_queue_or_die(context);
  CHECK_EQ(input.dims_size(), 2);
  CHECK_EQ(weight.dims_size(), 2);
  CHECK_EQ(output.dims_size(), 2);

  const int64_t m = batch_tokens;
  const int64_t n = weight.get_dim(0);
  const int64_t k = weight.get_dim(1);

  if (diag) {
    sync_queue_or_die(context, queue, "before", layer, op_name);
    LOG(WARNING) << "[batch-diag] layer=" << layer
                 << " op=" << op_name
                 << " input=" << dims_string(input)
                 << " weight=" << dims_string(weight)
                 << " output=" << dims_string(output)
                  << " dtype=" << data_type_name(input.data_type())
                 << " m=" << m
                 << " n=" << n
                 << " k=" << k;
  }

  kernel::get_matmul_batch_kernel(output.device_type())(
      input, weight, output, batch_tokens, context.get());

  if (diag) {
    sync_queue_or_die(context, queue, "after", layer, op_name);
  }
}

void ensure_int32_storage(tensor::Tensor& tensor,
                          int32_t capacity,
                          const std::shared_ptr<base::DeviceAllocator>& alloc,
                          base::DeviceType device_type) {
  if (capacity <= 0) {
    return;
  }
  if (!tensor.is_empty() && tensor.data_type() == base::DataType::kDataTypeInt32 &&
      static_cast<int32_t>(tensor.size()) >= capacity) {
    return;
  }

  tensor = tensor::Tensor(base::DataType::kDataTypeInt32, capacity, true, alloc);
  tensor.set_device_type(device_type);
}

void check_int32_storage_capacity(const tensor::Tensor& tensor,
                                  int32_t required_capacity,
                                  const char* workspace_name) {
  if (required_capacity <= 0) {
    return;
  }
  CHECK(!tensor.is_empty())
      << workspace_name << " was not preallocated";
  CHECK_EQ(tensor.data_type(), base::DataType::kDataTypeInt32)
      << workspace_name << " must be an int32 workspace";
  CHECK_GE(static_cast<int32_t>(tensor.size()), required_capacity)
      << workspace_name << " capacity is too small: required="
      << required_capacity << ", capacity=" << tensor.size();
}

void copy_int32_host_to_device(const tensor::Tensor& host_tensor,
                               tensor::Tensor& device_tensor,
                               int32_t count,
                               const std::shared_ptr<base::DeviceContext>& context,
                               void* queue) {
  if (count <= 0) {
    return;
  }
  CHECK(context != nullptr && context->runtime != nullptr);
  base::CopyParams params;
  params.src = host_tensor.ptr<int32_t>();
  params.dst = device_tensor.ptr<int32_t>();
  params.byte_size = static_cast<size_t>(count) * sizeof(int32_t);
  params.direction = base::CopyDirection::kHostToDevice;
  params.queue = queue;
  params.need_sync = false;
  auto status = context->runtime->copy(params);
  CHECK(status) << status.get_err_msg();
}

tensor::Tensor reshape_view(const tensor::Tensor& tensor, const std::vector<int32_t>& dims) {
  tensor::Tensor view = tensor;
  view.reshape_no_realloc(dims);
  return view;
}

tensor::Tensor make_int32_view(const tensor::Tensor& storage, int32_t count) {
  if (count <= 0) {
    return tensor::Tensor(base::DataType::kDataTypeInt32, 0);
  }
  return reshape_view(storage, {count});
}

tensor::Tensor make_int32_view(const tensor::Tensor& storage, int32_t dim0, int32_t dim1) {
  if (dim0 <= 0 || dim1 <= 0) {
    return tensor::Tensor(base::DataType::kDataTypeInt32, 0);
  }
  return reshape_view(storage, {dim0, dim1});
}

tensor::Tensor slice_tensor_view(const tensor::Tensor& tensor,
                                 int32_t element_offset,
                                 const std::vector<int32_t>& dims) {
  int32_t numel = 1;
  for (int32_t dim : dims) {
    numel *= dim;
  }
  CHECK_GE(element_offset, 0);
  CHECK_GE(numel, 0);
  CHECK_LE(element_offset + numel, static_cast<int32_t>(tensor.size()));

  const size_t byte_offset =
      static_cast<size_t>(element_offset) * base::DataTypeSize(tensor.data_type());
  const size_t byte_size =
      static_cast<size_t>(numel) * base::DataTypeSize(tensor.data_type());
  auto view_buffer = std::make_shared<base::Buffer>(
      byte_size, nullptr,
      const_cast<void*>(reinterpret_cast<const void*>(tensor.ptr<uint8_t>(byte_offset))),
      true);

  tensor::Tensor view(tensor.data_type(), dims);
  CHECK(view.assign(view_buffer));
  view.set_device_type(tensor.device_type());
  return view;
}

tensor::Tensor row_slice_view(const tensor::Tensor& tensor,
                              int32_t row_start,
                              int32_t row_count,
                              int32_t row_width) {
  return slice_tensor_view(tensor, row_start * row_width, {row_count, row_width});
}

tensor::Tensor flat_slice_view(const tensor::Tensor& tensor, int32_t start, int32_t count) {
  return slice_tensor_view(tensor, start, {count});
}

struct ExpandedRowAttentionMetadata {
  int32_t max_blocks_per_seq = 0;
  int32_t max_prefill_prefix_blocks = 0;
  tensor::Tensor slot_mapping;
  tensor::Tensor seq_lens;
  tensor::Tensor block_tables;
  tensor::Tensor prefill_base_context_lens;
  tensor::Tensor prefill_chunk_row_starts;
  tensor::Tensor prefill_local_token_offsets;
  tensor::Tensor prefill_request_indices;
};

struct RowAttentionWorkspace {
  tensor::Tensor* slot_mapping_host = nullptr;
  tensor::Tensor* slot_mapping_device = nullptr;
  tensor::Tensor* seq_lens_host = nullptr;
  tensor::Tensor* seq_lens_device = nullptr;
  tensor::Tensor* block_tables_host = nullptr;
  tensor::Tensor* block_tables_device = nullptr;
  tensor::Tensor* prefill_base_context_lens_host = nullptr;
  tensor::Tensor* prefill_base_context_lens_device = nullptr;
  tensor::Tensor* prefill_chunk_row_starts_host = nullptr;
  tensor::Tensor* prefill_chunk_row_starts_device = nullptr;
  tensor::Tensor* prefill_local_token_offsets_host = nullptr;
  tensor::Tensor* prefill_local_token_offsets_device = nullptr;
  tensor::Tensor* prefill_request_indices_host = nullptr;
  tensor::Tensor* prefill_request_indices_device = nullptr;
};

ExpandedRowAttentionMetadata build_row_attention_metadata(
    const serving::MixedBatchMetadata& batch,
    base::KVCacheManager* kv_manager,
    const RowAttentionWorkspace& workspace,
    const std::shared_ptr<base::DeviceContext>& context,
    void* queue) {
  base::nvtx::ScopedRange range("row_attention_metadata", base::nvtx::kColorMetadata);
  ExpandedRowAttentionMetadata meta;
  if (batch.num_tokens <= 0) {
    return meta;
  }

  check_int32_storage_capacity(*workspace.slot_mapping_host, batch.num_tokens,
                               "row_attn.slot_mapping.host");
  check_int32_storage_capacity(*workspace.slot_mapping_device, batch.num_tokens,
                               "row_attn.slot_mapping.device");
  check_int32_storage_capacity(*workspace.seq_lens_host, batch.num_tokens,
                               "row_attn.seq_lens.host");
  check_int32_storage_capacity(*workspace.seq_lens_device, batch.num_tokens,
                               "row_attn.seq_lens.device");
  if (batch.num_prefill_tokens > 0) {
    check_int32_storage_capacity(*workspace.prefill_base_context_lens_host,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_base_context_lens.host");
    check_int32_storage_capacity(*workspace.prefill_base_context_lens_device,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_base_context_lens.device");
    check_int32_storage_capacity(*workspace.prefill_chunk_row_starts_host,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_chunk_row_starts.host");
    check_int32_storage_capacity(*workspace.prefill_chunk_row_starts_device,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_chunk_row_starts.device");
    check_int32_storage_capacity(*workspace.prefill_local_token_offsets_host,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_local_token_offsets.host");
    check_int32_storage_capacity(*workspace.prefill_local_token_offsets_device,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_local_token_offsets.device");
    check_int32_storage_capacity(*workspace.prefill_request_indices_host,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_request_indices.host");
    check_int32_storage_capacity(*workspace.prefill_request_indices_device,
                                 batch.num_prefill_tokens,
                                 "row_attn.prefill_request_indices.device");
  }

  int32_t* slot_mapping_host = workspace.slot_mapping_host->ptr<int32_t>();
  int32_t* seq_lens_host = workspace.seq_lens_host->ptr<int32_t>();
  int32_t* prefill_base_context_lens_host =
      batch.num_prefill_tokens > 0 ? workspace.prefill_base_context_lens_host->ptr<int32_t>()
                                   : nullptr;
  int32_t* prefill_chunk_row_starts_host =
      batch.num_prefill_tokens > 0 ? workspace.prefill_chunk_row_starts_host->ptr<int32_t>()
                                   : nullptr;
  int32_t* prefill_local_token_offsets_host =
      batch.num_prefill_tokens > 0 ? workspace.prefill_local_token_offsets_host->ptr<int32_t>()
                                   : nullptr;
  int32_t* prefill_request_indices_host =
      batch.num_prefill_tokens > 0 ? workspace.prefill_request_indices_host->ptr<int32_t>()
                                   : nullptr;

  int32_t max_blocks = 0;
  int32_t row_cursor = 0;
  int32_t prefill_cursor = 0;
  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    const auto request_id = batch.request_ids[request_idx];
    const int32_t scheduled_tokens = batch.tokens_per_request[request_idx];
    const auto& block_ids = kv_manager->get_block_ids(request_id, 0);
    max_blocks = std::max(max_blocks, static_cast<int32_t>(block_ids.size()));

    if (request_idx < batch.num_decode_tokens) {
      CHECK_EQ(scheduled_tokens, 1);
      const int32_t seq_len = kv_manager->get_context_len(request_id);
      seq_lens_host[row_cursor] = seq_len;

      auto [block_id, offset] = kv_manager->get_slot(request_id, 0, seq_len - 1);
      slot_mapping_host[row_cursor] = block_id * kv_manager->block_size() + offset;
      ++row_cursor;
    } else {
      const int32_t seq_len_after_reserve = kv_manager->get_context_len(request_id);
      const int32_t base_ctx = seq_len_after_reserve - scheduled_tokens;
      const int32_t prefill_request_idx = request_idx - batch.num_decode_tokens;
      CHECK_GE(base_ctx, 0);
      CHECK_GE(prefill_request_idx, 0);
      meta.max_prefill_prefix_blocks = std::max(
          meta.max_prefill_prefix_blocks,
          (base_ctx + kv_manager->block_size() - 1) / kv_manager->block_size());
      const int32_t prefill_row_start = row_cursor - batch.num_decode_tokens;
      CHECK_GE(prefill_row_start, 0);
      for (int32_t token_offset = 0; token_offset < scheduled_tokens; ++token_offset) {
        seq_lens_host[row_cursor] = base_ctx + token_offset + 1;
        auto [block_id, offset] =
            kv_manager->get_slot(request_id, 0, base_ctx + token_offset);
        slot_mapping_host[row_cursor] = block_id * kv_manager->block_size() + offset;
        prefill_base_context_lens_host[prefill_cursor] = base_ctx;
        prefill_chunk_row_starts_host[prefill_cursor] = prefill_row_start;
        prefill_local_token_offsets_host[prefill_cursor] = token_offset;
        prefill_request_indices_host[prefill_cursor] = prefill_request_idx;
        ++row_cursor;
        ++prefill_cursor;
      }
    }
  }

  CHECK_EQ(prefill_cursor, batch.num_prefill_tokens);
  CHECK_EQ(row_cursor, batch.num_tokens);

  meta.max_blocks_per_seq = std::max(1, max_blocks);
  const int32_t block_table_entries = batch.num_requests * meta.max_blocks_per_seq;
  check_int32_storage_capacity(*workspace.block_tables_host, block_table_entries,
                               "row_attn.block_tables.host");
  check_int32_storage_capacity(*workspace.block_tables_device, block_table_entries,
                               "row_attn.block_tables.device");
  int32_t* block_tables_host = workspace.block_tables_host->ptr<int32_t>();
  std::fill_n(block_tables_host, block_table_entries, -1);

  for (int32_t request_idx = 0; request_idx < batch.num_requests; ++request_idx) {
    const auto request_id = batch.request_ids[request_idx];
    const auto& block_ids = kv_manager->get_block_ids(request_id, 0);
    for (int32_t block_idx = 0; block_idx < static_cast<int32_t>(block_ids.size()); ++block_idx) {
      block_tables_host[request_idx * meta.max_blocks_per_seq + block_idx] = block_ids[block_idx];
    }
  }

  {
    base::nvtx::ScopedRange copy_range("row_meta_core_h2d", base::nvtx::kColorMemcpy);
    copy_int32_host_to_device(*workspace.slot_mapping_host, *workspace.slot_mapping_device,
                              batch.num_tokens, context, queue);
    copy_int32_host_to_device(*workspace.seq_lens_host, *workspace.seq_lens_device,
                              batch.num_tokens, context, queue);
    copy_int32_host_to_device(*workspace.block_tables_host, *workspace.block_tables_device,
                              block_table_entries, context, queue);
  }

  meta.slot_mapping = make_int32_view(*workspace.slot_mapping_device, batch.num_tokens);
  meta.seq_lens = make_int32_view(*workspace.seq_lens_device, batch.num_tokens);
  meta.block_tables = make_int32_view(*workspace.block_tables_device,
                                      batch.num_requests, meta.max_blocks_per_seq);
  if (batch.num_prefill_tokens > 0) {
    {
      base::nvtx::ScopedRange copy_range("row_meta_prefill_h2d", base::nvtx::kColorMemcpy);
      copy_int32_host_to_device(*workspace.prefill_base_context_lens_host,
                                *workspace.prefill_base_context_lens_device,
                                batch.num_prefill_tokens, context, queue);
      copy_int32_host_to_device(*workspace.prefill_chunk_row_starts_host,
                                *workspace.prefill_chunk_row_starts_device,
                                batch.num_prefill_tokens, context, queue);
      copy_int32_host_to_device(*workspace.prefill_local_token_offsets_host,
                                *workspace.prefill_local_token_offsets_device,
                                batch.num_prefill_tokens, context, queue);
      copy_int32_host_to_device(*workspace.prefill_request_indices_host,
                                *workspace.prefill_request_indices_device,
                                batch.num_prefill_tokens, context, queue);
    }
    meta.prefill_base_context_lens =
        make_int32_view(*workspace.prefill_base_context_lens_device, batch.num_prefill_tokens);
    meta.prefill_chunk_row_starts =
        make_int32_view(*workspace.prefill_chunk_row_starts_device, batch.num_prefill_tokens);
    meta.prefill_local_token_offsets =
        make_int32_view(*workspace.prefill_local_token_offsets_device, batch.num_prefill_tokens);
    meta.prefill_request_indices =
        make_int32_view(*workspace.prefill_request_indices_device, batch.num_prefill_tokens);
  }
  return meta;
}

}  // namespace

base::Status Qwen2Model::forward_mixed_batch(const serving::MixedBatchMetadata& batch) const {
  base::nvtx::ScopedRange range("qwen2_forward_mixed_batch", base::nvtx::kColorForward);
  if (batch.num_tokens <= 0) {
    return base::error::InvalidArgument("Empty batch");
  }
  if (batch.num_prefill_tokens == 0) {
    return forward_decode_batch(batch);
  }
  require_blas_context_or_die(device_context_);
  constexpr int32_t kMaxSplitKVPartitions = 32;

  const int32_t bs = batch.num_tokens;
  CHECK_LE(bs, serving_workspace_token_capacity_)
      << "Mixed batch tokens exceed preallocated serving workspace capacity: batch_tokens="
      << bs << ", serving_workspace_token_capacity=" << serving_workspace_token_capacity_;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->hidden_dim_;
  const int32_t head_num = config_->head_num_;
  const int32_t head_size = config_->head_size_;
  const int32_t kv_mul = config_->kv_mul_;
  const int32_t kv_head_num = config_->kv_head_num_;
  void* queue = compute_queue_or_die(device_context_);

  if (batch.num_prefill_tokens > 0) {
    base::nvtx::ScopedRange reserve_range("prefill_append_slots", base::nvtx::kColorMetadata);
    for (int32_t request_idx = batch.num_decode_tokens; request_idx < batch.num_requests; ++request_idx) {
      const bool ok =
          kv_cache_manager_->append_slots(batch.request_ids[request_idx],
                                          batch.tokens_per_request[request_idx]);
      CHECK(ok) << "Failed to reserve KV slots for prefill request "
                << batch.request_ids[request_idx];
    }
  }

  RowAttentionWorkspace row_workspace{
      &row_attn_workspace_.slot_mapping.host,
      &row_attn_workspace_.slot_mapping.device,
      &row_attn_workspace_.seq_lens.host,
      &row_attn_workspace_.seq_lens.device,
      &row_attn_workspace_.block_tables.host,
      &row_attn_workspace_.block_tables.device,
      &row_attn_workspace_.prefill_base_context_lens.host,
      &row_attn_workspace_.prefill_base_context_lens.device,
      &row_attn_workspace_.prefill_chunk_row_starts.host,
      &row_attn_workspace_.prefill_chunk_row_starts.device,
      &row_attn_workspace_.prefill_local_token_offsets.host,
      &row_attn_workspace_.prefill_local_token_offsets.device,
      &row_attn_workspace_.prefill_request_indices.host,
      &row_attn_workspace_.prefill_request_indices.device,
  };
  const ExpandedRowAttentionMetadata row_meta =
      build_row_attention_metadata(batch, kv_cache_manager_.get(), row_workspace,
                                   device_context_, queue);

  // Get batch-sized buffers (pre-allocated in init_mem with max_batch_size)
  tensor::Tensor& input_emb =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kInputEmbeddings));
  tensor::Tensor& rms_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kOutputRMSNorm));
  tensor::Tensor& query_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kQuery));
  tensor::Tensor& key_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kPagedKeyTemp));
  tensor::Tensor& val_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kPagedValueTemp));
  tensor::Tensor& mha_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kOutputMHA));
  tensor::Tensor& attn_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kAttnOutput));
  tensor::Tensor& w1_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW1Output));
  tensor::Tensor& w3_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW3Output));
  tensor::Tensor& w2_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW2Output));
  tensor::Tensor& ffn_norm_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kFFNRMSNorm));
  tensor::Tensor& fwd_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kForwardOutput));
  tensor::Tensor& splitkv_partial_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialOut));
  tensor::Tensor& splitkv_partial_max =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialMax));
  tensor::Tensor& splitkv_partial_sum =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialSum));

  input_emb.reshape_no_realloc({bs, dim});
  rms_out.reshape_no_realloc({bs, dim});
  query_buf.reshape_no_realloc({bs, dim});
  key_buf.reshape_no_realloc({bs, kv_dim});
  val_buf.reshape_no_realloc({bs, kv_dim});
  mha_out.reshape_no_realloc({bs, dim});
  attn_out.reshape_no_realloc({bs, dim});
  w1_out.reshape_no_realloc({bs, hidden_dim});
  w3_out.reshape_no_realloc({bs, hidden_dim});
  w2_out.reshape_no_realloc({bs, dim});
  ffn_norm_out.reshape_no_realloc({bs, dim});
  fwd_out.reshape_no_realloc({bs, config_->vocab_size_});
  splitkv_partial_out.reshape_no_realloc(
      {bs * head_num * kMaxSplitKVPartitions * head_size});
  splitkv_partial_max.reshape_no_realloc({bs * head_num * kMaxSplitKVPartitions});
  splitkv_partial_sum.reshape_no_realloc({bs * head_num * kMaxSplitKVPartitions});

  // 1. Embedding: [bs] token_ids -> [bs, dim]
  kernel::get_emb_kernel(device_type_)(batch.token_ids,
                                       get_layer_weight0(qwen_layers_->embedding_layer_),
                                       input_emb, std::abs(config_->vocab_size_), queue);

  // 2. Transformer layers
  for (int32_t layer = 0; layer < config_->layer_num_; ++layer) {
    const std::string layer_name = "mixed_layer_" + std::to_string(layer);
    base::nvtx::ScopedRange layer_range(layer_name, base::nvtx::kColorLayer);
    auto& layer_allocator = kv_cache_manager_->allocator_mut(layer);

    // 2a. Attention RMSNorm: [bs, dim] -> [bs, dim]
    const auto& rms_weight = get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer));
    kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, rms_weight, rms_out, dim, queue);

    // 2b. Wq: [bs, dim] -> [bs, dim]
    const auto& wq_weight = get_layer_weight0(qwen_layers_->wq_layers_.at(layer));
    run_batched_matmul_diag("wq", layer, rms_out, wq_weight, query_buf, bs,
                            device_context_);

    // 2c. Wk: [bs, dim] -> [bs, kv_dim]
    const auto& wk_weight = get_layer_weight0(qwen_layers_->wk_layers_.at(layer));
    run_batched_matmul_diag("wk", layer, rms_out, wk_weight, key_buf, bs,
                            device_context_);

    // 2d. Wv: [bs, dim] -> [bs, kv_dim]
    const auto& wv_weight = get_layer_weight0(qwen_layers_->wv_layers_.at(layer));
    run_batched_matmul_diag("wv", layer, rms_out, wv_weight, val_buf, bs,
                            device_context_);

    // 2e. Bias add (if applicable)
    auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_.at(layer));
    auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_.at(layer));
    auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_.at(layer));
    // Qwen2 has bias on q/k/v
    if (wq_matmul && !wq_matmul->get_bias(0).is_empty()) {
      kernel::get_add_bias_kernel(device_type_)(query_buf, wq_matmul->get_bias(0), bs, dim,
                                                queue);
    }
    if (wk_matmul && !wk_matmul->get_bias(0).is_empty()) {
      kernel::get_add_bias_kernel(device_type_)(key_buf, wk_matmul->get_bias(0), bs, kv_dim,
                                                queue);
    }
    if (wv_matmul && !wv_matmul->get_bias(0).is_empty()) {
      kernel::get_add_bias_kernel(device_type_)(val_buf, wv_matmul->get_bias(0), bs, kv_dim,
                                                queue);
    }

    // 2f. Batched RoPE
    kernel::get_rope_batch_kernel(device_type_)(dim, kv_dim, head_size,
                                                 query_buf, key_buf, batch.positions,
                                                 get_buffer(ModelBufferType::kSinCache),
                                                 get_buffer(ModelBufferType::kCosCache),
                                                 bs, queue);

    // 2g. Batched scatter KV to pages
    CHECK(paged_kv_runtime_ != nullptr);
    paged_kv_runtime_->scatter(
        key_buf, val_buf, layer_allocator, row_meta.slot_mapping,
        model_block_size, kv_head_num, head_size, bs);

    // 2h. Mixed attention split:
    // decode rows keep the fast decode path; prefill rows use a dedicated
    // chunked-prefill kernel with prefix + chunk-local causal attention.
    if (batch.num_decode_tokens > 0) {
      tensor::Tensor decode_query =
          row_slice_view(query_buf, 0, batch.num_decode_tokens, dim);
      tensor::Tensor decode_mha_out =
          row_slice_view(mha_out, 0, batch.num_decode_tokens, dim);
      tensor::Tensor decode_seq_lens =
          flat_slice_view(row_meta.seq_lens, 0, batch.num_decode_tokens);
      tensor::Tensor decode_block_tables =
          row_slice_view(row_meta.block_tables, 0, batch.num_decode_tokens,
                         row_meta.max_blocks_per_seq);

      PagedKVDecodeRuntimeArgs decode_args;
      decode_args.batch_size = batch.num_decode_tokens;
      decode_args.head_num = head_num;
      decode_args.head_size = head_size;
      decode_args.kv_mul = kv_mul;
      decode_args.max_blocks_per_seq = row_meta.max_blocks_per_seq;
      decode_args.block_size = model_block_size;
      decode_args.num_kv_heads = kv_head_num;
      decode_args.queries = &decode_query;
      decode_args.outputs = &decode_mha_out;
      decode_args.block_tables = &decode_block_tables;
      decode_args.seq_lens = &decode_seq_lens;
      decode_args.partial_out = &splitkv_partial_out;
      decode_args.partial_max = &splitkv_partial_max;
      decode_args.partial_sum = &splitkv_partial_sum;
      CHECK(paged_kv_runtime_ != nullptr);
      paged_kv_runtime_->decode(layer_allocator, decode_args);
    }

    if (batch.num_prefill_tokens > 0) {
      const int32_t num_prefill_requests = batch.num_requests - batch.num_decode_tokens;
      tensor::Tensor prefill_query =
          row_slice_view(query_buf, batch.num_decode_tokens, batch.num_prefill_tokens, dim);
      tensor::Tensor prefill_key =
          row_slice_view(key_buf, batch.num_decode_tokens, batch.num_prefill_tokens, kv_dim);
      tensor::Tensor prefill_value =
          row_slice_view(val_buf, batch.num_decode_tokens, batch.num_prefill_tokens, kv_dim);
      tensor::Tensor prefill_mha_out =
          row_slice_view(mha_out, batch.num_decode_tokens, batch.num_prefill_tokens, dim);
      tensor::Tensor prefill_block_tables =
          row_slice_view(row_meta.block_tables, batch.num_decode_tokens, num_prefill_requests,
                         row_meta.max_blocks_per_seq);

      PagedKVPrefillRuntimeArgs prefill_args;
      prefill_args.batch_size = batch.num_prefill_tokens;
      prefill_args.head_num = head_num;
      prefill_args.head_size = head_size;
      prefill_args.kv_mul = kv_mul;
      prefill_args.max_blocks_per_seq = row_meta.max_blocks_per_seq;
      prefill_args.max_prefix_blocks = row_meta.max_prefill_prefix_blocks;
      prefill_args.block_size = model_block_size;
      prefill_args.num_kv_heads = kv_head_num;
      prefill_args.queries = &prefill_query;
      prefill_args.chunk_keys = &prefill_key;
      prefill_args.chunk_values = &prefill_value;
      prefill_args.outputs = &prefill_mha_out;
      prefill_args.block_tables = &prefill_block_tables;
      prefill_args.request_indices = &row_meta.prefill_request_indices;
      prefill_args.base_context_lens = &row_meta.prefill_base_context_lens;
      prefill_args.chunk_row_starts = &row_meta.prefill_chunk_row_starts;
      prefill_args.local_token_offsets = &row_meta.prefill_local_token_offsets;
      prefill_args.partial_out = &splitkv_partial_out;
      prefill_args.partial_max = &splitkv_partial_max;
      prefill_args.partial_sum = &splitkv_partial_sum;
      CHECK(paged_kv_runtime_ != nullptr);
      paged_kv_runtime_->prefill(layer_allocator, prefill_args);
    }

    // 2i. Wo: [bs, dim] -> [bs, dim]
    const auto& wo_weight = get_layer_weight0(qwen_layers_->wo_layers_.at(layer));
    run_batched_matmul_diag("wo", layer, mha_out, wo_weight, attn_out, bs,
                            device_context_);

    // 2j. Residual add: input_emb += attn_out (element-wise, size=bs*dim)
    kernel::get_add_kernel(device_type_)(input_emb, attn_out, input_emb, queue);

    // 2k. FFN RMSNorm
    const auto& ffn_rms_weight =
      get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer + config_->layer_num_));
    kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, ffn_rms_weight, ffn_norm_out, dim,
                                                 queue);

    // 2l. W1: [bs, dim] -> [bs, hidden_dim]
    const auto& w1_weight = get_layer_weight0(qwen_layers_->w1_layers_.at(layer));
    run_batched_matmul_diag("w1", layer, ffn_norm_out, w1_weight, w1_out, bs,
                            device_context_);

    // 2m. W3: [bs, dim] -> [bs, hidden_dim]
    const auto& w3_weight = get_layer_weight0(qwen_layers_->w3_layers_.at(layer));
    run_batched_matmul_diag("w3", layer, ffn_norm_out, w3_weight, w3_out, bs,
                            device_context_);

    // 2n. SwiGLU: element-wise on [bs * hidden_dim]
    kernel::get_swiglu_kernel(device_type_)(w1_out, w3_out, w1_out, queue);

    // 2o. W2: [bs, hidden_dim] -> [bs, dim]
    const auto& w2_weight = get_layer_weight0(qwen_layers_->w2_layers_.at(layer));
    run_batched_matmul_diag("w2", layer, w1_out, w2_weight, w2_out, bs,
                            device_context_);

    // 2p. Residual add
    kernel::get_add_kernel(device_type_)(input_emb, w2_out, input_emb, queue);
  }

  // 3. Final RMSNorm
  {
    base::nvtx::ScopedRange final_norm_range("mixed_final_rmsnorm", base::nvtx::kColorForward);
    const auto& final_rms_weight =
        get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_));
    kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, final_rms_weight, input_emb, dim,
                                                 queue);
  }

  // 4. Cls logits: [bs, dim] -> [bs, vocab_size]
  {
    base::nvtx::ScopedRange cls_range("mixed_cls_logits", base::nvtx::kColorForward);
    const auto& cls_weight = get_layer_weight0(qwen_layers_->cls_layer_);
    run_batched_matmul_diag("cls", config_->layer_num_, input_emb, cls_weight, fwd_out, bs,
                            device_context_);
  }

  return base::error::Success();
}

base::Status Qwen2Model::forward_decode_batch(const serving::MixedBatchMetadata& batch) const {
  base::nvtx::ScopedRange range("qwen2_forward_decode_batch", base::nvtx::kColorForward);
  if (batch.num_tokens <= 0) {
    return base::error::InvalidArgument("Empty batch");
  }
  CHECK_EQ(batch.num_prefill_tokens, 0);
  CHECK_EQ(batch.num_decode_tokens, batch.num_tokens);
  CHECK_EQ(batch.num_requests, batch.num_tokens);
  CHECK(!batch.slot_mapping.is_empty());
  CHECK(!batch.seq_lens.is_empty());
  CHECK(!batch.block_tables.is_empty());
  require_blas_context_or_die(device_context_);

  constexpr int32_t kMaxSplitKVPartitions = 32;
  const int32_t bs = batch.num_tokens;
  CHECK_LE(bs, serving_workspace_token_capacity_)
      << "Decode batch tokens exceed preallocated serving workspace capacity: batch_tokens="
      << bs << ", serving_workspace_token_capacity=" << serving_workspace_token_capacity_;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->hidden_dim_;
  const int32_t head_num = config_->head_num_;
  const int32_t head_size = config_->head_size_;
  const int32_t kv_mul = config_->kv_mul_;
  const int32_t kv_head_num = config_->kv_head_num_;
  void* queue = compute_queue_or_die(device_context_);

  tensor::Tensor& input_emb =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kInputEmbeddings));
  tensor::Tensor& rms_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kOutputRMSNorm));
  tensor::Tensor& query_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kQuery));
  tensor::Tensor& key_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kPagedKeyTemp));
  tensor::Tensor& val_buf =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kPagedValueTemp));
  tensor::Tensor& mha_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kOutputMHA));
  tensor::Tensor& attn_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kAttnOutput));
  tensor::Tensor& w1_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW1Output));
  tensor::Tensor& w3_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW3Output));
  tensor::Tensor& w2_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kW2Output));
  tensor::Tensor& ffn_norm_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kFFNRMSNorm));
  tensor::Tensor& fwd_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kForwardOutput));
  tensor::Tensor& splitkv_partial_out =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialOut));
  tensor::Tensor& splitkv_partial_max =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialMax));
  tensor::Tensor& splitkv_partial_sum =
      const_cast<tensor::Tensor&>(get_buffer(ModelBufferType::kSplitKVPartialSum));

  input_emb.reshape_no_realloc({bs, dim});
  rms_out.reshape_no_realloc({bs, dim});
  query_buf.reshape_no_realloc({bs, dim});
  key_buf.reshape_no_realloc({bs, kv_dim});
  val_buf.reshape_no_realloc({bs, kv_dim});
  mha_out.reshape_no_realloc({bs, dim});
  attn_out.reshape_no_realloc({bs, dim});
  w1_out.reshape_no_realloc({bs, hidden_dim});
  w3_out.reshape_no_realloc({bs, hidden_dim});
  w2_out.reshape_no_realloc({bs, dim});
  ffn_norm_out.reshape_no_realloc({bs, dim});
  fwd_out.reshape_no_realloc({bs, config_->vocab_size_});
  splitkv_partial_out.reshape_no_realloc(
      {bs * head_num * kMaxSplitKVPartitions * head_size});
  splitkv_partial_max.reshape_no_realloc({bs * head_num * kMaxSplitKVPartitions});
  splitkv_partial_sum.reshape_no_realloc({bs * head_num * kMaxSplitKVPartitions});

  const bool detailed_nvtx = detailed_nvtx_enabled();

  {
    auto embed_range =
        make_detailed_range(detailed_nvtx, "decode_embed", base::nvtx::kColorMetadata);
    kernel::get_emb_kernel(device_type_)(batch.token_ids,
                                         get_layer_weight0(qwen_layers_->embedding_layer_),
                                         input_emb, std::abs(config_->vocab_size_), queue);
  }

  for (int32_t layer = 0; layer < config_->layer_num_; ++layer) {
    const std::string layer_name = "decode_layer_" + std::to_string(layer);
    base::nvtx::ScopedRange layer_range(layer_name, base::nvtx::kColorLayer);
    auto& layer_allocator = kv_cache_manager_->allocator_mut(layer);

    {
      auto rms_range =
          make_detailed_range(detailed_nvtx, "decode_rmsnorm", base::nvtx::kColorMetadata);
      const auto& rms_weight = get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer));
      kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, rms_weight, rms_out, dim, queue);
    }

    {
      auto qkv_range =
          make_detailed_range(detailed_nvtx, "decode_qkv", base::nvtx::kColorForward);
      const auto& wq_weight = get_layer_weight0(qwen_layers_->wq_layers_.at(layer));
      run_batched_matmul_diag("wq", layer, rms_out, wq_weight, query_buf, bs,
                              device_context_);

      const auto& wk_weight = get_layer_weight0(qwen_layers_->wk_layers_.at(layer));
      run_batched_matmul_diag("wk", layer, rms_out, wk_weight, key_buf, bs,
                              device_context_);

      const auto& wv_weight = get_layer_weight0(qwen_layers_->wv_layers_.at(layer));
      run_batched_matmul_diag("wv", layer, rms_out, wv_weight, val_buf, bs,
                              device_context_);
    }

    {
      auto bias_rope_range =
          make_detailed_range(detailed_nvtx, "decode_bias_rope", base::nvtx::kColorProcess);
      auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_.at(layer));
      auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_.at(layer));
      auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_.at(layer));
      if (wq_matmul && !wq_matmul->get_bias(0).is_empty()) {
        kernel::get_add_bias_kernel(device_type_)(query_buf, wq_matmul->get_bias(0), bs, dim,
                                                  queue);
      }
      if (wk_matmul && !wk_matmul->get_bias(0).is_empty()) {
        kernel::get_add_bias_kernel(device_type_)(key_buf, wk_matmul->get_bias(0), bs, kv_dim,
                                                  queue);
      }
      if (wv_matmul && !wv_matmul->get_bias(0).is_empty()) {
        kernel::get_add_bias_kernel(device_type_)(val_buf, wv_matmul->get_bias(0), bs, kv_dim,
                                                  queue);
      }

      kernel::get_rope_batch_kernel(device_type_)(dim, kv_dim, head_size,
                                                   query_buf, key_buf, batch.positions,
                                                   get_buffer(ModelBufferType::kSinCache),
                                                   get_buffer(ModelBufferType::kCosCache),
                                                   bs, queue);
    }

    {
      auto scatter_range =
          make_detailed_range(detailed_nvtx, "decode_scatter_kv", base::nvtx::kColorMemcpy);
      CHECK(paged_kv_runtime_ != nullptr);
      paged_kv_runtime_->scatter(
          key_buf, val_buf, layer_allocator, batch.slot_mapping,
          model_block_size, kv_head_num, head_size, bs);
    }

    {
      auto attention_range =
          make_detailed_range(detailed_nvtx, "decode_attention", base::nvtx::kColorStep);
      PagedKVDecodeRuntimeArgs decode_args;
      decode_args.batch_size = bs;
      decode_args.head_num = head_num;
      decode_args.head_size = head_size;
      decode_args.kv_mul = kv_mul;
      decode_args.max_blocks_per_seq = batch.max_blocks_per_seq;
      decode_args.block_size = model_block_size;
      decode_args.num_kv_heads = kv_head_num;
      decode_args.queries = &query_buf;
      decode_args.outputs = &mha_out;
      decode_args.block_tables = &batch.block_tables;
      decode_args.seq_lens = &batch.seq_lens;
      decode_args.partial_out = &splitkv_partial_out;
      decode_args.partial_max = &splitkv_partial_max;
      decode_args.partial_sum = &splitkv_partial_sum;
      CHECK(paged_kv_runtime_ != nullptr);
      paged_kv_runtime_->decode(layer_allocator, decode_args);
    }

    {
      auto wo_range =
          make_detailed_range(detailed_nvtx, "decode_wo_residual", base::nvtx::kColorForward);
      const auto& wo_weight = get_layer_weight0(qwen_layers_->wo_layers_.at(layer));
      run_batched_matmul_diag("wo", layer, mha_out, wo_weight, attn_out, bs,
                              device_context_);
      kernel::get_add_kernel(device_type_)(input_emb, attn_out, input_emb, queue);
    }

    {
      auto ffn_norm_range =
          make_detailed_range(detailed_nvtx, "decode_ffn_norm", base::nvtx::kColorMetadata);
      const auto& ffn_rms_weight =
          get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer + config_->layer_num_));
      kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, ffn_rms_weight, ffn_norm_out, dim,
                                                   queue);
    }

    {
      auto ffn_up_range =
          make_detailed_range(detailed_nvtx, "decode_ffn_up", base::nvtx::kColorForward);
      const auto& w1_weight = get_layer_weight0(qwen_layers_->w1_layers_.at(layer));
      run_batched_matmul_diag("w1", layer, ffn_norm_out, w1_weight, w1_out, bs,
                              device_context_);

      const auto& w3_weight = get_layer_weight0(qwen_layers_->w3_layers_.at(layer));
      run_batched_matmul_diag("w3", layer, ffn_norm_out, w3_weight, w3_out, bs,
                              device_context_);
    }

    {
      auto swiglu_range =
          make_detailed_range(detailed_nvtx, "decode_swiglu", base::nvtx::kColorProcess);
      kernel::get_swiglu_kernel(device_type_)(w1_out, w3_out, w1_out, queue);
    }

    {
      auto ffn_down_range = make_detailed_range(detailed_nvtx,
                                                "decode_ffn_down_residual",
                                                base::nvtx::kColorForward);
      const auto& w2_weight = get_layer_weight0(qwen_layers_->w2_layers_.at(layer));
      run_batched_matmul_diag("w2", layer, w1_out, w2_weight, w2_out, bs,
                              device_context_);
      kernel::get_add_kernel(device_type_)(input_emb, w2_out, input_emb, queue);
    }
  }

  {
    base::nvtx::ScopedRange final_norm_range("decode_final_rmsnorm", base::nvtx::kColorForward);
    const auto& final_rms_weight =
        get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_));
    kernel::get_rmsnorm_dim_kernel(device_type_)(input_emb, final_rms_weight, input_emb, dim,
                                                 queue);
  }

  {
    base::nvtx::ScopedRange cls_range("decode_cls_logits", base::nvtx::kColorForward);
    const auto& cls_weight = get_layer_weight0(qwen_layers_->cls_layer_);
    run_batched_matmul_diag("cls", config_->layer_num_, input_emb, cls_weight, fwd_out, bs,
                            device_context_);
  }

  return base::error::Success();
}

}  // namespace model
