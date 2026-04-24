// Updated on March 31, 2026
#include "model/qwen2.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <numeric>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include <vector>
#include "base/bf16.h"
#include "base/nvtx_utils.h"
#include "base/tick.h"
#include "model/paged_kv_runtime.h"
#include "model/serving_memory_planner.h"
#include "../op/kernels/kernels_interface.h"
namespace model {

namespace {
constexpr int32_t kServingMaxSplitKVPartitions = 32;
constexpr int32_t kMinDynamicKVBlocks = model_max_batch_size;

void prepare_layer_for_runtime(const std::shared_ptr<op::Layer>& layer,
                               const std::shared_ptr<base::DeviceContext>& context,
                               base::DataType runtime_data_type) {
  if (!layer) {
    return;
  }
  layer->set_device_context(context);
  layer->set_data_type(runtime_data_type);
  layer->materialize();
}

tensor::Tensor reshape_view(const tensor::Tensor& tensor, const std::vector<int32_t>& dims) {
  tensor::Tensor view = tensor;
  view.reshape_no_realloc(dims);
  return view;
}

tensor::Tensor first_row_view(const tensor::Tensor& tensor, int32_t width) {
  CHECK_GE(static_cast<int32_t>(tensor.size()), width);
  return reshape_view(tensor, {width});
}

bool batch_sample_cpu_fallback_enabled() {
  const char* env = std::getenv("KUIPER_BATCH_SAMPLE_CPU_FALLBACK");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

template <typename T>
float host_logit_to_float(T value);

template <>
float host_logit_to_float<float>(float value) {
  return value;
}

template <>
float host_logit_to_float<uint16_t>(uint16_t value) {
  return base::bf16_bits_to_float(value);
}

template <typename T>
void argmax_host_rows_into(const T* logits,
                           int32_t num_rows,
                           int32_t vocab_size,
                           const std::vector<int32_t>& row_indices,
                           int32_t* results) {
  CHECK(results != nullptr);
  for (size_t sample_idx = 0; sample_idx < row_indices.size(); ++sample_idx) {
    const int32_t row_idx = row_indices[sample_idx];
    CHECK_GE(row_idx, 0);
    CHECK_LT(row_idx, num_rows);

    const T* row = logits + static_cast<int64_t>(row_idx) * vocab_size;
    int32_t best = 0;
    float best_val = host_logit_to_float(row[0]);
    for (int32_t vocab_idx = 1; vocab_idx < vocab_size; ++vocab_idx) {
      const float value = host_logit_to_float(row[vocab_idx]);
      if (value > best_val) {
        best_val = value;
        best = vocab_idx;
      }
    }
    results[sample_idx] = best;
  }
}

void ensure_tensor_storage(tensor::Tensor& tensor,
                           base::DataType data_type,
                           int32_t capacity,
                           const std::shared_ptr<base::DeviceAllocator>& alloc,
                           base::DeviceType device_type) {
  if (capacity <= 0) {
    return;
  }
  if (!tensor.is_empty() && tensor.data_type() == data_type &&
      static_cast<int32_t>(tensor.size()) >= capacity) {
    return;
  }

  tensor = tensor::Tensor(data_type, capacity, true, alloc);
  tensor.set_device_type(device_type);
}

void check_tensor_storage_capacity(const tensor::Tensor& tensor,
                                   base::DataType data_type,
                                   int32_t required_capacity,
                                   const char* workspace_name) {
  if (required_capacity <= 0) {
    return;
  }
  CHECK(!tensor.is_empty())
      << workspace_name << " was not preallocated";
  CHECK_EQ(tensor.data_type(), data_type)
      << workspace_name << " has an unexpected data type";
  CHECK_GE(static_cast<int32_t>(tensor.size()), required_capacity)
      << workspace_name << " capacity is too small: required="
      << required_capacity << ", capacity=" << tensor.size();
}

void runtime_copy_or_die(const std::shared_ptr<base::DeviceContext>& context,
                         const void* src,
                         void* dst,
                         size_t byte_size,
                         base::CopyDirection direction,
                         void* queue,
                         bool need_sync) {
  CHECK(context != nullptr && context->runtime != nullptr);
  base::CopyParams params;
  params.src = src;
  params.dst = dst;
  params.byte_size = byte_size;
  params.direction = direction;
  params.queue = queue;
  params.need_sync = need_sync;
  auto status = context->runtime->copy(params);
  CHECK(status) << status.get_err_msg();
}

void* compute_queue_or_null(const std::shared_ptr<base::DeviceContext>& context) {
  return context != nullptr ? context->compute_queue : nullptr;
}

void* compute_queue_or_die(const std::shared_ptr<base::DeviceContext>& context) {
  CHECK(context != nullptr);
  CHECK_NE(context->compute_queue, nullptr);
  return context->compute_queue;
}

void require_compute_queue_or_die(const std::shared_ptr<base::DeviceContext>& context) {
  CHECK(context != nullptr);
  CHECK_NE(context->compute_queue, nullptr) << "Compute queue not initialized";
}

std::shared_ptr<base::DeviceAllocator> host_allocator_for(
    const std::shared_ptr<base::DeviceContext>& context) {
  if (context != nullptr && context->runtime != nullptr) {
    auto allocator = context->runtime->host_allocator();
    if (allocator != nullptr) {
      return allocator;
    }
  }
  return base::CPUDeviceAllocatorFactory::get_instance();
}

std::shared_ptr<base::DeviceAllocator> pinned_host_allocator_for(
    const std::shared_ptr<base::DeviceContext>& context) {
  if (context != nullptr && context->runtime != nullptr) {
    auto allocator = context->runtime->pinned_host_allocator();
    if (allocator != nullptr) {
      return allocator;
    }
  }
  return base::PinnedCPUDeviceAllocatorFactory::get_instance();
}

std::shared_ptr<base::DeviceAllocator> device_allocator_for(
    base::DeviceType device_type,
    const std::shared_ptr<base::DeviceContext>& context) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return host_allocator_for(context);
  }
  if (context != nullptr && context->runtime != nullptr) {
    auto allocator = context->runtime->device_allocator();
    if (allocator != nullptr) {
      return allocator;
    }
  }
  CHECK_EQ(device_type, base::DeviceType::kDeviceCUDA)
      << "Missing backend runtime device allocator for unsupported device type "
      << static_cast<int>(device_type);
  return base::CUDADeviceAllocatorFactory::get_instance();
}
}  // namespace

void Qwen2Layers::materialize(std::shared_ptr<base::DeviceContext> context,
                              base::DataType runtime_data_type) {
  prepare_layer_for_runtime(add_layer_, context, runtime_data_type);
  prepare_layer_for_runtime(rope_layer_, context, runtime_data_type);
  prepare_layer_for_runtime(swiglu_layer_, context, runtime_data_type);
  prepare_layer_for_runtime(cls_layer_, context, runtime_data_type);
  prepare_layer_for_runtime(embedding_layer_, context, runtime_data_type);
  prepare_layer_for_runtime(mha_layer_, context, runtime_data_type);

  for (auto& weight_layer : wq_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : wk_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : wv_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : wo_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : w1_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : w2_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& weight_layer : w3_layers_) {
    prepare_layer_for_runtime(weight_layer, context, runtime_data_type);
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    prepare_layer_for_runtime(rms_norm_layer, context, runtime_data_type);
  }
}

Qwen2Model::Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

void runtime_status_or_die(const base::Status& status, const char* when) {
  CHECK(status) << when << ": " << status.get_err_msg();
}

Qwen2Model::~Qwen2Model() = default;

void Qwen2Model::set_kv_cache_memory_utilization(double utilization) {
  if (!std::isfinite(utilization)) {
    LOG(WARNING) << "Ignoring non-finite KV cache memory utilization: " << utilization;
    return;
  }
  kv_cache_memory_utilization_ =
      clamp_kv_cache_memory_utilization(utilization);
  if (kv_cache_memory_utilization_ != utilization) {
    LOG(WARNING) << "Clamped KV cache memory utilization from " << utilization
                 << " to " << kv_cache_memory_utilization_;
  }
}

void Qwen2Model::set_radix_cache_enabled(bool enable) {
  radix_cache_enabled_override_set_ = true;
  radix_cache_enabled_override_ = enable;
}

void Qwen2Model::set_kv_cache_gpu_memory_utilization(double utilization) {
  set_kv_cache_memory_utilization(utilization);
}

ServingWorkspaceProfile Qwen2Model::serving_workspace_profile(
    base::DataType runtime_data_type) const {
  CHECK(config_ != nullptr);
  // Qwen2-specific serving workspace profile. The generic planner consumes
  // the aggregated bytes-per-token number, while each model describes its own
  // activation and auxiliary scratch footprint here.
  const size_t runtime_dtype_bytes = base::DataTypeSize(runtime_data_type);
  const size_t dim = static_cast<size_t>(config_->dim_);
  const size_t hidden = static_cast<size_t>(config_->hidden_dim_);
  const size_t kv_dim = static_cast<size_t>(config_->kv_dim_);
  const size_t vocab = static_cast<size_t>(config_->vocab_size_);
  const size_t head_num = static_cast<size_t>(config_->head_num_);
  const size_t head_size = static_cast<size_t>(config_->head_size_);

  ServingWorkspaceProfile profile;
  profile.activation_bytes_per_token =
      (7u * dim + 2u * hidden + 2u * kv_dim + vocab) * runtime_dtype_bytes;
  profile.aux_bytes_per_token =
      head_num * static_cast<size_t>(kServingMaxSplitKVPartitions) * head_size * sizeof(float) +
      2u * head_num * static_cast<size_t>(kServingMaxSplitKVPartitions) * sizeof(float);
  return profile;
}

void Qwen2Model::set_serving_workspace_token_capacity(int32_t token_capacity) {
  serving_workspace_token_capacity_ =
      std::max(model_max_batch_size, token_capacity);
}

base::Status Qwen2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }
  if (device_type == base::DeviceType::kDeviceCPU &&
      runtime_data_type_ == base::DataType::kDataTypeBf16) {
    return error::InternalError("BF16 runtime is only supported on non-CPU devices.");
  }
  if (runtime_data_type_ != base::DataType::kDataTypeFp32 &&
      runtime_data_type_ != base::DataType::kDataTypeBf16) {
    return error::InternalError("Unsupported runtime data type for Qwen2Model.");
  }

  device_type_ = device_type;
  if (device_type != DeviceType::kDeviceCPU) {
    std::shared_ptr<base::DeviceContext> context;
    auto init_status = base::initialize_device_context(&context, device_type, 0);
    if (!init_status) {
      return init_status;
    }
    set_device_context(context);
    paged_kv_runtime_ = create_paged_kv_runtime(device_type, device_context_);
    require_compute_queue_or_die(device_context_);
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA &&
      raw_model_data_ != nullptr &&
      raw_model_data_->data_type == base::DataType::kDataTypeBf16 &&
      runtime_data_type_ == base::DataType::kDataTypeFp32) {
    runtime_data_type_ = base::DataType::kDataTypeBf16;
  }
  if (device_type_ == base::DeviceType::kDeviceCPU &&
      raw_model_data_ != nullptr &&
      raw_model_data_->data_type == base::DataType::kDataTypeBf16) {
    return error::InternalError("BF16 weight files are only supported on non-CPU devices.");
  }
  CHECK(device_type_ == base::DeviceType::kDeviceCPU || paged_kv_runtime_ != nullptr)
      << "Paged KV runtime must be initialized for non-CPU devices.";
  auto kv_status = validate_kv_cache_runtime(device_type_);
  if (!kv_status) {
    return kv_status;
  }
  LOG(INFO) << "Qwen2 init weight dtype: " << raw_model_data_->data_type
            << ", runtime dtype: " << runtime_data_type_;
  init_mem();
  kernel::get_sin_cos_cache_kernel(device_type_)(
      config_->head_size_, config_->seq_len_,
      get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache),
      device_type_ == base::DeviceType::kDeviceCUDA ? compute_queue_or_die(device_context_)
                                                    : nullptr);

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status Qwen2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  // Single-sequence convenience path: append slot automatically
  CHECK(kv_cache_manager_ != nullptr)
      << "KV cache manager must be initialized before forward";
  bool ok = kv_cache_manager_->append_slot(single_seq_request_id_);
  CHECK(ok) << "Failed to allocate paged KV block for single-seq forward";
  return forward_with_request(input, pos_tensor, single_seq_request_id_, next);
}

base::Status Qwen2Model::forward_with_request(const tensor::Tensor& input,
                                               const tensor::Tensor& pos_tensor,
                                               base::RequestId request_id,
                                               int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  CHECK(kv_cache_manager_ != nullptr)
      << "KV cache manager must be initialized before forward_with_request";

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);

    tensor::Tensor query = first_row_view(get_buffer(ModelBufferType::kQuery), config_->dim_);
    tensor::Tensor key_temp = first_row_view(get_buffer(ModelBufferType::kPagedKeyTemp),
                                             config_->kv_dim_);
    tensor::Tensor value_temp = first_row_view(get_buffer(ModelBufferType::kPagedValueTemp),
                                               config_->kv_dim_);
    tensor::Tensor rmsnorm_output =
        first_row_view(get_buffer(ModelBufferType::kOutputRMSNorm), config_->dim_);

    const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

    const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key_temp));

    const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
    STATUS_CHECK(value_layer->forward(rmsnorm_output, value_temp));

    STATUS_CHECK(qwen_layers_->rope_layer_->forward(
        query, key_temp, pos_tensor, get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));

    // Scatter KV to the correct request's pool
    auto [block_id, offset] = kv_cache_manager_->current_slot(request_id, layer_idx);
    CHECK_GE(block_id, 0);

    base::BlockAllocator& layer_allocator = kv_cache_manager_->allocator_mut(layer_idx);
    CHECK(paged_kv_runtime_ != nullptr);
    paged_kv_runtime_->scatter_single_token(
        key_temp, value_temp, layer_allocator, block_id, offset, model_block_size,
        config_->kv_head_num_, config_->head_size_);

    // Paged MHA — reuse pre-allocated device buffers
    const auto& block_ids = kv_cache_manager_->get_block_ids(request_id, layer_idx);
    int32_t num_kv_blocks = static_cast<int32_t>(block_ids.size());
    CHECK_GT(num_kv_blocks, 0);

    int32_t context_len = kv_cache_manager_->get_context_len(request_id);

    runtime_copy_or_die(device_context_, block_ids.data(),
                        const_cast<int32_t*>(
                            single_seq_workspace_.block_table_device.ptr<int32_t>()),
                        static_cast<size_t>(num_kv_blocks) * sizeof(int32_t),
                        base::CopyDirection::kHostToDevice,
                        compute_queue_or_die(device_context_), false);
    runtime_copy_or_die(device_context_, &context_len,
                        const_cast<int32_t*>(
                            single_seq_workspace_.seq_lens_device.ptr<int32_t>()),
                        sizeof(int32_t), base::CopyDirection::kHostToDevice,
                        compute_queue_or_die(device_context_), false);

    tensor::Tensor mha_output = first_row_view(get_buffer(ModelBufferType::kOutputMHA),
                                               config_->dim_);
    PagedKVDecodeRuntimeArgs decode_args;
    decode_args.batch_size = 1;
    decode_args.head_num = config_->head_num_;
    decode_args.head_size = config_->head_size_;
    decode_args.kv_mul = config_->kv_mul_;
    decode_args.max_blocks_per_seq = num_kv_blocks;
    decode_args.block_size = model_block_size;
    decode_args.num_kv_heads = config_->kv_head_num_;
    decode_args.queries = &query;
    decode_args.outputs = &mha_output;
    decode_args.block_tables = &single_seq_workspace_.block_table_device;
    decode_args.seq_lens = &single_seq_workspace_.seq_lens_device;
    decode_args.partial_out = &get_buffer(ModelBufferType::kSplitKVPartialOut);
    decode_args.partial_max = &get_buffer(ModelBufferType::kSplitKVPartialMax);
    decode_args.partial_sum = &get_buffer(ModelBufferType::kSplitKVPartialSum);
    CHECK(paged_kv_runtime_ != nullptr);
    paged_kv_runtime_->decode(layer_allocator, decode_args);

    tensor::Tensor attn_output = first_row_view(get_buffer(ModelBufferType::kAttnOutput),
                                                config_->dim_);
    const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));

    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

base::Status Qwen2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

base::Status Qwen2Model::predict_with_request(const tensor::Tensor& input,
                                               const tensor::Tensor& pos_tensor,
                                               base::RequestId request_id,
                                               bool is_prompt, int& next) const {
  auto status = forward_with_request(input, pos_tensor, request_id, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void Qwen2Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void Qwen2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(qwen_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  qwen_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                             cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void Qwen2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(qwen_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  auto model_weight_dtype = raw_model_data_->data_type;
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                             weight_embedding, cpu_device_type, model_weight_dtype);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    pos += dim * dim;
    wq->set_bias(0, dim, this->raw_model_data_->weight(pos), cpu_device_type, model_weight_dtype);
    pos += dim;
    qwen_layers_->wq_layers_.push_back(wq);
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    pos += config_->kv_dim_ * dim;
    wk->set_bias(0, config_->kv_dim_, this->raw_model_data_->weight(pos), cpu_device_type,
                 model_weight_dtype);
    pos += config_->kv_dim_;
    qwen_layers_->wk_layers_.push_back(wk);
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    pos += config_->kv_dim_ * dim;
    wv->set_bias(0, config_->kv_dim_, this->raw_model_data_->weight(pos), cpu_device_type,
                 model_weight_dtype);
    pos += config_->kv_dim_;
    qwen_layers_->wv_layers_.push_back(wv);
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    qwen_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    qwen_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    qwen_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   model_weight_dtype);
    qwen_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  qwen_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    qwen_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                         this->raw_model_data_->weight(0), cpu_device_type,
                                         model_weight_dtype);
  } else {
    qwen_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                         this->raw_model_data_->weight(pos), cpu_device_type,
                                         model_weight_dtype);
  }

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type,
                               model_weight_dtype);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->dim_ + config_->dim_);
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
  rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type,
                               model_weight_dtype);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type,
                              model_weight_dtype);
  qwen_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

void Qwen2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc =
      device_allocator_for(device_type_, device_context_);

  if (device_type_ != base::DeviceType::kDeviceCPU) {
    require_compute_queue_or_die(device_context_);
    qwen_layers_->materialize(device_context_, runtime_data_type_);
  }

  const base::DataType act_dtype =
      device_type_ == base::DeviceType::kDeviceCPU ? base::DataType::kDataTypeFp32
                                                   : runtime_data_type_;

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      host_allocator_for(device_context_);
  std::shared_ptr<base::DeviceAllocator> alloc_device = alloc;

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));

  const base::KVCacheStorageSpec kv_storage_spec = kv_cache_storage_spec();
  const size_t workspace_bytes_per_token =
      serving_workspace_bytes_per_token(act_dtype);
  WorkspaceTokenSizingConfig workspace_sizing_config;
  workspace_sizing_config.workspace_bytes_per_token = workspace_bytes_per_token;
  workspace_sizing_config.kv_bytes_per_block_per_layer =
      kv_cache_bytes_per_block_per_layer(
          kv_storage_spec, model_block_size, config_->kv_head_num_,
          config_->head_size_);
  workspace_sizing_config.layer_num = config_->layer_num_;
  workspace_sizing_config.requested_token_capacity =
      serving_workspace_token_capacity_;
  workspace_sizing_config.min_token_capacity = model_max_batch_size;
  workspace_sizing_config.min_dynamic_kv_blocks = kMinDynamicKVBlocks;
  base::DeviceMemoryInfo memory_info_before_workspace{};
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
    auto memory_status = device_context_->runtime->query_memory(&memory_info_before_workspace);
    CHECK(memory_status) << memory_status.get_err_msg();
  }
  const int32_t workspace_tokens =
      device_type_ != base::DeviceType::kDeviceCPU
          ? resolve_serving_workspace_token_capacity(workspace_sizing_config,
                                                     memory_info_before_workspace)
          : std::max(model_max_batch_size, serving_workspace_token_capacity_);
  serving_workspace_token_capacity_ = workspace_tokens;
  serving_workspace_reserved_bytes_ =
      serving_workspace_bytes_for_tokens(workspace_bytes_per_token, workspace_tokens);
  LOG(INFO) << "Serving workspace preallocation: token_capacity="
            << workspace_tokens
            << ", reserved_bytes=" << serving_workspace_reserved_bytes_;

  // Reusable activation buffers are allocated to the serving token capacity
  // before sizing the KV pool. This makes the backend memory query treat non-KV
  // runtime workspace as already reserved, so an aggressive KV pool cannot
  // starve later batch reshapes.
  tensor::Tensor input_embeddings(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));

  tensor::Tensor mha_output(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, mha_output));

  tensor::Tensor w2_output(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW2Output, w2_output));

  tensor::Tensor ffn_norm_output(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, ffn_norm_output));

  tensor::Tensor w1_output(act_dtype, workspace_tokens * config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(act_dtype, workspace_tokens * config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // Pre-allocated device buffers for single-seq forward.
  int32_t max_blocks_per_seq = (config_->seq_len_ + model_block_size - 1) / model_block_size;
  single_seq_workspace_.block_table_device =
      tensor::Tensor(base::DataType::kDataTypeInt32, max_blocks_per_seq, true, alloc_device);
  single_seq_workspace_.block_table_device.set_device_type(device_type_);
  single_seq_workspace_.seq_lens_device =
      tensor::Tensor(base::DataType::kDataTypeInt32, 1, true, alloc_device);
  single_seq_workspace_.seq_lens_device.set_device_type(device_type_);

  // Paged key/value temp buffers for batch scatter
  tensor::Tensor paged_key_temp(act_dtype, workspace_tokens * config_->kv_dim_, true, alloc);
  tensor::Tensor paged_value_temp(act_dtype, workspace_tokens * config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kPagedKeyTemp, paged_key_temp));
  CHECK(insert_buffer(ModelBufferType::kPagedValueTemp, paged_value_temp));

  // Split-KV workspace (always fp32)
  tensor::Tensor splitkv_partial_out(base::DataType::kDataTypeFp32,
      workspace_tokens * config_->head_num_ * kServingMaxSplitKVPartitions * config_->head_size_,
      true, alloc_device);
  tensor::Tensor splitkv_partial_max(base::DataType::kDataTypeFp32,
      workspace_tokens * config_->head_num_ * kServingMaxSplitKVPartitions,
      true, alloc_device);
  tensor::Tensor splitkv_partial_sum(base::DataType::kDataTypeFp32,
      workspace_tokens * config_->head_num_ * kServingMaxSplitKVPartitions,
      true, alloc_device);
  splitkv_partial_out.set_device_type(device_type_);
  splitkv_partial_max.set_device_type(device_type_);
  splitkv_partial_sum.set_device_type(device_type_);
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialOut, splitkv_partial_out));
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialMax, splitkv_partial_max));
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialSum, splitkv_partial_sum));

  // Query output
  tensor::Tensor query(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Score storage (kept for legacy non-paged mha Layer, unused in batch path)
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));

  // Attention output
  tensor::Tensor attn_output(act_dtype, workspace_tokens * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // Forward output
  tensor::Tensor forward_output(act_dtype, workspace_tokens * config_->vocab_size_, true, alloc);
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));

  if (device_type_ != base::DeviceType::kDeviceCPU) {
    // These small metadata buffers are normally grown lazily on the first
    // mixed-batch step. Preallocating them before KV sizing prevents high KV
    // cache memory utilization settings from leaving no room for first-use
    // scheduler/model metadata.
    const int32_t metadata_request_capacity = model_max_batch_size;
    const int32_t metadata_block_table_capacity =
        metadata_request_capacity * max_blocks_per_seq;
    ensure_row_attention_metadata_workspace(workspace_tokens,
                                            metadata_request_capacity,
                                            metadata_block_table_capacity);
    ensure_batch_sampler_index_capacity(metadata_request_capacity);
    ensure_batch_sampler_workspace(metadata_request_capacity);
    ensure_batch_sampler_sync_event();
  }

  // KV cache: single unified KVCacheManager for all paths (single-seq + batch).
  // Size it after non-KV runtime workspace is allocated, then use the requested
  // fraction of the remaining free device memory with a safety reserve and retry
  // fallback for oversized requests.
  int32_t total_blocks = model_num_blocks * model_max_batch_size;
  if (device_type_ != base::DeviceType::kDeviceCPU) {
    DynamicKVCacheSizingConfig kv_sizing_config;
    kv_sizing_config.layer_num = config_->layer_num_;
    kv_sizing_config.block_size = model_block_size;
    kv_sizing_config.num_kv_heads = config_->kv_head_num_;
    kv_sizing_config.head_size = config_->head_size_;
    kv_sizing_config.min_dynamic_kv_blocks = kMinDynamicKVBlocks;
    kv_sizing_config.kv_cache_memory_utilization = kv_cache_memory_utilization_;
    base::DeviceMemoryInfo memory_info_before_kv{};
    CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
    auto memory_status = device_context_->runtime->query_memory(&memory_info_before_kv);
    CHECK(memory_status) << memory_status.get_err_msg();
    total_blocks = estimate_dynamic_kv_blocks_per_layer(
        kv_storage_spec, kv_sizing_config, memory_info_before_kv);
  }

  while (true) {
    std::vector<std::unique_ptr<base::BlockAllocator>> layer_allocators;
    layer_allocators.reserve(config_->layer_num_);
    bool allocation_ok = true;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
      auto allocator = std::make_unique<base::BlockAllocator>(
          total_blocks, model_block_size, config_->kv_head_num_, config_->head_size_,
          kv_storage_spec, device_type_);
      if (allocator->key_pool().is_empty() || allocator->value_pool().is_empty() ||
          (kv_storage_spec.has_scales() &&
           (allocator->key_scale_pool().is_empty() ||
            allocator->value_scale_pool().is_empty()))) {
        allocation_ok = false;
        break;
      }
      layer_allocators.emplace_back(std::move(allocator));
    }

    if (allocation_ok) {
      kv_cache_manager_ = std::make_unique<base::KVCacheManager>(
          model_block_size, config_->layer_num_, std::move(layer_allocators));
      if (radix_cache_enabled_override_set_) {
        kv_cache_manager_->set_radix_cache_enabled(radix_cache_enabled_override_);
      }
      break;
    }

    const int32_t next_blocks = std::max(kMinDynamicKVBlocks, total_blocks * 90 / 100);
    CHECK_LT(next_blocks, total_blocks)
        << "Unable to allocate minimum KV cache pool. kv_cache_memory_utilization="
        << kv_cache_memory_utilization_ << ", blocks_per_layer=" << total_blocks;
    LOG(WARNING) << "KV cache allocation failed for blocks_per_layer=" << total_blocks
                 << ". Retrying with blocks_per_layer=" << next_blocks;
    total_blocks = next_blocks;
  }

  // Register a persistent request for the single-sequence forward() path.
  single_seq_request_id_ = kv_cache_manager_->register_request();
}

serving::ServingCapacityInfo Qwen2Model::serving_capacity_info() const {
  serving::ServingCapacityInfo info;
  CHECK(config_ != nullptr);
  CHECK(kv_cache_manager_ != nullptr);

  info.max_batch_size = model_max_batch_size;
  info.block_size = kv_cache_manager_->block_size();
  info.total_kv_blocks = kv_cache_manager_->num_total_blocks(0);
  info.free_kv_blocks = kv_cache_manager_->num_free_blocks(0);
  info.layer_num = config_->layer_num_;
  info.model_dim = config_->dim_;
  info.head_num = config_->head_num_;
  info.kv_head_num = config_->kv_head_num_;
  info.head_size = config_->head_size_;
  info.kv_dim = config_->kv_dim_;
  info.hidden_dim = config_->hidden_dim_;
  info.vocab_size = config_->vocab_size_;
  info.runtime_data_type = runtime_data_type_;

  if (device_type_ != base::DeviceType::kDeviceCPU) {
    base::DeviceMemoryInfo memory_info{};
    CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
    auto memory_status = device_context_->runtime->query_memory(&memory_info);
    CHECK(memory_status) << memory_status.get_err_msg();
    info.device_free_memory_bytes = memory_info.free_bytes;
    info.device_total_memory_bytes = memory_info.total_bytes;
  }

  const base::KVCacheStorageSpec kv_storage_spec = kv_cache_storage_spec();
  info.kv_bytes_per_token = base::KVCacheBytesPerToken(
      kv_storage_spec, info.layer_num, info.kv_head_num, info.head_size);
  info.kv_bytes_per_block_per_layer =
      kv_cache_bytes_per_block_per_layer(kv_storage_spec, info.block_size,
                                         info.kv_head_num, info.head_size);
  info.total_kv_pool_bytes = static_cast<size_t>(info.total_kv_blocks) *
                             static_cast<size_t>(info.layer_num) *
                             info.kv_bytes_per_block_per_layer;
  const base::DataType act_dtype =
      device_type_ == base::DeviceType::kDeviceCPU ? base::DataType::kDataTypeFp32
                                                   : runtime_data_type_;
  info.workspace_bytes_per_token = serving_workspace_bytes_per_token(act_dtype);
  info.serving_workspace_token_capacity = serving_workspace_token_capacity_;
  info.serving_workspace_reserved_bytes = serving_workspace_reserved_bytes_;
  return info;
}

base::Status Qwen2Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->wq_layers_.at(i) || !qwen_layers_->wk_layers_.at(i) ||
        !qwen_layers_->wv_layers_.at(i) || !qwen_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void Qwen2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output =
      first_row_view(get_buffer(ModelBufferType::kOutputRMSNorm), config_->dim_);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  tensor::Tensor attn_output = first_row_view(get_buffer(ModelBufferType::kAttnOutput),
                                              config_->dim_);
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, attn_output, input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output =
      first_row_view(get_buffer(ModelBufferType::kFFNRMSNorm), config_->dim_);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = first_row_view(get_buffer(ModelBufferType::kW1Output),
                                            config_->hidden_dim_);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = first_row_view(get_buffer(ModelBufferType::kW3Output),
                                           config_->hidden_dim_);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = first_row_view(get_buffer(ModelBufferType::kW2Output),
                                            config_->dim_);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

op::EmbeddingOutput Qwen2Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  const int32_t token_num = static_cast<int32_t>(tokens.size());
  const bool token_shape_mismatch =
      input_tokens.dims_size() != 1 || input_tokens.get_dim(0) != token_num;
  const bool embedding_shape_mismatch =
      input_embeddings.dims_size() != 2 || input_embeddings.get_dim(0) != token_num ||
      input_embeddings.get_dim(1) != config_->dim_;
  if (token_shape_mismatch) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
  }
  if (embedding_shape_mismatch) {
    input_embeddings.reshape_no_realloc({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}


void Qwen2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = first_row_view(get_buffer(ModelBufferType::kForwardOutput),
                                                 config_->vocab_size_);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = first_row_view(get_buffer(ModelBufferType::kForwardOutput),
                                                 config_->vocab_size_);

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_output.get_buffer()->ptr(),
                                                 forward_output.size(), forward_output.data_type(),
                                                 compute_queue_or_null(device_context_)));
  }
  return next;
}

void Qwen2Model::ensure_batch_sampler_index_capacity(int32_t index_count) const {
  if (index_count <= 0) {
    return;
  }
  auto host_alloc = pinned_host_allocator_for(device_context_);
  auto device_alloc =
      device_allocator_for(device_type_, device_context_);
  ensure_tensor_storage(batch_sampler_workspace_.row_indices.host,
                        base::DataType::kDataTypeInt32,
                        index_count, host_alloc, base::DeviceType::kDeviceCPU);
  ensure_tensor_storage(batch_sampler_workspace_.row_indices.device,
                        base::DataType::kDataTypeInt32,
                        index_count, device_alloc, device_type_);
}

void Qwen2Model::ensure_batch_sampler_workspace(int32_t sample_count) const {
  if (sample_count <= 0) {
    return;
  }
  auto host_alloc = pinned_host_allocator_for(device_context_);
  auto device_alloc =
      device_allocator_for(device_type_, device_context_);
  ensure_tensor_storage(batch_sampler_workspace_.token_ids.device,
                        base::DataType::kDataTypeInt32,
                        sample_count, device_alloc, device_type_);
  ensure_tensor_storage(batch_sampler_workspace_.token_ids.host,
                        base::DataType::kDataTypeInt32,
                        sample_count, host_alloc, base::DeviceType::kDeviceCPU);
}

void Qwen2Model::ensure_batch_sampler_sync_event() const {
  if (batch_sampler_workspace_.copy_done_event != nullptr ||
      device_type_ == base::DeviceType::kDeviceCPU) {
    return;
  }
  CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
  auto status = device_context_->runtime->create_event(
      &batch_sampler_workspace_.copy_done_event, true);
  runtime_status_or_die(status, "Failed to create sampler sync event");
}

void Qwen2Model::ensure_row_attention_metadata_workspace(
    int32_t token_capacity,
    int32_t request_capacity,
    int32_t block_table_capacity) const {
  if (token_capacity <= 0) {
    return;
  }

  auto host_alloc = pinned_host_allocator_for(device_context_);
  auto device_alloc =
      device_allocator_for(device_type_, device_context_);
  const auto ensure_pair = [&](Qwen2HostDeviceInt32TensorPair& pair,
                               int32_t capacity) {
    ensure_tensor_storage(pair.host, base::DataType::kDataTypeInt32,
                          capacity, host_alloc, base::DeviceType::kDeviceCPU);
    ensure_tensor_storage(pair.device, base::DataType::kDataTypeInt32,
                          capacity, device_alloc, device_type_);
  };

  ensure_pair(row_attn_workspace_.slot_mapping, token_capacity);
  ensure_pair(row_attn_workspace_.seq_lens, token_capacity);
  ensure_pair(row_attn_workspace_.prefill_base_context_lens, token_capacity);
  ensure_pair(row_attn_workspace_.prefill_chunk_row_starts, token_capacity);
  ensure_pair(row_attn_workspace_.prefill_local_token_offsets, token_capacity);
  ensure_pair(row_attn_workspace_.prefill_request_indices, token_capacity);
  ensure_pair(row_attn_workspace_.block_tables,
              std::max(request_capacity, block_table_capacity));
}

serving::SampledTokenView Qwen2Model::batch_sample_device(const tensor::Tensor& logits,
                                                          const tensor::Tensor& row_indices,
                                                          int32_t sample_count) const {
  base::nvtx::ScopedRange range("batch_sample_device", base::nvtx::kColorSample);
  if (sample_count <= 0) {
    return {};
  }
  CHECK_NE(logits.device_type(), base::DeviceType::kDeviceCPU);
  CHECK_EQ(logits.device_type(), row_indices.device_type());
  CHECK_EQ(logits.dims_size(), 2);
  CHECK_EQ(row_indices.dims_size(), 1);
  CHECK_EQ(row_indices.data_type(), base::DataType::kDataTypeInt32);

  check_tensor_storage_capacity(batch_sampler_workspace_.token_ids.device,
                                base::DataType::kDataTypeInt32, sample_count,
                                "batch_sampler.token_ids.device");
  check_tensor_storage_capacity(batch_sampler_workspace_.token_ids.host,
                                base::DataType::kDataTypeInt32, sample_count,
                                "batch_sampler.token_ids.host");
  ensure_batch_sampler_sync_event();

  tensor::Tensor token_ids_device =
      reshape_view(batch_sampler_workspace_.token_ids.device, {sample_count});
  tensor::Tensor token_ids_host =
      reshape_view(batch_sampler_workspace_.token_ids.host, {sample_count});

  {
    base::nvtx::ScopedRange kernel_range("argmax_selected_rows", base::nvtx::kColorSample);
    kernel::get_argmax_selected_rows_kernel(logits.device_type())(
        logits, row_indices, token_ids_device, compute_queue_or_null(device_context_));
  }

  {
    base::nvtx::ScopedRange memcpy_range("sample_d2h", base::nvtx::kColorMemcpy);
    runtime_copy_or_die(device_context_, token_ids_device.ptr<int32_t>(),
                        token_ids_host.ptr<int32_t>(),
                        static_cast<size_t>(sample_count) * sizeof(int32_t),
                        base::CopyDirection::kDeviceToHost,
                        compute_queue_or_null(device_context_), false);
  }

  void* queue = compute_queue_or_null(device_context_);
  if (queue != nullptr && batch_sampler_workspace_.copy_done_event != nullptr) {
    base::nvtx::ScopedRange wait_range("sample_stream_wait", base::nvtx::kColorMemcpy);
    CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
    auto status = device_context_->runtime->record_event(
        batch_sampler_workspace_.copy_done_event, queue);
    runtime_status_or_die(status, "Failed to record sampler copy event");
    status = device_context_->runtime->wait_event(batch_sampler_workspace_.copy_done_event);
    runtime_status_or_die(status, "Failed to synchronize sampler copy event");
  } else if (queue != nullptr) {
    CHECK(device_context_ != nullptr && device_context_->runtime != nullptr);
    auto status = device_context_->runtime->synchronize_queue(queue);
    runtime_status_or_die(status, "Failed to synchronize sampler stream");
  }

  return {token_ids_host.ptr<int32_t>(), sample_count};
}

serving::SampledTokenView Qwen2Model::batch_sample_cpu(const tensor::Tensor& logits,
                                                       int32_t num_rows,
                                                       const std::vector<int32_t>& row_indices) const {
  if (row_indices.empty()) {
    return {};
  }

  const int32_t vocab_size = config_->vocab_size_;
  const int32_t sample_count = static_cast<int32_t>(row_indices.size());
  check_tensor_storage_capacity(batch_sampler_workspace_.token_ids.host,
                                base::DataType::kDataTypeInt32, sample_count,
                                "batch_sampler.token_ids.host");
  int32_t* host_tokens = batch_sampler_workspace_.token_ids.host.ptr<int32_t>();
  if (logits.device_type() == base::DeviceType::kDeviceCPU) {
    if (logits.data_type() == base::DataType::kDataTypeFp32) {
      argmax_host_rows_into(logits.ptr<float>(), num_rows, vocab_size, row_indices, host_tokens);
      return {host_tokens, sample_count};
    }
    CHECK_EQ(logits.data_type(), base::DataType::kDataTypeBf16);
    argmax_host_rows_into(logits.ptr<uint16_t>(), num_rows, vocab_size, row_indices, host_tokens);
    return {host_tokens, sample_count};
  }

  const int64_t logits_elements = static_cast<int64_t>(num_rows) * vocab_size;
  if (logits.data_type() == base::DataType::kDataTypeFp32) {
    std::vector<float> logits_cpu(logits_elements);
    runtime_copy_or_die(device_context_, logits.ptr<float>(), logits_cpu.data(),
                        static_cast<size_t>(logits_elements) * sizeof(float),
                        base::CopyDirection::kDeviceToHost, nullptr, true);
    argmax_host_rows_into(logits_cpu.data(), num_rows, vocab_size, row_indices, host_tokens);
    return {host_tokens, sample_count};
  }

  CHECK_EQ(logits.data_type(), base::DataType::kDataTypeBf16);
  std::vector<uint16_t> logits_cpu(logits_elements);
  runtime_copy_or_die(device_context_, logits.ptr<uint16_t>(), logits_cpu.data(),
                      static_cast<size_t>(logits_elements) * sizeof(uint16_t),
                      base::CopyDirection::kDeviceToHost, nullptr, true);
  argmax_host_rows_into(logits_cpu.data(), num_rows, vocab_size, row_indices, host_tokens);
  return {host_tokens, sample_count};
}

serving::SampledTokenView Qwen2Model::batch_sample(int32_t batch_size) const {
  base::nvtx::ScopedRange range("batch_sample_decode_only", base::nvtx::kColorSample);
  if (batch_size <= 0) {
    return {};
  }

  tensor::Tensor forward_output =
      reshape_view(get_buffer(ModelBufferType::kForwardOutput),
                   {batch_size, config_->vocab_size_});
  std::vector<int32_t> row_indices(batch_size);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  if (device_type_ == base::DeviceType::kDeviceCPU || batch_sample_cpu_fallback_enabled()) {
    return batch_sample_cpu(forward_output, batch_size, row_indices);
  }

  check_tensor_storage_capacity(batch_sampler_workspace_.row_indices.host,
                                base::DataType::kDataTypeInt32, batch_size,
                                "batch_sampler.row_indices.host");
  check_tensor_storage_capacity(batch_sampler_workspace_.row_indices.device,
                                base::DataType::kDataTypeInt32, batch_size,
                                "batch_sampler.row_indices.device");
  int32_t* host_indices = batch_sampler_workspace_.row_indices.host.ptr<int32_t>();
  std::iota(host_indices, host_indices + batch_size, 0);

  {
    base::nvtx::ScopedRange memcpy_range("sample_row_indices_h2d", base::nvtx::kColorMemcpy);
    runtime_copy_or_die(device_context_, host_indices,
                        batch_sampler_workspace_.row_indices.device.ptr<int32_t>(),
                        static_cast<size_t>(batch_size) * sizeof(int32_t),
                        base::CopyDirection::kHostToDevice,
                        compute_queue_or_null(device_context_), true);
  }

  tensor::Tensor row_indices_device =
      reshape_view(batch_sampler_workspace_.row_indices.device, {batch_size});
  return batch_sample_device(forward_output, row_indices_device, batch_size);
}

serving::SampledTokenView Qwen2Model::batch_sample(const serving::MixedBatchMetadata& batch) const {
  base::nvtx::ScopedRange range("batch_sample_mixed", base::nvtx::kColorSample);
  if (batch.logits_row_indices.empty()) {
    return {};
  }

  tensor::Tensor forward_output =
      reshape_view(get_buffer(ModelBufferType::kForwardOutput),
                   {batch.num_tokens, config_->vocab_size_});

  if (device_type_ == base::DeviceType::kDeviceCPU || batch_sample_cpu_fallback_enabled()) {
    return batch_sample_cpu(forward_output, batch.num_tokens, batch.logits_row_indices);
  }

  CHECK(!batch.logits_indices.is_empty());
  return batch_sample_device(forward_output, batch.logits_indices,
                             static_cast<int32_t>(batch.logits_row_indices.size()));
}

int32_t Qwen2Model::prefill_chunk(base::RequestId request_id,
                                   const std::vector<int32_t>& prompt_tokens,
                                   int32_t start_pos, int32_t chunk_size) {
  CHECK(kv_cache_manager_ != nullptr);
  CHECK_GT(chunk_size, 0);
  CHECK_LE(start_pos + chunk_size, static_cast<int32_t>(prompt_tokens.size()));

  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  int next = -1;

  for (int32_t t = 0; t < chunk_size; ++t) {
    int32_t pos = start_pos + t;
    bool is_last_in_prompt = (pos == static_cast<int32_t>(prompt_tokens.size()) - 1);

    pos_tensor.index<int32_t>(0) = pos;

    // Allocate KV slot for this token
    bool ok = kv_cache_manager_->append_slot(request_id);
    CHECK(ok) << "Failed to allocate KV slot during prefill at pos " << pos;

    // Embed only the current token so we can reuse the existing single-token buffers.
    const auto emb_output = embedding(std::vector<int>{prompt_tokens[pos]});
    tensor::Tensor input = fill_input(pos_tensor, emb_output, false);

    // Forward + sample on last token only
    if (is_last_in_prompt) {
      auto status = predict_with_request(input, pos_tensor, request_id, false, next);
      CHECK(status) << "prefill_chunk predict failed: " << status.get_err_msg();
    } else {
      auto status = forward_with_request(input, pos_tensor, request_id, next);
      CHECK(status) << "prefill_chunk forward failed: " << status.get_err_msg();
    }
  }
  return next;
}

}  // namespace model
