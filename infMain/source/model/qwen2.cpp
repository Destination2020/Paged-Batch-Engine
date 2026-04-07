// Updated on March 31, 2026
#include "model/qwen2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include <vector>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"
namespace model {

namespace {
void prepare_cuda_layer(const std::shared_ptr<op::Layer>& layer,
                        const std::shared_ptr<kernel::CudaConfig>& config,
                        base::DataType runtime_data_type) {
  if (!layer) {
    return;
  }
  layer->set_cuda_config(config);
  layer->set_data_type(runtime_data_type);
  layer->to_cuda();
}
}  // namespace

void Qwen2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config,
                          base::DataType runtime_data_type) {
  prepare_cuda_layer(add_layer_, config, runtime_data_type);
  prepare_cuda_layer(rope_layer_, config, runtime_data_type);
  prepare_cuda_layer(swiglu_layer_, config, runtime_data_type);
  prepare_cuda_layer(cls_layer_, config, runtime_data_type);
  prepare_cuda_layer(embedding_layer_, config, runtime_data_type);
  prepare_cuda_layer(mha_layer_, config, runtime_data_type);

  for (auto& weight_layer : wq_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : wk_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : wv_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : wo_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : w1_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : w2_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& weight_layer : w3_layers_) {
    prepare_cuda_layer(weight_layer, config, runtime_data_type);
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    prepare_cuda_layer(rms_norm_layer, config, runtime_data_type);
  }
}

Qwen2Model::Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

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
    return error::InternalError("BF16 runtime is only supported on CUDA.");
  }
  if (runtime_data_type_ != base::DataType::kDataTypeFp32 &&
      runtime_data_type_ != base::DataType::kDataTypeBf16) {
    return error::InternalError("Unsupported runtime data type for Qwen2Model.");
  }
  if (use_fp8_kv_cache_) {
    if (device_type != base::DeviceType::kDeviceCUDA) {
      return error::InternalError("FP8 KV cache is only supported on CUDA.");
    }
    if (runtime_data_type_ != base::DataType::kDataTypeBf16) {
      return error::InternalError("FP8 KV cache v1 requires BF16 runtime.");
    }
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cublasCreate(&cuda_config_->cublas_handle);
    cublasSetStream(cuda_config_->cublas_handle, cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
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
    return error::InternalError("BF16 weight files are only supported on CUDA.");
  }
  LOG(INFO) << "Qwen2 init weight dtype: " << raw_model_data_->data_type
            << ", runtime dtype: " << runtime_data_type_;
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

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

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);

    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
    tensor::Tensor key_temp = this->get_buffer(ModelBufferType::kPagedKeyTemp);
    tensor::Tensor value_temp = this->get_buffer(ModelBufferType::kPagedValueTemp);
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

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

    kernel::scatter_kv_to_page_cu(
        key_temp, value_temp,
        const_cast<tensor::Tensor&>(kv_cache_manager_->allocator(layer_idx).key_pool()),
        const_cast<tensor::Tensor&>(kv_cache_manager_->allocator(layer_idx).value_pool()),
        block_id, offset, model_block_size,
        config_->kv_head_num_, config_->head_size_,
        device_type_, cuda_config_.get());

    // Paged MHA
    const auto& block_ids = kv_cache_manager_->get_block_ids(request_id, layer_idx);
    int32_t num_kv_blocks = static_cast<int32_t>(block_ids.size());
    CHECK_GT(num_kv_blocks, 0);

    int32_t context_len = kv_cache_manager_->get_context_len(request_id);
    tensor::Tensor block_table_gpu(base::DataType::kDataTypeInt32, num_kv_blocks, true, alloc_cu);
    tensor::Tensor seq_lens_gpu(base::DataType::kDataTypeInt32, 1, true, alloc_cu);
    block_table_gpu.set_device_type(base::DeviceType::kDeviceCUDA);
    seq_lens_gpu.set_device_type(base::DeviceType::kDeviceCUDA);

    alloc_cu->memcpy(block_ids.data(), const_cast<int32_t*>(block_table_gpu.ptr<int32_t>()),
                     num_kv_blocks * sizeof(int32_t),
                     base::MemcpyKind::kMemcpyCPU2CUDA, cuda_config_->stream, true);
    alloc_cu->memcpy(&context_len, const_cast<int32_t*>(seq_lens_gpu.ptr<int32_t>()),
                     sizeof(int32_t),
                     base::MemcpyKind::kMemcpyCPU2CUDA, cuda_config_->stream, true);

    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    kernel::splitkv_batched_paged_mha_decode_cu(
        1, config_->head_num_, config_->head_size_, config_->kv_mul_,
        query, mha_output,
        kv_cache_manager_->allocator(layer_idx).key_pool(),
        kv_cache_manager_->allocator(layer_idx).value_pool(),
        block_table_gpu, seq_lens_gpu,
        num_kv_blocks, model_block_size, config_->kv_head_num_,
        get_buffer(ModelBufferType::kSplitKVPartialOut),
        get_buffer(ModelBufferType::kSplitKVPartialMax),
        get_buffer(ModelBufferType::kSplitKVPartialSum),
        device_type_, cuda_config_.get());

    tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
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
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    qwen_layers_->to_cuda(cuda_config_, runtime_data_type_);
  }

  const base::DataType act_dtype =
      device_type_ == base::DeviceType::kDeviceCUDA ? runtime_data_type_
                                                    : base::DataType::kDataTypeFp32;

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));

  // All intermediate buffers are 1D.
  // forward_decode_batch() uses raw pointers to treat them as [batch, dim].
  // Embedding is 2D [1, dim] for the Layer check() in single-token path.
  tensor::Tensor input_embeddings(act_dtype, 1, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));

  tensor::Tensor mha_output(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, mha_output));

  tensor::Tensor w2_output(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW2Output, w2_output));

  tensor::Tensor ffn_norm_output(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, ffn_norm_output));

  tensor::Tensor w1_output(act_dtype, model_max_batch_size * config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(act_dtype, model_max_batch_size * config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // KV cache: single unified KVCacheManager for all paths (single-seq + batch)
  const int32_t total_blocks = model_num_blocks * model_max_batch_size;
  std::vector<std::unique_ptr<base::BlockAllocator>> layer_allocators;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    layer_allocators.emplace_back(std::make_unique<base::BlockAllocator>(
        total_blocks, model_block_size,
        config_->kv_head_num_, config_->head_size_,
        act_dtype, base::DeviceType::kDeviceCUDA));
  }
  kv_cache_manager_ = std::make_unique<base::KVCacheManager>(
      model_block_size, config_->layer_num_, std::move(layer_allocators));

  // Register a persistent request for the single-sequence forward() path
  single_seq_request_id_ = kv_cache_manager_->register_request();

  // Paged key/value temp buffers for batch scatter
  tensor::Tensor paged_key_temp(act_dtype, model_max_batch_size * config_->kv_dim_, true, alloc);
  tensor::Tensor paged_value_temp(act_dtype, model_max_batch_size * config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kPagedKeyTemp, paged_key_temp));
  CHECK(insert_buffer(ModelBufferType::kPagedValueTemp, paged_value_temp));

  // Split-KV workspace (always fp32)
  constexpr int32_t MAX_PARTITIONS = 32;
  tensor::Tensor splitkv_partial_out(base::DataType::kDataTypeFp32,
      model_max_batch_size * config_->head_num_ * MAX_PARTITIONS * config_->head_size_,
      true, alloc_cu);
  tensor::Tensor splitkv_partial_max(base::DataType::kDataTypeFp32,
      model_max_batch_size * config_->head_num_ * MAX_PARTITIONS,
      true, alloc_cu);
  tensor::Tensor splitkv_partial_sum(base::DataType::kDataTypeFp32,
      model_max_batch_size * config_->head_num_ * MAX_PARTITIONS,
      true, alloc_cu);
  splitkv_partial_out.set_device_type(base::DeviceType::kDeviceCUDA);
  splitkv_partial_max.set_device_type(base::DeviceType::kDeviceCUDA);
  splitkv_partial_sum.set_device_type(base::DeviceType::kDeviceCUDA);
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialOut, splitkv_partial_out));
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialMax, splitkv_partial_max));
  CHECK(insert_buffer(ModelBufferType::kSplitKVPartialSum, splitkv_partial_sum));

  // Query output
  tensor::Tensor query(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Score storage (kept for legacy non-paged mha Layer, unused in batch path)
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));

  // Attention output
  tensor::Tensor attn_output(act_dtype, model_max_batch_size * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  // Forward output
  tensor::Tensor forward_output(act_dtype, model_max_batch_size * config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
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
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
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
  STATUS_CHECK(
      qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
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
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
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

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Qwen2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_output.get_buffer()->ptr(),
                                                 forward_output.size(), forward_output.data_type(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

std::vector<int32_t> Qwen2Model::batch_sample(int32_t batch_size) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const int32_t vocab_size = config_->vocab_size_;

  // Copy logits to CPU and do argmax per sequence
  std::vector<float> logits_cpu(batch_size * vocab_size);
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  if (forward_output.data_type() == base::DataType::kDataTypeFp32) {
    alloc_cu->memcpy(forward_output.ptr<float>(), logits_cpu.data(),
                     batch_size * vocab_size * sizeof(float),
                     base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
  } else {
    // BF16: convert on GPU first would be better, but for simplicity copy raw and convert
    std::vector<uint16_t> raw(batch_size * vocab_size);
    alloc_cu->memcpy(forward_output.ptr<uint16_t>(), raw.data(),
                     batch_size * vocab_size * sizeof(uint16_t),
                     base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
    for (int i = 0; i < batch_size * vocab_size; ++i) {
      // BF16 to float: shift left 16 bits
      uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
      float val;
      memcpy(&val, &bits, sizeof(float));
      logits_cpu[i] = val;
    }
  }

  std::vector<int32_t> results(batch_size);
  for (int32_t b = 0; b < batch_size; ++b) {
    const float* row = logits_cpu.data() + b * vocab_size;
    int32_t best = 0;
    float best_val = row[0];
    for (int32_t v = 1; v < vocab_size; ++v) {
      if (row[v] > best_val) {
        best_val = row[v];
        best = v;
      }
    }
    results[b] = best;
  }
  return results;
}

}  // namespace model
