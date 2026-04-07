// Batched decode forward implementation for Qwen2Model
#include "model/qwen2.h"
#include <glog/logging.h>
#include <op/layer.h>
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include "op/kernels/cuda/matmul_kernel_batch.cuh"
#include "op/kernels/cuda/add_bias_kernel.cuh"
#include "op/kernels/cuda/paged_mha_fast_kernel.cuh"
#include "op/kernels/cuda/paged_mha_kernel.cuh"
#include "op/kernels/cuda/scatter_kv_kernel.cuh"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/rmsnorm_kernel.cuh"
#include "../op/kernels/cuda/emb_kernel.cuh"
#include "../op/kernels/cuda/add_kernel.cuh"
#include "../op/kernels/cuda/swiglu_kernel.cuh"

namespace model {

namespace {
const tensor::Tensor& get_layer_weight0(const std::shared_ptr<op::Layer>& layer) {
  auto param_layer = std::dynamic_pointer_cast<op::LayerParam>(layer);
  CHECK(param_layer != nullptr) << "Layer is not op::LayerParam, cannot access weight";
  return param_layer->get_weight(0);
}
}  // namespace

base::Status Qwen2Model::forward_decode_batch(const serving::DecodeBatchMetadata& batch) const {
  if (batch.batch_size <= 0) {
    return base::error::InvalidArgument("Empty batch");
  }
  CHECK_NE(cuda_config_, nullptr);
  CHECK_NE(cuda_config_->cublas_handle, nullptr);

  const int32_t bs = batch.batch_size;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->hidden_dim_;
  const int32_t head_num = config_->head_num_;
  const int32_t head_size = config_->head_size_;
  const int32_t kv_mul = config_->kv_mul_;
  const int32_t kv_head_num = config_->kv_head_num_;
  cudaStream_t stream = cuda_config_->stream;

  // Get batch-sized buffers (pre-allocated in init_mem with max_batch_size)
  tensor::Tensor input_emb = get_buffer(ModelBufferType::kInputEmbeddings);
  tensor::Tensor rms_out = get_buffer(ModelBufferType::kOutputRMSNorm);
  tensor::Tensor query_buf = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor key_buf = get_buffer(ModelBufferType::kPagedKeyTemp);
  tensor::Tensor val_buf = get_buffer(ModelBufferType::kPagedValueTemp);
  tensor::Tensor mha_out = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor attn_out = get_buffer(ModelBufferType::kAttnOutput);
  tensor::Tensor w1_out = get_buffer(ModelBufferType::kW1Output);
  tensor::Tensor w3_out = get_buffer(ModelBufferType::kW3Output);
  tensor::Tensor w2_out = get_buffer(ModelBufferType::kW2Output);
  tensor::Tensor ffn_norm_out = get_buffer(ModelBufferType::kFFNRMSNorm);
  tensor::Tensor fwd_out = get_buffer(ModelBufferType::kForwardOutput);

  // 1. Embedding: [bs] token_ids -> [bs, dim]
  //    emb_kernel_cu already supports token_num > 1
  kernel::emb_kernel_cu(batch.token_ids, get_layer_weight0(qwen_layers_->embedding_layer_),
                        input_emb, std::abs(config_->vocab_size_), stream);

  // 2. Transformer layers
  for (int32_t layer = 0; layer < config_->layer_num_; ++layer) {
    auto& layer_allocator = kv_cache_manager_->allocator_mut(layer);

    // 2a. Attention RMSNorm: [bs, dim] -> [bs, dim]
    const auto& rms_weight = get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer));
    kernel::rmsnorm_kernel_cu_dim(input_emb, rms_weight, rms_out, dim, stream);

    // 2b. Wq: [bs, dim] -> [bs, dim]
    const auto& wq_weight = get_layer_weight0(qwen_layers_->wq_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(rms_out, wq_weight, query_buf, bs, cuda_config_.get());

    // 2c. Wk: [bs, dim] -> [bs, kv_dim]
    const auto& wk_weight = get_layer_weight0(qwen_layers_->wk_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(rms_out, wk_weight, key_buf, bs, cuda_config_.get());

    // 2d. Wv: [bs, dim] -> [bs, kv_dim]
    const auto& wv_weight = get_layer_weight0(qwen_layers_->wv_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(rms_out, wv_weight, val_buf, bs, cuda_config_.get());

    // 2e. Bias add (if applicable)
    auto wq_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_.at(layer));
    auto wk_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_.at(layer));
    auto wv_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_.at(layer));
    // Qwen2 has bias on q/k/v
    if (wq_matmul && !wq_matmul->get_bias(0).is_empty()) {
      kernel::add_bias_kernel_cu(query_buf, wq_matmul->get_bias(0), bs, dim, stream);
    }
    if (wk_matmul && !wk_matmul->get_bias(0).is_empty()) {
      kernel::add_bias_kernel_cu(key_buf, wk_matmul->get_bias(0), bs, kv_dim, stream);
    }
    if (wv_matmul && !wv_matmul->get_bias(0).is_empty()) {
      kernel::add_bias_kernel_cu(val_buf, wv_matmul->get_bias(0), bs, kv_dim, stream);
    }

    // 2f. Batched RoPE
    kernel::rope_kernel_batched_cu(dim, kv_dim, head_size,
                                   query_buf, key_buf, batch.positions,
                                   get_buffer(ModelBufferType::kSinCache),
                                   get_buffer(ModelBufferType::kCosCache),
                                   bs, stream);

    // 2g. Batched scatter KV to pages
    if (layer_allocator.uses_fp8_storage()) {
      kernel::scatter_kv_batch_to_pages_fp8_e4m3_cu(
          key_buf,
          val_buf,
          layer_allocator.key_pool(),
          layer_allocator.value_pool(),
          layer_allocator.key_scale_pool(),
          layer_allocator.value_scale_pool(),
          batch.slot_mapping,
          model_block_size,
          kv_head_num,
          head_size,
          bs,
          device_type_,
          cuda_config_.get());

      const bool launched = kernel::splitkv_batched_paged_mha_fp8_decode_cu(
          bs,
          head_num,
          head_size,
          kv_mul,
          query_buf,
          mha_out,
          layer_allocator.key_pool(),
          layer_allocator.value_pool(),
          layer_allocator.key_scale_pool(),
          layer_allocator.value_scale_pool(),
          batch.block_tables,
          batch.seq_lens,
          batch.max_blocks_per_seq,
          model_block_size,
          kv_head_num,
          get_buffer(ModelBufferType::kSplitKVPartialOut),
          get_buffer(ModelBufferType::kSplitKVPartialMax),
          get_buffer(ModelBufferType::kSplitKVPartialSum),
          device_type_,
          cuda_config_.get());
      CHECK(launched) << "FP8 KV cache decode launch failed for layer " << layer;
    } else {
      kernel::scatter_kv_batch_to_pages_cu(
          key_buf, val_buf,
          layer_allocator.key_pool(),
          layer_allocator.value_pool(),
          batch.slot_mapping, model_block_size,
          kv_head_num, head_size, bs,
          device_type_, cuda_config_.get());

      // 2h. Split-KV batched paged decode attention (FlashDecoding style)
      if (!kernel::splitkv_batched_paged_mha_fast_decode_cu(
              bs, head_num, head_size, kv_mul,
              query_buf, mha_out,
              layer_allocator.key_pool(),
              layer_allocator.value_pool(),
              batch.block_tables, batch.seq_lens,
              batch.max_blocks_per_seq, model_block_size, kv_head_num,
              get_buffer(ModelBufferType::kSplitKVPartialOut),
              get_buffer(ModelBufferType::kSplitKVPartialMax),
              get_buffer(ModelBufferType::kSplitKVPartialSum),
              device_type_, cuda_config_.get())) {
        kernel::splitkv_batched_paged_mha_decode_cu(
            bs, head_num, head_size, kv_mul,
            query_buf, mha_out,
            layer_allocator.key_pool(),
            layer_allocator.value_pool(),
            batch.block_tables, batch.seq_lens,
            batch.max_blocks_per_seq, model_block_size, kv_head_num,
            get_buffer(ModelBufferType::kSplitKVPartialOut),
            get_buffer(ModelBufferType::kSplitKVPartialMax),
            get_buffer(ModelBufferType::kSplitKVPartialSum),
            device_type_, cuda_config_.get());
      }
    }

    // 2i. Wo: [bs, dim] -> [bs, dim]
    const auto& wo_weight = get_layer_weight0(qwen_layers_->wo_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(mha_out, wo_weight, attn_out, bs, cuda_config_.get());

    // 2j. Residual add: input_emb += attn_out (element-wise, size=bs*dim)
    kernel::add_kernel_cu(input_emb, attn_out, input_emb, stream);

    // 2k. FFN RMSNorm
    const auto& ffn_rms_weight =
      get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(layer + config_->layer_num_));
    kernel::rmsnorm_kernel_cu_dim(input_emb, ffn_rms_weight, ffn_norm_out, dim, stream);

    // 2l. W1: [bs, dim] -> [bs, hidden_dim]
    const auto& w1_weight = get_layer_weight0(qwen_layers_->w1_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(ffn_norm_out, w1_weight, w1_out, bs, cuda_config_.get());

    // 2m. W3: [bs, dim] -> [bs, hidden_dim]
    const auto& w3_weight = get_layer_weight0(qwen_layers_->w3_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(ffn_norm_out, w3_weight, w3_out, bs, cuda_config_.get());

    // 2n. SwiGLU: element-wise on [bs * hidden_dim]
    kernel::swiglu_kernel_cu(w1_out, w3_out, w1_out, stream);

    // 2o. W2: [bs, hidden_dim] -> [bs, dim]
    const auto& w2_weight = get_layer_weight0(qwen_layers_->w2_layers_.at(layer));
    kernel::matmul_batch_kernel_cu(w1_out, w2_weight, w2_out, bs, cuda_config_.get());

    // 2p. Residual add
    kernel::add_kernel_cu(input_emb, w2_out, input_emb, stream);
  }

  // 3. Final RMSNorm
    const auto& final_rms_weight =
      get_layer_weight0(qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_));
  kernel::rmsnorm_kernel_cu_dim(input_emb, final_rms_weight, input_emb, dim, stream);

  // 4. Cls logits: [bs, dim] -> [bs, vocab_size]
  const auto& cls_weight = get_layer_weight0(qwen_layers_->cls_layer_);
  kernel::matmul_batch_kernel_cu(input_emb, cls_weight, fwd_out, bs, cuda_config_.get());

  return base::error::Success();
}

}  // namespace model
