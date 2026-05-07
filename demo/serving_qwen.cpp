// Continuous batching serving demo for Qwen2 Instruct.
#include <glog/logging.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "model/qwen2.h"
#include "serving/serving_benchmark_app.h"
#include "serving/serving_zmq_rpc.h"

namespace {

bool fp8_kv_cache_enabled() {
  const char* env = std::getenv("KUIPER_USE_FP8_KV_CACHE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

std::string build_chatml_prompt(const std::string& user_prompt) {
  return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
         "assistant.\n<|im_end|>\n<|im_start|>user\n" +
         user_prompt + "\n<|im_end|>\n<|im_start|>assistant\n";
}

std::string trim_response(std::string response) {
  const std::vector<std::string> stop_markers = {
      "<|im_end|>", "<|endoftext|>", "<|im_start|>"};
  for (const auto& marker : stop_markers) {
    auto pos = response.find(marker);
    if (pos != std::string::npos) {
      response = response.substr(0, pos);
    }
  }
  while (!response.empty() &&
         (response.back() == '\n' || response.back() == '\r' || response.back() == ' ')) {
    response.pop_back();
  }
  return response;
}

class QwenServingBenchmarkApp final : public serving::ServingBenchmarkApp {
 protected:
  const char* usage_name() const override { return "./serving_qwen"; }

  bool initialize_model(const std::string& model_path,
                        const std::string& tokenizer_path,
                        const serving::BenchConfig& bench_config) override {
    const bool remote_prefill_core =
        bench_config.online_process_role ==
        serving::kOnlineProcessRoleZmqPrefillEngineCore;
    const bool remote_decode_core =
        bench_config.online_process_role ==
        serving::kOnlineProcessRoleZmqDecodeEngineCore;
    const bool dual_gpu_local_pd =
        serving::is_dual_gpu_pd_mode(bench_config.pd_mode) &&
        !remote_prefill_core && !remote_decode_core;
    model_ = std::make_unique<model::Qwen2Model>(
        base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
    const double per_model_kv_utilization =
        dual_gpu_local_pd
            ? std::min(bench_config.kv_cache_memory_utilization, 0.45)
            : bench_config.kv_cache_memory_utilization;
    model_->set_kv_cache_memory_utilization(per_model_kv_utilization);
    model_->set_serving_workspace_token_capacity(bench_config.max_num_batched_tokens);
    if (bench_config.radix_cache_config_explicit) {
      model_->set_radix_cache_enabled(bench_config.radix_cache_enabled);
    }
    if (fp8_kv_cache_enabled()) {
      model_->set_use_fp8_kv_cache(true);
      model_->set_runtime_data_type(base::DataType::kDataTypeBf16);
    }

    auto init_status = model_->init(base::DeviceType::kDeviceCUDA,
                                    remote_decode_core
                                        ? bench_config.decode_device_id
                                        : remote_prefill_core
                                        ? bench_config.prefill_device_id
                                        : dual_gpu_local_pd
                                        ? bench_config.prefill_device_id
                                        : bench_config.device_id);
    if (!init_status) {
      LOG(FATAL) << "Model init failed: " << init_status.get_err_msg();
      return false;
    }

    if (dual_gpu_local_pd) {
      decode_model_ = std::make_unique<model::Qwen2Model>(
          base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
      decode_model_->set_kv_cache_memory_utilization(per_model_kv_utilization);
      decode_model_->set_serving_workspace_token_capacity(bench_config.max_num_batched_tokens);
      if (bench_config.radix_cache_config_explicit) {
        decode_model_->set_radix_cache_enabled(bench_config.radix_cache_enabled);
      }
      if (fp8_kv_cache_enabled()) {
        decode_model_->set_use_fp8_kv_cache(true);
        decode_model_->set_runtime_data_type(base::DataType::kDataTypeBf16);
      }

      auto decode_status = decode_model_->init(base::DeviceType::kDeviceCUDA,
                                               bench_config.decode_device_id);
      if (!decode_status) {
        LOG(FATAL) << "Decode model init failed: " << decode_status.get_err_msg();
        return false;
      }
    }
    return true;
  }

  int32_t max_model_batch_size() const override {
    return model::model_max_batch_size;
  }

  base::KVCacheManager* kv_cache_manager() const override {
    return model_->kv_cache_manager();
  }

  serving::ServingCapacityInfo serving_capacity_info() const override {
    if (decode_model_ != nullptr) {
      return decode_model_->serving_capacity_info();
    }
    return model_->serving_capacity_info();
  }

  void* model_stream() const override {
    auto context = model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }

  std::vector<int32_t> encode_prompt(const std::string& user_prompt) const override {
    if (user_prompt.rfind("<|im_start|>", 0) == 0) {
      return model_->encode(user_prompt);
    }
    return model_->encode(build_chatml_prompt(user_prompt));
  }

  std::string decode_tokens(const std::vector<int32_t>& token_ids) const override {
    return model_->decode(token_ids);
  }

  bool is_sentence_ending(int32_t token) const override {
    return model_->is_sentence_ending(token);
  }

  base::Status forward_mixed_batch(const serving::MixedBatchMetadata& batch) const override {
    return model_->forward_mixed_batch(batch);
  }

  base::Status forward_decode_batch(const serving::MixedBatchMetadata& batch) const override {
    return model_->forward_decode_batch(batch);
  }

  serving::SampledTokenView batch_sample(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& sched_out) const override {
    return model_->batch_sample(batch, sched_out);
  }

  bool pd_dual_gpu_supported() const override {
    return model_ != nullptr && decode_model_ != nullptr;
  }

  base::KVCacheManager* pd_prefill_kv_cache_manager() const override {
    return model_->kv_cache_manager();
  }

  base::KVCacheManager* pd_decode_kv_cache_manager() const override {
    return decode_model_ != nullptr ? decode_model_->kv_cache_manager()
                                    : model_->kv_cache_manager();
  }

  serving::ServingCapacityInfo pd_prefill_serving_capacity_info() const override {
    return model_->serving_capacity_info();
  }

  serving::ServingCapacityInfo pd_decode_serving_capacity_info() const override {
    return decode_model_ != nullptr ? decode_model_->serving_capacity_info()
                                    : model_->serving_capacity_info();
  }

  serving::KVPoolDescriptor pd_prefill_kv_pool() const override {
    return make_pool_descriptor(model_.get());
  }

  serving::KVPoolDescriptor pd_decode_kv_pool() const override {
    return make_pool_descriptor(decode_model_ != nullptr ? decode_model_.get() : model_.get());
  }

  void* pd_prefill_stream() const override {
    auto context = model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }

  void* pd_decode_stream() const override {
    const auto* active_model = decode_model_ != nullptr ? decode_model_.get() : model_.get();
    auto context = active_model->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }

  void* pd_transfer_stream() const override {
    const auto* active_model = decode_model_ != nullptr ? decode_model_.get() : model_.get();
    auto context = active_model->device_context();
    return context != nullptr ? context->transfer_queue : nullptr;
  }

  void set_pd_prefill_layer_kv_connector(
      serving::LayerKVTransferConnector* connector,
      serving::LayerKVConnectorRole role) const override {
    model_->set_layer_kv_transfer_connector(connector, role);
  }

  void set_pd_decode_layer_kv_connector(
      serving::LayerKVTransferConnector* connector,
      serving::LayerKVConnectorRole role) const override {
    (decode_model_ != nullptr ? decode_model_.get() : model_.get())
        ->set_layer_kv_transfer_connector(connector, role);
  }

  base::Status pd_forward_prefill_batch(
      const serving::MixedBatchMetadata& batch) const override {
    return model_->forward_mixed_batch(batch);
  }

  base::Status pd_forward_decode_batch(
      const serving::MixedBatchMetadata& batch) const override {
    return (decode_model_ != nullptr ? decode_model_.get() : model_.get())
        ->forward_decode_batch(batch);
  }

  serving::SampledTokenView pd_batch_sample_prefill(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& sched_out) const override {
    return model_->batch_sample(batch, sched_out);
  }

  serving::SampledTokenView pd_batch_sample_decode(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& sched_out) const override {
    return (decode_model_ != nullptr ? decode_model_.get() : model_.get())
        ->batch_sample(batch, sched_out);
  }

  std::string postprocess_decoded_text(std::string text) const override {
    return trim_response(std::move(text));
  }

 private:
  serving::KVPoolDescriptor make_pool_descriptor(const model::Qwen2Model* model) const {
    auto info = model->serving_capacity_info();
    serving::KVPoolDescriptor pool;
    auto context = model->device_context();
    pool.device_id = context != nullptr ? context->device_id : 0;
    pool.layer_num = info.layer_num;
    pool.block_size = info.block_size;
    pool.kv_head_num = info.kv_head_num;
    pool.head_size = info.head_size;
    pool.dtype = info.runtime_data_type;
    pool.storage_mode = model->kv_cache_storage_mode();
    return pool;
  }

  std::unique_ptr<model::Qwen2Model> model_;
  std::unique_ptr<model::Qwen2Model> decode_model_;
};

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  QwenServingBenchmarkApp app;
  return app.run(argc, argv);
}
