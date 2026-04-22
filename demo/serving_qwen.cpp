// Continuous batching serving demo for Qwen2 Instruct.
#include <glog/logging.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "model/qwen2.h"
#include "serving/serving_benchmark_app.h"

namespace {

bool fp8_kv_cache_enabled() {
  const char* env = std::getenv("KUIPER_USE_FP8_KV_CACHE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool prefix_cache_enabled() {
  const char* env = std::getenv("KUIPER_ENABLE_PREFIX_CACHE");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return env[0] != '0';
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
    model_ = std::make_unique<model::Qwen2Model>(
        base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
    model_->set_kv_cache_memory_utilization(bench_config.kv_cache_memory_utilization);
    model_->set_serving_workspace_token_capacity(bench_config.max_num_batched_tokens);
    if (fp8_kv_cache_enabled()) {
      model_->set_use_fp8_kv_cache(true);
      model_->set_runtime_data_type(base::DataType::kDataTypeBf16);
    }

    auto init_status = model_->init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
      LOG(FATAL) << "Model init failed: " << init_status.get_err_msg();
      return false;
    }
    model_->kv_cache_manager()->set_prefix_cache_enabled(prefix_cache_enabled());
    return true;
  }

  int32_t max_model_batch_size() const override {
    return model::model_max_batch_size;
  }

  base::KVCacheManager* kv_cache_manager() const override {
    return model_->kv_cache_manager();
  }

  serving::ServingCapacityInfo serving_capacity_info() const override {
    return model_->serving_capacity_info();
  }

  void* model_stream() const override {
    auto context = model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }

  std::vector<int32_t> encode_prompt(const std::string& user_prompt) const override {
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
      const serving::MixedBatchMetadata& batch) const override {
    return model_->batch_sample(batch);
  }

  std::string postprocess_decoded_text(std::string text) const override {
    return trim_response(std::move(text));
  }

 private:
  std::unique_ptr<model::Qwen2Model> model_;
};

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  QwenServingBenchmarkApp app;
  return app.run(argc, argv);
}
