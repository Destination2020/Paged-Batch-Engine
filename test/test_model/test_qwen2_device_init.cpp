#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/base.h"
#include "model/qwen2.h"
#include "serving/scheduler.h"
#include "serving/serving_benchmark_app.h"

namespace {

bool cuda_device_count_at_least(int count) {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count >= count;
}

std::string env_or_empty(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

class DualQwenTestApp final : public serving::ServingBenchmarkApp {
 public:
	  DualQwenTestApp(std::unique_ptr<model::Qwen2Model> prefill_model,
	                  std::unique_ptr<model::Qwen2Model> decode_model)
	      : prefill_model_(std::move(prefill_model)),
	        decode_model_(std::move(decode_model)) {}

  const char* usage_name() const override { return "dual-qwen-test"; }
  bool initialize_model(const std::string&, const std::string&,
                        const serving::BenchConfig&) override {
    return true;
  }
  int32_t max_model_batch_size() const override { return model::model_max_batch_size; }
  base::KVCacheManager* kv_cache_manager() const override {
    return prefill_model_->kv_cache_manager();
  }
  serving::ServingCapacityInfo serving_capacity_info() const override {
    return prefill_model_->serving_capacity_info();
  }
  void* model_stream() const override {
    auto context = prefill_model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }
  std::vector<int32_t> encode_prompt(const std::string& prompt) const override {
    return prefill_model_->encode(prompt);
  }
  std::string decode_tokens(const std::vector<int32_t>& token_ids) const override {
    return prefill_model_->decode(token_ids);
  }
  bool is_sentence_ending(int32_t token) const override {
    return prefill_model_->is_sentence_ending(token);
  }
  base::Status forward_mixed_batch(const serving::MixedBatchMetadata& batch) const override {
    return prefill_model_->forward_mixed_batch(batch);
  }
  base::Status forward_decode_batch(const serving::MixedBatchMetadata& batch) const override {
    return prefill_model_->forward_decode_batch(batch);
  }
  serving::SampledTokenView batch_sample(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& output) const override {
    return prefill_model_->batch_sample(batch, output);
  }

  bool pd_dual_gpu_supported() const override {
    return prefill_model_ != nullptr && decode_model_ != nullptr;
  }
  base::KVCacheManager* pd_prefill_kv_cache_manager() const override {
    return prefill_model_->kv_cache_manager();
  }
  base::KVCacheManager* pd_decode_kv_cache_manager() const override {
    return decode_model_->kv_cache_manager();
  }
  serving::ServingCapacityInfo pd_prefill_serving_capacity_info() const override {
    return prefill_model_->serving_capacity_info();
  }
  serving::ServingCapacityInfo pd_decode_serving_capacity_info() const override {
    return decode_model_->serving_capacity_info();
  }
  serving::KVPoolDescriptor pd_prefill_kv_pool() const override {
    return make_pool(prefill_model_.get());
  }
  serving::KVPoolDescriptor pd_decode_kv_pool() const override {
    return make_pool(decode_model_.get());
  }
  void* pd_prefill_stream() const override {
    auto context = prefill_model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }
  void* pd_decode_stream() const override {
    auto context = decode_model_->device_context();
    return context != nullptr ? context->compute_queue : nullptr;
  }
  void* pd_transfer_stream() const override {
    auto context = decode_model_->device_context();
    return context != nullptr ? context->transfer_queue : nullptr;
  }
  void set_pd_prefill_layer_kv_connector(
      serving::LayerKVTransferConnector* connector,
      serving::LayerKVConnectorRole role) const override {
    prefill_model_->set_layer_kv_transfer_connector(connector, role);
  }
  void set_pd_decode_layer_kv_connector(
      serving::LayerKVTransferConnector* connector,
      serving::LayerKVConnectorRole role) const override {
    decode_model_->set_layer_kv_transfer_connector(connector, role);
  }
  base::Status pd_forward_prefill_batch(
      const serving::MixedBatchMetadata& batch) const override {
    return prefill_model_->forward_mixed_batch(batch);
  }
  base::Status pd_forward_decode_batch(
      const serving::MixedBatchMetadata& batch) const override {
    return decode_model_->forward_decode_batch(batch);
  }
  serving::SampledTokenView pd_batch_sample_prefill(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& output) const override {
    return prefill_model_->batch_sample(batch, output);
  }
  serving::SampledTokenView pd_batch_sample_decode(
      const serving::MixedBatchMetadata& batch,
      const serving::SchedulerOutput& output) const override {
    return decode_model_->batch_sample(batch, output);
  }

 private:
  static serving::KVPoolDescriptor make_pool(const model::Qwen2Model* model) {
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

	  std::unique_ptr<model::Qwen2Model> prefill_model_;
	  std::unique_ptr<model::Qwen2Model> decode_model_;
	};

std::vector<int32_t> run_single_model_generation(
    model::Qwen2Model* model,
    const std::vector<int32_t>& prompt_tokens,
    serving::GenerationConfig generation_config,
    const std::vector<int32_t>& checkpoint_after_steps = {},
    double* generation_ms = nullptr,
    double* checkpoint_ms = nullptr) {
  serving::SchedulerConfig config;
  config.max_num_seqs = model::model_max_batch_size;
  config.max_num_batched_tokens = 16;
  config.prefill_chunk_cap = 16;
  auto context = model->device_context();
  void* stream = context != nullptr ? context->compute_queue : nullptr;
  if (context != nullptr) {
    cudaSetDevice(context->device_id);
  }
  serving::Scheduler scheduler(config, model->kv_cache_manager());
  const int64_t client_id = scheduler.add_request(prompt_tokens, generation_config);
  int32_t completed_steps = 0;
  double checkpoint_total_ms = 0.0;
  const auto generation_start = std::chrono::steady_clock::now();
  while (scheduler.has_active_requests()) {
    if (context != nullptr) {
      cudaSetDevice(context->device_id);
    }
    serving::SchedulerOutput output = scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    const bool decode_only =
        output.num_decode_seqs > 0 && output.num_prefill_seqs == 0;
    serving::MixedBatchMetadata batch =
        decode_only ? scheduler.build_decode_batch(output, stream)
                    : scheduler.build_mixed_batch(output, stream);
    base::Status status = decode_only ? model->forward_decode_batch(batch)
                                      : model->forward_mixed_batch(batch);
    CHECK(status) << status.get_err_msg();
    serving::SampledTokenView sampled = model->batch_sample(batch, output);
    scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return model->is_sentence_ending(token); });
    ++completed_steps;
    if (std::find(checkpoint_after_steps.begin(), checkpoint_after_steps.end(),
                  completed_steps) != checkpoint_after_steps.end() &&
        scheduler.has_active_requests()) {
      const auto checkpoint_start = std::chrono::steady_clock::now();
      if (context != nullptr) {
        CHECK_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), cudaSuccess);
      }
      uint64_t revision = 0;
      CHECK(scheduler.suspend_request(client_id, &revision));
      CHECK(scheduler.restore_request(client_id, revision));
      checkpoint_total_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - checkpoint_start).count();
    }
  }
  if (generation_ms) {
    *generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_start).count();
  }
  if (checkpoint_ms) *checkpoint_ms = checkpoint_total_ms;
  auto finished = scheduler.pop_finished();
  for (const auto& seq : finished) {
    if (seq.client_request_id == client_id) {
      CHECK(!seq.failed) << seq.finish_reason;
      return seq.output_tokens;
    }
  }
  LOG(FATAL) << "single model generation result missing";
  return {};
}

}  // namespace

TEST(Qwen2DeviceInitTest, InitPassesRequestedDeviceIdToCudaRuntime) {
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, "dummy-tokenizer.json",
                          "dummy-model.bin", false);

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  const int invalid_device_id = device_count + 64;
  auto status = model.init(base::DeviceType::kDeviceCUDA, invalid_device_id);
  EXPECT_FALSE(status);
  EXPECT_NE(status.get_err_msg().find("cudaSetDevice(" +
                                      std::to_string(invalid_device_id) + ")"),
            std::string::npos);
  cudaGetLastError();
  if (device_count > 0) {
    cudaSetDevice(0);
  }
}

TEST(Qwen2DeviceInitTest, TwoInstancesCanLoadOnDistinctDevicesWithFixtureModel) {
  if (!cuda_device_count_at_least(2)) {
    GTEST_SKIP() << "At least two CUDA devices are required";
  }
  const std::string tokenizer_path = env_or_empty("PBE_QWEN2_TEST_TOKENIZER");
  const std::string model_path = env_or_empty("PBE_QWEN2_TEST_MODEL");
  if (tokenizer_path.empty() || model_path.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN2_TEST_TOKENIZER and PBE_QWEN2_TEST_MODEL to run";
  }

  model::Qwen2Model prefill_model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                                  model_path, false);
  model::Qwen2Model decode_model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                                 model_path, false);

  auto prefill_status = prefill_model.init(base::DeviceType::kDeviceCUDA, 0);
  auto decode_status = decode_model.init(base::DeviceType::kDeviceCUDA, 1);

  ASSERT_TRUE(prefill_status) << prefill_status.get_err_msg();
  ASSERT_TRUE(decode_status) << decode_status.get_err_msg();
  ASSERT_NE(prefill_model.device_context(), nullptr);
  ASSERT_NE(decode_model.device_context(), nullptr);
  EXPECT_EQ(prefill_model.device_context()->device_id, 0);
  EXPECT_EQ(decode_model.device_context()->device_id, 1);
}

TEST(Qwen2DeviceInitTest, DualGpuP2PMatchesSingleModelGreedyTokens) {
  if (!cuda_device_count_at_least(2)) {
    GTEST_SKIP() << "At least two CUDA devices are required";
  }
  const std::string tokenizer_path = env_or_empty("PBE_QWEN2_TEST_TOKENIZER");
  const std::string model_path = env_or_empty("PBE_QWEN2_TEST_MODEL");
  if (tokenizer_path.empty() || model_path.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN2_TEST_TOKENIZER and PBE_QWEN2_TEST_MODEL to run";
  }

  auto baseline_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  baseline_model->set_kv_cache_memory_utilization(0.20);
  baseline_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(baseline_model->init(base::DeviceType::kDeviceCUDA, 0));
  const std::vector<int32_t> prompt_tokens =
      baseline_model->encode("What is AI?");
  serving::GenerationConfig generation_config(4);
  const std::vector<int32_t> baseline_tokens =
      run_single_model_generation(baseline_model.get(), prompt_tokens, generation_config);

  auto prefill_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  auto decode_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  prefill_model->set_kv_cache_memory_utilization(0.20);
  decode_model->set_kv_cache_memory_utilization(0.20);
  prefill_model->set_serving_workspace_token_capacity(16);
  decode_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(prefill_model->init(base::DeviceType::kDeviceCUDA, 0));
  ASSERT_TRUE(decode_model->init(base::DeviceType::kDeviceCUDA, 1));

  DualQwenTestApp pd_app(std::move(prefill_model), std::move(decode_model));

  auto pd = pd_app.run_dual_gpu_p2p_generation(prompt_tokens, generation_config);
  ASSERT_FALSE(pd.failed) << pd.error;
  EXPECT_EQ(pd.output_tokens, baseline_tokens);
}

TEST(Qwen2DeviceInitTest, CheckpointResumeMatchesUninterruptedSampledTokens) {
  if (!cuda_device_count_at_least(1)) {
    GTEST_SKIP() << "A CUDA device is required";
  }
  const std::string tokenizer_path = env_or_empty("PBE_QWEN2_TEST_TOKENIZER");
  const std::string model_path = env_or_empty("PBE_QWEN2_TEST_MODEL");
  if (tokenizer_path.empty() || model_path.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN2_TEST_TOKENIZER and PBE_QWEN2_TEST_MODEL to run";
  }
  auto model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  model->set_kv_cache_memory_utilization(0.20);
  model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(model->init(base::DeviceType::kDeviceCUDA, 0));
  const auto prompt = model->encode("Explain a paged KV cache briefly.");
  serving::GenerationConfig generation_config(6, 0, true);
  generation_config.sampling.seed = UINT64_C(0xf123456789abcdef);
  generation_config.sampling.temperature = 0.8;
  generation_config.sampling.top_p = 0.9;
  generation_config.sampling.top_k = 20;
  double baseline_ms = 0.0;
  double resumed_ms = 0.0;
  double checkpoint_ms = 0.0;
  const auto baseline = run_single_model_generation(
      model.get(), prompt, generation_config, {}, &baseline_ms);
  model->kv_cache_manager()->clear_radix_cache();
  const auto resumed = run_single_model_generation(
      model.get(), prompt, generation_config, {1, 3}, &resumed_ms,
      &checkpoint_ms);
  EXPECT_EQ(resumed, baseline);
  std::cout << "P5_CHECKPOINT baseline_ms=" << baseline_ms
            << " resumed_ms=" << resumed_ms
            << " checkpoint_ms=" << checkpoint_ms
            << " checkpoints=2 output_tokens=" << resumed.size() << std::endl;
}

TEST(Qwen2DeviceInitTest, DualGpuNcclMatchesSingleModelGreedyTokens) {
#ifndef KUIPER_ENABLE_NCCL
  GTEST_SKIP() << "NCCL support is not enabled in this build";
#endif
  if (!cuda_device_count_at_least(2)) {
    GTEST_SKIP() << "At least two CUDA devices are required";
  }
  const std::string tokenizer_path = env_or_empty("PBE_QWEN2_TEST_TOKENIZER");
  const std::string model_path = env_or_empty("PBE_QWEN2_TEST_MODEL");
  if (tokenizer_path.empty() || model_path.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN2_TEST_TOKENIZER and PBE_QWEN2_TEST_MODEL to run";
  }

  auto baseline_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  baseline_model->set_kv_cache_memory_utilization(0.20);
  baseline_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(baseline_model->init(base::DeviceType::kDeviceCUDA, 0));
  const std::vector<int32_t> prompt_tokens =
      baseline_model->encode("What is AI?");
  serving::GenerationConfig generation_config(4);
  const std::vector<int32_t> baseline_tokens =
      run_single_model_generation(baseline_model.get(), prompt_tokens, generation_config);

  auto prefill_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  auto decode_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  prefill_model->set_kv_cache_memory_utilization(0.20);
  decode_model->set_kv_cache_memory_utilization(0.20);
  prefill_model->set_serving_workspace_token_capacity(16);
  decode_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(prefill_model->init(base::DeviceType::kDeviceCUDA, 0));
  ASSERT_TRUE(decode_model->init(base::DeviceType::kDeviceCUDA, 1));

  DualQwenTestApp pd_app(std::move(prefill_model), std::move(decode_model));

  auto pd = pd_app.run_dual_gpu_nccl_generation(prompt_tokens, generation_config);
  ASSERT_FALSE(pd.failed) << pd.error;
  EXPECT_EQ(pd.output_tokens, baseline_tokens);
}

TEST(Qwen2DeviceInitTest, DualGpuNcclLayerMatchesSingleModelGreedyTokens) {
#ifndef KUIPER_ENABLE_NCCL
  GTEST_SKIP() << "NCCL support is not enabled in this build";
#endif
  if (!cuda_device_count_at_least(2)) {
    GTEST_SKIP() << "At least two CUDA devices are required";
  }
  const std::string tokenizer_path = env_or_empty("PBE_QWEN2_TEST_TOKENIZER");
  const std::string model_path = env_or_empty("PBE_QWEN2_TEST_MODEL");
  if (tokenizer_path.empty() || model_path.empty()) {
    GTEST_SKIP() << "Set PBE_QWEN2_TEST_TOKENIZER and PBE_QWEN2_TEST_MODEL to run";
  }

  auto baseline_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  baseline_model->set_kv_cache_memory_utilization(0.20);
  baseline_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(baseline_model->init(base::DeviceType::kDeviceCUDA, 0));
  const std::vector<int32_t> prompt_tokens =
      baseline_model->encode("What is AI?");
  serving::GenerationConfig generation_config(4);
  const std::vector<int32_t> baseline_tokens =
      run_single_model_generation(baseline_model.get(), prompt_tokens, generation_config);

  auto prefill_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  auto decode_model = std::make_unique<model::Qwen2Model>(
      base::TokenizerType::kEncodeBpe, tokenizer_path, model_path, false);
  prefill_model->set_kv_cache_memory_utilization(0.20);
  decode_model->set_kv_cache_memory_utilization(0.20);
  prefill_model->set_serving_workspace_token_capacity(16);
  decode_model->set_serving_workspace_token_capacity(16);
  ASSERT_TRUE(prefill_model->init(base::DeviceType::kDeviceCUDA, 0));
  ASSERT_TRUE(decode_model->init(base::DeviceType::kDeviceCUDA, 1));

  DualQwenTestApp pd_app(std::move(prefill_model), std::move(decode_model));

  auto pd = pd_app.run_dual_gpu_nccl_layer_generation(prompt_tokens, generation_config);
  ASSERT_FALSE(pd.failed) << pd.error;
  EXPECT_EQ(pd.output_tokens, baseline_tokens);
}
