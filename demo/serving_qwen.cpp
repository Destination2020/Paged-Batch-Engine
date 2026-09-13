// Continuous batching serving demo for Qwen2 Instruct.
#include <glog/logging.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "base/base.h"
#include "data/data_client.h"
#include "data/ipc_pool.h"
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
 public:
  ~QwenServingBenchmarkApp() override { release_external_kv_pool(); }

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
    model_->set_kv_cache_blocks_per_layer(
        bench_config.kv_cache_blocks_per_layer);
    model_->set_serving_workspace_token_capacity(bench_config.max_num_batched_tokens);
    if (!bench_config.data_service_endpoint.empty()) {
      CHECK(!dual_gpu_local_pd && bench_config.pd_mode == "off")
          << "--data-service-endpoint currently supports the single-role "
             "serving path; use the multi-role launcher for PD";
      if (!bind_external_kv_pool(model_.get(), bench_config.data_service_endpoint,
                                 remote_decode_core
                                     ? bench_config.decode_device_id
                                     : remote_prefill_core
                                           ? bench_config.prefill_device_id
                                           : bench_config.device_id)) {
        return false;
      }
    }
    if (bench_config.radix_cache_config_explicit) {
      model_->set_radix_cache_enabled(bench_config.radix_cache_enabled);
    }
    if (bench_config.host_cache_enabled) {
      model_->set_host_cache_config({true, bench_config.host_cache_bytes,
          bench_config.host_cache_pages, bench_config.host_cache_inflight_pages});
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
      decode_model_->set_kv_cache_blocks_per_layer(
          bench_config.kv_cache_blocks_per_layer);
      decode_model_->set_serving_workspace_token_capacity(bench_config.max_num_batched_tokens);
      if (bench_config.radix_cache_config_explicit) {
        decode_model_->set_radix_cache_enabled(bench_config.radix_cache_enabled);
      }
      if (bench_config.host_cache_enabled) {
        decode_model_->set_host_cache_config({true, bench_config.host_cache_bytes,
            bench_config.host_cache_pages, bench_config.host_cache_inflight_pages});
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
  bool bind_external_kv_pool(model::Qwen2Model* model,
                             const std::string& endpoint, int32_t device) {
    CHECK_NE(model, nullptr);
    external_data_client_ = std::make_unique<data::DataClient>(endpoint);
    data::IpcPoolDescriptor descriptor;
    data::DataError status = external_data_client_->get_ipc_pool(&descriptor);
    if (status != data::DataError::kOk) {
      LOG(ERROR) << "Failed to get external KV pool descriptor: "
                 << static_cast<int>(status);
      return false;
    }
    if (descriptor.device != device) {
      LOG(ERROR) << "Serving external KV main path requires the pool and model "
                    "on the same GPU: pool_device="
                 << descriptor.device << " model_device=" << device;
      return false;
    }

    status = external_data_client_->reserve_ipc_slots(
        static_cast<uint32_t>(descriptor.num_blocks), &external_slot_grant_);
    if (status != data::DataError::kOk) {
      LOG(ERROR) << "Failed to reserve external KV slots: "
                 << static_cast<int>(status);
      return false;
    }

    external_ipc_pool_ = std::make_shared<data::CudaIpcPoolImport>();
    std::string error;
    uint64_t transferred_bytes = 0;
    double transfer_ms = 0.0;
    std::string transport;
    if (!external_ipc_pool_->open(descriptor, device, &error) ||
        !external_ipc_pool_->prepare_compute_view(
            nullptr, &transferred_bytes, &transfer_ms, &transport, &error)) {
      LOG(ERROR) << "Failed to import external KV pool: " << error;
      external_data_client_->release_ipc_slots(external_slot_grant_.token);
      external_slot_grant_ = {};
      external_ipc_pool_.reset();
      return false;
    }

    base::ExternalKVPoolBinding binding;
    binding.base = external_ipc_pool_->mapped();
    binding.bytes = descriptor.bytes;
    binding.device = device;
    binding.num_layers = descriptor.num_layers;
    binding.num_blocks = descriptor.num_blocks;
    binding.block_size = descriptor.block_size;
    binding.num_kv_heads = descriptor.num_kv_heads;
    binding.head_size = descriptor.head_size;
    binding.storage_dtype =
        static_cast<base::DataType>(descriptor.storage_dtype);
    binding.allocatable_blocks = external_slot_grant_.slots;
    binding.capsule = external_ipc_pool_;
    model->set_external_kv_pool(std::move(binding));
    external_kv_device_ = device;
    std::cout << "PBE_SERVING_EXTERNAL_KV_BOUND"
              << " pool_owner=" << descriptor.owner_incarnation
              << " grant_service="
              << external_slot_grant_.token.service_incarnation
              << " grant_consumer="
              << external_slot_grant_.token.consumer_incarnation
              << " grant=" << external_slot_grant_.token.lease_id
              << " slots=" << external_slot_grant_.slots.size()
              << " bytes=" << descriptor.bytes
              << " transport=" << transport
              << " transferred_bytes=" << transferred_bytes
              << " transfer_ms=" << transfer_ms << std::endl;
    return true;
  }

  void release_external_kv_pool() {
    // The model owns all request page tables and kernel-visible tensor views.
    // Destroy it and the IPC mapping before returning the authoritative grant.
    if (external_kv_device_ >= 0) {
      cudaSetDevice(external_kv_device_);
      const cudaError_t sync = cudaDeviceSynchronize();
      if (sync != cudaSuccess) {
        LOG(ERROR) << "External KV teardown synchronize failed: "
                   << cudaGetErrorString(sync);
      }
    }
    decode_model_.reset();
    model_.reset();
    external_ipc_pool_.reset();
    if (external_data_client_ && external_slot_grant_.token.lease_id != 0) {
      const data::DataError status = external_data_client_->release_ipc_slots(
          external_slot_grant_.token);
      if (status != data::DataError::kOk) {
        LOG(ERROR) << "Failed to release external KV grant: "
                   << static_cast<int>(status);
      } else {
        std::cout << "PBE_SERVING_EXTERNAL_KV_RELEASED"
                  << " grant=" << external_slot_grant_.token.lease_id
                  << std::endl;
      }
    }
    external_slot_grant_ = {};
    external_data_client_.reset();
    external_kv_device_ = -1;
  }

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
  std::unique_ptr<data::DataClient> external_data_client_;
  std::shared_ptr<data::CudaIpcPoolImport> external_ipc_pool_;
  data::IpcSlotGrant external_slot_grant_;
  int32_t external_kv_device_ = -1;
};

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  QwenServingBenchmarkApp app;
  return app.run(argc, argv);
}
