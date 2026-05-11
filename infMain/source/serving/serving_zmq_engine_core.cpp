#include "serving/serving_zmq_engine_core.h"

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include "base/nvtx_utils.h"
#include "serving/serving_online_engine_pool.h"
#include "serving/serving_benchmark_app.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {
namespace {

struct ActiveRequest {
  std::shared_ptr<OnlineRequestHandle> handle;
  std::thread token_thread;
  std::deque<nlohmann::json> events;
  bool final_delivered = false;
  std::atomic<bool> cancelled{false};
};

nlohmann::json make_error_response(const std::string& error) {
  return {
      {"ok", false},
      {"error", error},
  };
}

nlohmann::json make_ok_response() {
  return {
      {"ok", true},
  };
}

SchedulerConfig make_scheduler_config(const ServingBenchmarkApp* app,
                                      const BenchConfig& config) {
  SchedulerConfig sched_config;
  sched_config.max_num_seqs = app->max_model_batch_size();
  sched_config.max_num_batched_tokens = config.max_num_batched_tokens;
  sched_config.prefill_chunk_cap = config.prefill_chunk_cap;
  sched_config.policy = config.scheduling_policy;
  sched_config.long_prefill_token_threshold =
      config.long_prefill_token_threshold;
  sched_config.max_partial_prefills = config.max_partial_prefills;
  sched_config.max_long_partial_prefills = config.max_long_partial_prefills;
  return sched_config;
}

int32_t blocks_for_tokens(int32_t num_tokens, int32_t block_size) {
  if (num_tokens <= 0) {
    return 0;
  }
  return (num_tokens + block_size - 1) / block_size;
}

int32_t additional_blocks_needed_for_tokens(int32_t current_tokens,
                                            int32_t appended_tokens,
                                            int32_t block_size) {
  if (appended_tokens <= 0) {
    return 0;
  }
  return blocks_for_tokens(current_tokens + appended_tokens, block_size) -
         blocks_for_tokens(current_tokens, block_size);
}

class RemotePrefillBatchWorker {
 public:
  RemotePrefillBatchWorker(ServingBenchmarkApp* app, BenchConfig config)
      : app_(app), config_(std::move(config)) {
    CHECK_NE(app_, nullptr);
  }

  ~RemotePrefillBatchWorker() { stop(); }

  void start() {
    stopping_.store(false);
    worker_ = std::thread([this]() { run_loop(); });
  }

  void stop() {
    stopping_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  int64_t submit(std::vector<int32_t> prompt_tokens,
                 GenerationConfig generation_config,
                 std::string* error) {
    base::nvtx::ScopedRange range("prefill_worker_submit",
                                  base::nvtx::kColorProcess);
    if (prompt_tokens.empty()) {
      if (error != nullptr) {
        *error = "empty_prompt";
      }
      return -1;
    }
    generation_config.normalize();
    const int64_t prefill_request_id = next_request_id_.fetch_add(1);
    auto request = std::make_shared<RequestState>();
    request->prefill_request_id = prefill_request_id;
    request->client_request_id.value =
        "remote-zmq-prefill-" + std::to_string(prefill_request_id);
    request->handoff_id.value =
        static_cast<uint64_t>(prefill_request_id + 1);
    request->prompt_tokens = std::move(prompt_tokens);
    request->generation_config = generation_config;
    request->seq.request_id = -1;
    request->seq.client_request_id = prefill_request_id;
    request->seq.prompt_tokens = request->prompt_tokens;
    request->seq.generation_config = generation_config;
    request->seq.status = SequenceStatus::kWaiting;

    {
      std::lock_guard<std::mutex> exec_lock(exec_mu_);
      request->seq.request_id =
          app_->kv_cache_manager()->register_request_with_radix_cache(
              request->prompt_tokens);
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      requests_[prefill_request_id] = request;
      schedule_order_.push_back(prefill_request_id);
    }
    cv_.notify_all();
    return prefill_request_id;
  }

  bool poll(int64_t prefill_request_id, RemotePrefillResult* result,
            bool* ready, std::string* error) {
    base::nvtx::ScopedRange range("prefill_worker_poll",
                                  base::nvtx::kColorProcess);
    CHECK_NE(ready, nullptr);
    *ready = false;
    std::shared_ptr<RequestState> request;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = requests_.find(prefill_request_id);
      if (it == requests_.end()) {
        if (error != nullptr) {
          *error = "prefill_request_not_found";
        }
        return false;
      }
      request = it->second;
      if (!request->ready) {
        return true;
      }
      if (result != nullptr) {
        *result = request->result;
      }
      *ready = true;
      requests_.erase(it);
      schedule_order_.erase(std::remove(schedule_order_.begin(),
                                        schedule_order_.end(),
                                        prefill_request_id),
                            schedule_order_.end());
    }
    return true;
  }

  base::Status send_kv(const KVBlockManifest& manifest,
                       const std::string& nccl_unique_id) {
    base::nvtx::ScopedRange range("prefill_kv_send_nccl",
                                  base::nvtx::kColorMemcpy);
    std::lock_guard<std::mutex> lock(exec_mu_);
    return app_->run_remote_nccl_kv_send(manifest, nccl_unique_id);
  }

  void release(HandoffId handoff_id) {
    base::nvtx::ScopedRange range("prefill_release_kv",
                                  base::nvtx::kColorProcess);
    std::lock_guard<std::mutex> lock(exec_mu_);
    app_->release_remote_prefill(handoff_id);
  }

  nlohmann::json metrics_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    int32_t pending = 0;
    int32_t ready = 0;
    for (const auto& item : requests_) {
      if (item.second->ready) {
        ++ready;
      } else {
        ++pending;
      }
    }
    return {{"pending_requests", pending},
            {"ready_results", ready},
            {"completed_requests", completed_requests_.load()},
            {"failed_requests", failed_requests_.load()},
            {"prefill_tokens", prefill_tokens_.load()}};
  }

 private:
  struct RequestState {
    int64_t prefill_request_id = -1;
    GlobalRequestId client_request_id;
    HandoffId handoff_id;
    SequenceState seq;
    std::vector<int32_t> prompt_tokens;
    GenerationConfig generation_config;
    bool in_flight = false;
    bool ready = false;
    RemotePrefillResult result;
  };

  void run_loop() {
    const cudaError_t cuda_status = cudaSetDevice(config_.prefill_device_id);
    if (cuda_status != cudaSuccess) {
      LOG(ERROR) << "cudaSetDevice(" << config_.prefill_device_id
                 << ") failed in prefill worker: "
                 << cudaGetErrorString(cuda_status);
    }
    Scheduler metadata_builder(make_scheduler_config(app_, config_),
                               app_->kv_cache_manager());
    while (!stopping_.load()) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
          return stopping_.load() || has_schedulable_request_locked();
        });
        if (stopping_.load()) {
          break;
        }
      }

      SchedulerOutput output;
      std::vector<std::shared_ptr<RequestState>> scheduled;
      base::Status status = base::error::Success();
      MixedBatchMetadata batch;
      SampledTokenView sampled;
      {
        std::lock_guard<std::mutex> exec_lock(exec_mu_);
        {
          base::nvtx::ScopedRange range("prefill_build_batch",
                                        base::nvtx::kColorSchedule);
          std::lock_guard<std::mutex> lock(mu_);
          build_prefill_batch_locked(&output, &scheduled);
        }
        if (output.total_tokens <= 0) {
          continue;
        }
        {
          base::nvtx::ScopedRange range("prefill_build_metadata",
                                        base::nvtx::kColorMetadata);
          batch = metadata_builder.build_mixed_batch(output, app_->model_stream());
        }
        {
          base::nvtx::ScopedRange range("prefill_forward",
                                        base::nvtx::kColorForward);
          status = app_->pd_forward_prefill_batch(batch);
        }
        if (status) {
          base::nvtx::ScopedRange range("prefill_sample",
                                        base::nvtx::kColorSample);
          sampled = app_->pd_batch_sample_prefill(batch, output);
        }
        {
          base::nvtx::ScopedRange range("prefill_finish_step",
                                        base::nvtx::kColorProcess);
          finish_prefill_step(output, batch, sampled, scheduled, status);
        }
      }
    }
  }

  bool has_schedulable_request_locked() const {
    for (const auto id : schedule_order_) {
      auto it = requests_.find(id);
      if (it == requests_.end()) {
        continue;
      }
      const auto& request = it->second;
      if (!request->ready && !request->in_flight &&
          request->seq.computed_tokens <
              static_cast<int32_t>(request->prompt_tokens.size())) {
        return true;
      }
    }
    return false;
  }

  void build_prefill_batch_locked(
      SchedulerOutput* output,
      std::vector<std::shared_ptr<RequestState>>* scheduled) {
    CHECK_NE(output, nullptr);
    CHECK_NE(scheduled, nullptr);
    int32_t remaining_budget = config_.max_num_batched_tokens;
    int32_t remaining_free_blocks =
        app_->kv_cache_manager()->num_free_blocks(0) +
        app_->kv_cache_manager()->radix_cache_evictable_blocks();
    const int32_t block_size = app_->kv_cache_manager()->block_size();
    for (const auto id : schedule_order_) {
      if (remaining_budget <= 0 ||
          static_cast<int32_t>(scheduled->size()) >= app_->max_model_batch_size()) {
        break;
      }
      auto it = requests_.find(id);
      if (it == requests_.end()) {
        continue;
      }
      auto request = it->second;
      if (request->ready || request->in_flight) {
        continue;
      }
      const int32_t target =
          static_cast<int32_t>(request->prompt_tokens.size());
      const int32_t remaining = target - request->seq.computed_tokens;
      if (remaining <= 0) {
        continue;
      }
      int32_t chunk =
          std::min({remaining, remaining_budget, config_.prefill_chunk_cap});
      const int32_t current_tokens =
          app_->kv_cache_manager()->get_context_len(request->seq.request_id);
      const int32_t room_in_current_block =
          current_tokens > 0 && current_tokens % block_size != 0
              ? block_size - (current_tokens % block_size)
              : 0;
      const int32_t max_fit =
          room_in_current_block + remaining_free_blocks * block_size;
      chunk = std::min(chunk, max_fit);
      if (chunk <= 0) {
        continue;
      }
      remaining_free_blocks -= additional_blocks_needed_for_tokens(
          current_tokens, chunk, block_size);
      request->seq.scheduled_tokens = chunk;
      request->seq.status = SequenceStatus::kRunning;
      request->in_flight = true;
      output->scheduled_seqs.push_back(&request->seq);
      output->num_tokens_per_seq.push_back(chunk);
      output->num_prefill_seqs++;
      output->total_tokens += chunk;
      remaining_budget -= chunk;
      scheduled->push_back(request);
    }
  }

  void finish_prefill_step(
      const SchedulerOutput& output,
      const MixedBatchMetadata& batch,
      const SampledTokenView& sampled,
      const std::vector<std::shared_ptr<RequestState>>& scheduled,
      const base::Status& status) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!status) {
      for (const auto& request : scheduled) {
        request->in_flight = false;
        request->result = make_failed_result(*request, status.get_err_msg());
        request->ready = true;
        ++failed_requests_;
        free_source_request_unlocked(*request);
      }
      cv_.notify_all();
      return;
    }

    for (size_t i = 0; i < scheduled.size(); ++i) {
      auto& request = scheduled[i];
      request->in_flight = false;
      request->seq.computed_tokens += output.num_tokens_per_seq[i];
      prefill_tokens_ += output.num_tokens_per_seq[i];
      const int32_t target =
          static_cast<int32_t>(request->prompt_tokens.size());
      if (request->seq.computed_tokens < target) {
        continue;
      }
      int32_t first_token = -1;
      for (int32_t sample_idx = 0; sample_idx < sampled.size(); ++sample_idx) {
        if (sample_idx < static_cast<int32_t>(batch.sample_row_to_request.size()) &&
            batch.sample_row_to_request[sample_idx] == static_cast<int32_t>(i)) {
          first_token = sampled.tokens[sample_idx];
          break;
        }
      }
      if (first_token < 0) {
        request->result =
            make_failed_result(*request, "prefill_missing_first_token");
        request->ready = true;
        ++failed_requests_;
        free_source_request_unlocked(*request);
        continue;
      }
      request->result = make_success_result(*request, first_token);
      if (request->result.failed) {
        ++failed_requests_;
        free_source_request_unlocked(*request);
      } else {
        app_->retain_remote_prefill(request->handoff_id,
                                   request->seq.request_id);
        ++completed_requests_;
      }
      request->ready = true;
    }
    cv_.notify_all();
  }

  RemotePrefillResult make_failed_result(const RequestState& request,
                                         const std::string& error) const {
    RemotePrefillResult result;
    result.client_request_id = request.client_request_id;
    result.handoff_id = request.handoff_id;
    result.prompt_tokens = request.prompt_tokens;
    result.failed = true;
    result.error = error;
    return result;
  }

  RemotePrefillResult make_success_result(const RequestState& request,
                                          int32_t first_token) const {
    RemotePrefillResult result;
    result.client_request_id = request.client_request_id;
    result.handoff_id = request.handoff_id;
    result.prompt_tokens = request.prompt_tokens;
    result.computed_tokens =
        static_cast<int32_t>(request.prompt_tokens.size());
    result.first_token = first_token;
    result.first_tokens = {first_token};
    result.src_pool = app_->pd_prefill_kv_pool();
    const int32_t required_blocks =
        blocks_for_tokens(result.computed_tokens, result.src_pool.block_size);
    result.layers.reserve(result.src_pool.layer_num);
    for (int32_t layer_idx = 0; layer_idx < result.src_pool.layer_num;
         ++layer_idx) {
      RemoteKVLayerPayload layer;
      layer.layer_idx = layer_idx;
      const auto& block_ids =
          app_->kv_cache_manager()->get_block_ids(request.seq.request_id,
                                                  layer_idx);
      if (static_cast<int32_t>(block_ids.size()) < required_blocks) {
        result.failed = true;
        result.error = "remote_prefill_source_block_count_insufficient";
        return result;
      }
      layer.src_block_ids.assign(block_ids.begin(),
                                 block_ids.begin() + required_blocks);
      result.layers.push_back(std::move(layer));
    }
    return result;
  }

  void free_source_request_unlocked(const RequestState& request) const {
    if (request.seq.request_id >= 0 &&
        app_->kv_cache_manager()->is_valid_request(request.seq.request_id)) {
      app_->kv_cache_manager()->free_request(request.seq.request_id);
    }
  }

  ServingBenchmarkApp* app_ = nullptr;
  BenchConfig config_;
  std::thread worker_;
  mutable std::mutex mu_;
  mutable std::mutex exec_mu_;
  std::condition_variable cv_;
  std::deque<int64_t> schedule_order_;
  std::unordered_map<int64_t, std::shared_ptr<RequestState>> requests_;
  std::atomic<int64_t> next_request_id_{1};
  std::atomic<bool> stopping_{false};
  std::atomic<int64_t> completed_requests_{0};
  std::atomic<int64_t> failed_requests_{0};
  std::atomic<int64_t> prefill_tokens_{0};
};

base::Status run_zmq_prefill_engine_core_server(ServingBenchmarkApp* app,
                                                const BenchConfig& config) {
  CHECK_NE(app, nullptr);
  ZmqRpcConfig rpc_config = make_prefill_zmq_rpc_config(config);
  std::unique_ptr<ZmqSocket> socket;
  auto status = make_zmq_rep_socket(rpc_config, &socket);
  if (!status) {
    return status;
  }

  LOG(INFO) << "ZMQ Prefill EngineCore listening on " << rpc_config.endpoint;
  std::cout << "ZMQ_PREFILL_ENGINE_CORE_READY endpoint=" << rpc_config.endpoint
            << " role=" << kOnlineProcessRoleZmqPrefillEngineCore << std::endl;

  RemotePrefillBatchWorker prefill_worker(app, config);
  prefill_worker.start();

  while (true) {
    nlohmann::json request;
    status = socket->recv_json(&request);
    if (!status) {
      LOG(ERROR) << status.get_err_msg();
      continue;
    }

    nlohmann::json response;
    const auto type = zmq_rpc_message_type_from_string(
        request.value("type", std::string()));
    if (type == ZmqRpcMessageType::kHealth) {
      response = make_ok_response();
      response["status"] = "ok";
    } else if (type == ZmqRpcMessageType::kMetrics) {
      response = make_ok_response();
      response["metrics"]["engine"]["mode"] = "prefill";
      response["metrics"]["prefill_worker"] = prefill_worker.metrics_json();
      response["metrics"]["process"]["role"] =
          kOnlineProcessRoleZmqPrefillEngineCore;
      response["metrics"]["process"]["zmq_endpoint"] = rpc_config.endpoint;
      response["metrics"]["pd"]["mode"] = config.pd_mode;
      response["metrics"]["pd"]["transfer_backend"] =
          pd_transfer_backend(config.pd_mode);
    } else if (type == ZmqRpcMessageType::kPrefill) {
      std::vector<int32_t> prompt_tokens;
      if (request.contains("prompt_tokens")) {
        prompt_tokens = request.at("prompt_tokens").get<std::vector<int32_t>>();
      } else if (request.contains("request")) {
        auto generate_request =
            online_generate_request_from_json(request.at("request"));
        prompt_tokens = app->encode_prompt(generate_request.prompt);
      }
      GenerationConfig generation_config;
      if (request.contains("generation_config")) {
        generation_config =
            generation_config_from_json(request.at("generation_config"));
      } else if (request.contains("request") &&
                 request.at("request").contains("generation_config")) {
        generation_config = generation_config_from_json(
            request.at("request").at("generation_config"));
      }
      RemotePrefillResult result =
          app->run_remote_prefill_generation(std::move(prompt_tokens),
                                             generation_config);
      response = make_ok_response();
      response["type"] =
          zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefillResult);
      response["prefill_result"] = remote_prefill_result_to_json(result);
    } else if (type == ZmqRpcMessageType::kPrefillSubmit) {
      base::nvtx::ScopedRange range("prefill_core_handle_submit",
                                    base::nvtx::kColorProcess);
      std::vector<int32_t> prompt_tokens;
      if (request.contains("prompt_tokens")) {
        prompt_tokens = request.at("prompt_tokens").get<std::vector<int32_t>>();
      } else if (request.contains("request")) {
        auto generate_request =
            online_generate_request_from_json(request.at("request"));
        prompt_tokens = app->encode_prompt(generate_request.prompt);
      }
      GenerationConfig generation_config;
      if (request.contains("generation_config")) {
        generation_config =
            generation_config_from_json(request.at("generation_config"));
      } else if (request.contains("request") &&
                 request.at("request").contains("generation_config")) {
        generation_config = generation_config_from_json(
            request.at("request").at("generation_config"));
      }
      std::string error;
      const int64_t prefill_request_id =
          prefill_worker.submit(std::move(prompt_tokens), generation_config,
                                &error);
      if (prefill_request_id < 0) {
        response = make_error_response(error);
      } else {
        response = make_ok_response();
        response["prefill_request_id"] = prefill_request_id;
      }
    } else if (type == ZmqRpcMessageType::kPrefillPoll) {
      base::nvtx::ScopedRange range("prefill_core_handle_poll",
                                    base::nvtx::kColorProcess);
      const int64_t prefill_request_id =
          request.value("prefill_request_id", int64_t{-1});
      RemotePrefillResult result;
      bool ready = false;
      std::string error;
      if (!prefill_worker.poll(prefill_request_id, &result, &ready, &error)) {
        response = make_error_response(error);
      } else {
        response = make_ok_response();
        response["ready"] = ready;
        if (ready) {
          response["type"] =
              zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefillResult);
          response["prefill_result"] = remote_prefill_result_to_json(result);
        }
      }
    } else if (type == ZmqRpcMessageType::kKvTransfer) {
      base::nvtx::ScopedRange range("prefill_core_handle_kv_transfer",
                                    base::nvtx::kColorMemcpy);
      if (request.value("backend", std::string()) != "nccl") {
        response = make_error_response("unsupported_kv_transfer_backend");
      } else {
        KVBlockManifest manifest =
            kv_block_manifest_from_json(request.at("manifest"));
        const std::string nccl_unique_id =
            hex_json_to_binary(request.value("nccl_unique_id", std::string()));
        status = prefill_worker.send_kv(manifest, nccl_unique_id);
        if (status) {
          response = make_ok_response();
          response["type"] =
              zmq_rpc_message_type_name(ZmqRpcMessageType::kKvTransferResult);
        } else {
          response = make_error_response(status.get_err_msg());
        }
      }
    } else if (type == ZmqRpcMessageType::kKvRelease) {
      base::nvtx::ScopedRange range("prefill_core_handle_kv_release",
                                    base::nvtx::kColorProcess);
      HandoffId handoff_id;
      handoff_id.value = request.value("handoff_id", uint64_t{0});
      prefill_worker.release(handoff_id);
      response = make_ok_response();
    } else {
      response = make_error_response("unknown_message_type_for_prefill_core");
    }

    status = socket->send_json(response);
    if (!status) {
      LOG(ERROR) << status.get_err_msg();
    }
  }
}

void append_event(std::mutex* mu,
                  std::condition_variable* cv,
                  std::shared_ptr<ActiveRequest> active,
                  nlohmann::json event) {
  {
    std::lock_guard<std::mutex> lock(*mu);
    active->events.push_back(std::move(event));
  }
  cv->notify_all();
}

void stream_handle_events(std::mutex* mu,
                          std::condition_variable* cv,
                          int64_t request_id,
                          std::shared_ptr<ActiveRequest> active) {
  base::nvtx::ScopedRange thread_range("zmq_stream_handle_events",
                                       base::nvtx::kColorProcess);
  while (true) {
    if (active->cancelled.load() || active->handle->cancelled()) {
      return;
    }
    std::string token_text;
    bool finished = false;
    bool failed = false;
    std::string error;
    const bool has_token = active->handle->wait_next_token_for(
        100, &token_text, &finished, &failed, &error);
    if (has_token) {
      base::nvtx::ScopedRange range("zmq_stream_append_token",
                                    base::nvtx::kColorProcess);
      append_event(mu, cv, active,
                   {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kToken)},
                    {"request_id", request_id},
                    {"text", token_text}});
      continue;
    }
    if (finished) {
      base::nvtx::ScopedRange range("zmq_stream_append_final",
                                    base::nvtx::kColorProcess);
      bool full_failed = failed;
      std::string full_error = error;
      std::string full_text;
      active->handle->wait_full_text(0, &full_text, &full_failed, &full_error);
      append_event(mu, cv, active,
                   {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kFinal)},
                    {"request_id", request_id},
                    {"failed", full_failed},
                    {"error", full_error},
                    {"text", full_text}});
      return;
    }
  }
}

}  // namespace

base::Status run_zmq_engine_core_server(ServingBenchmarkApp* app,
                                        const BenchConfig& config) {
  CHECK_NE(app, nullptr);
  if (config.online_process_role == kOnlineProcessRoleZmqPrefillEngineCore) {
    return run_zmq_prefill_engine_core_server(app, config);
  }
  ZmqRpcConfig rpc_config = make_zmq_rpc_config(config);
  std::unique_ptr<ZmqSocket> socket;
  auto status = make_zmq_rep_socket(rpc_config, &socket);
  if (!status) {
    return status;
  }

  SingleOnlineEnginePool engine_pool(app, config);
  engine_pool.start();

  std::mutex mu;
  std::condition_variable cv;
  std::unordered_map<int64_t, std::shared_ptr<ActiveRequest>> active_requests;
  std::atomic<int64_t> next_request_id{1};

  LOG(INFO) << "ZMQ EngineCore listening on " << rpc_config.endpoint;
  std::cout << "ZMQ_ENGINE_CORE_READY endpoint=" << rpc_config.endpoint
            << " role=" << config.online_process_role << std::endl;

  while (true) {
    nlohmann::json request;
    status = socket->recv_json(&request);
    if (!status) {
      LOG(ERROR) << status.get_err_msg();
      continue;
    }

    nlohmann::json response;
    const auto type = zmq_rpc_message_type_from_string(
        request.value("type", std::string()));
    if (type == ZmqRpcMessageType::kHealth) {
      response = make_ok_response();
      response["status"] = "ok";
    } else if (type == ZmqRpcMessageType::kMetrics) {
      response = make_ok_response();
      response["metrics"] = engine_pool.metrics_json();
      response["metrics"]["process"]["role"] = config.online_process_role;
      response["metrics"]["process"]["zmq_endpoint"] = rpc_config.endpoint;
    } else if (type == ZmqRpcMessageType::kGenerate) {
      OnlineGenerateRequest generate_request =
          online_generate_request_from_json(request.at("request"));
      generate_request.stream = true;
      const int64_t request_id = next_request_id.fetch_add(1);
      std::string error;
      auto handle = engine_pool.submit(generate_request, &error);
      if (!handle) {
        response = make_error_response(error);
      } else {
        auto active = std::make_shared<ActiveRequest>();
        active->handle = handle;
        {
          std::lock_guard<std::mutex> lock(mu);
          active_requests[request_id] = active;
        }
        active->token_thread = std::thread([&mu, &cv, request_id, active]() {
          stream_handle_events(&mu, &cv, request_id, active);
        });
        response = make_ok_response();
        response["request_id"] = request_id;
      }
    } else if (type == ZmqRpcMessageType::kToken) {
      const int64_t request_id = request.value("request_id", -1);
      std::shared_ptr<ActiveRequest> active;
      {
        std::unique_lock<std::mutex> lock(mu);
        auto it = active_requests.find(request_id);
        if (it != active_requests.end()) {
          active = it->second;
        }
        if (active == nullptr) {
          response = make_error_response("request_not_found");
        } else if (active->events.empty()) {
          response = make_ok_response();
          response["event"] = {{"type", "none"}, {"request_id", request_id}};
        } else {
          response = make_ok_response();
          response["event"] = std::move(active->events.front());
          active->events.pop_front();
          if (zmq_rpc_message_type_from_string(
                  response["event"].value("type", std::string())) ==
              ZmqRpcMessageType::kFinal) {
            active->final_delivered = true;
          }
        }
      }

      if (active != nullptr && active->final_delivered) {
        if (active->token_thread.joinable() &&
            active->token_thread.get_id() != std::this_thread::get_id()) {
          active->token_thread.join();
        }
        std::lock_guard<std::mutex> lock(mu);
        active_requests.erase(request_id);
      }
    } else if (type == ZmqRpcMessageType::kCancel) {
      const int64_t request_id = request.value("request_id", -1);
      const std::string reason = request.value("reason", "cancelled");
      std::shared_ptr<ActiveRequest> active;
      {
        std::lock_guard<std::mutex> lock(mu);
        auto it = active_requests.find(request_id);
        if (it != active_requests.end()) {
          active = it->second;
        }
      }
      if (active != nullptr) {
        active->cancelled.store(true);
        engine_pool.cancel(active->handle, reason);
        append_event(&mu, &cv, active,
                     {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kFinal)},
                      {"request_id", request_id},
                      {"failed", true},
                      {"error", reason},
                      {"text", ""}});
        if (active->token_thread.joinable()) {
          active->token_thread.detach();
        }
      }
      response = make_ok_response();
    } else {
      response = make_error_response("unknown_message_type");
    }

    status = socket->send_json(response);
    if (!status) {
      LOG(ERROR) << status.get_err_msg();
    }
  }
}

}  // namespace serving
