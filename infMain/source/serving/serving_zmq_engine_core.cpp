#include "serving/serving_zmq_engine_core.h"

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

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
    } else if (type == ZmqRpcMessageType::kLayerKvTransfer) {
      if (request.value("backend", std::string()) != "nccl") {
        response = make_error_response("unsupported_layer_kv_transfer_backend");
      } else {
        std::vector<int32_t> prompt_tokens;
        if (request.contains("prompt_tokens")) {
          prompt_tokens =
              request.at("prompt_tokens").get<std::vector<int32_t>>();
        }
        GenerationConfig generation_config;
        if (request.contains("generation_config")) {
          generation_config =
              generation_config_from_json(request.at("generation_config"));
        }
        LayerKVTransferRequest layer_request =
            layer_kv_transfer_request_from_json(request.at("layer_request"));
        const std::string nccl_unique_id =
            hex_json_to_binary(request.value("nccl_unique_id", std::string()));
        RemotePrefillResult result =
            app->run_remote_prefill_layer_generation(
                std::move(prompt_tokens), generation_config, layer_request,
                nccl_unique_id);
        response = make_ok_response();
        response["type"] =
            zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefillResult);
        response["prefill_result"] = remote_prefill_result_to_json(result);
      }
    } else if (type == ZmqRpcMessageType::kKvTransfer) {
      if (request.value("backend", std::string()) != "nccl") {
        response = make_error_response("unsupported_kv_transfer_backend");
      } else {
        KVBlockManifest manifest =
            kv_block_manifest_from_json(request.at("manifest"));
        const std::string nccl_unique_id =
            hex_json_to_binary(request.value("nccl_unique_id", std::string()));
        status = app->run_remote_nccl_kv_send(manifest, nccl_unique_id);
        if (status) {
          response = make_ok_response();
          response["type"] =
              zmq_rpc_message_type_name(ZmqRpcMessageType::kKvTransferResult);
        } else {
          response = make_error_response(status.get_err_msg());
        }
      }
    } else if (type == ZmqRpcMessageType::kKvRelease) {
      HandoffId handoff_id;
      handoff_id.value = request.value("handoff_id", uint64_t{0});
      app->release_remote_prefill(handoff_id);
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
      append_event(mu, cv, active,
                   {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kToken)},
                    {"request_id", request_id},
                    {"text", token_text}});
      continue;
    }
    if (finished) {
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
