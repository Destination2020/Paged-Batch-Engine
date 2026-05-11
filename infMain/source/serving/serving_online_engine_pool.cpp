#include "serving/serving_online_engine_pool.h"

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "base/nvtx_utils.h"
#include "serving/serving_config.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {
namespace {

class ZmqRemoteRequestHandle final : public OnlineRequestHandle {
 public:
  void set_remote_request_id(int64_t request_id) {
    remote_request_id_.store(request_id);
    set_request_id(request_id);
  }

  int64_t remote_request_id() const { return remote_request_id_.load(); }

 private:
  std::atomic<int64_t> remote_request_id_{-1};
};

}  // namespace

SingleOnlineEnginePool::SingleOnlineEnginePool(ServingBenchmarkApp* app,
                                               const BenchConfig& config)
    : engine_(std::make_unique<OnlineServingEngine>(app, config)) {}

SingleOnlineEnginePool::~SingleOnlineEnginePool() { stop(); }

void SingleOnlineEnginePool::start() { engine_->start(); }

void SingleOnlineEnginePool::stop() {
  if (engine_) {
    engine_->stop();
  }
}

std::shared_ptr<OnlineRequestHandle> SingleOnlineEnginePool::submit(
    const OnlineGenerateRequest& request, std::string* error) {
  return engine_->submit(request, error);
}

void SingleOnlineEnginePool::cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
                                    const std::string& reason) {
  engine_->cancel(handle, reason);
}

int32_t SingleOnlineEnginePool::default_timeout_ms() const {
  return engine_->default_timeout_ms();
}

nlohmann::json SingleOnlineEnginePool::metrics_json() const {
  nlohmann::json body = engine_->metrics_json();
  body["engine_pool"]["type"] =
      body.contains("pd") ? body["pd"].value("mode", "dual-gpu") : "single";
  body["engine_pool"]["engine_count"] = body.contains("pd") ? 2 : 1;
  return body;
}

struct ZmqOnlineEnginePool::Impl {
  explicit Impl(const BenchConfig& config_value)
      : config(config_value), rpc(make_zmq_rpc_config(config_value)) {}

  base::Status request_response(const nlohmann::json& request,
                                nlohmann::json* response) const {
    return zmq_request_response(rpc, request, response);
  }

  void poll_tokens(std::shared_ptr<ZmqRemoteRequestHandle> handle) {
    while (!stopping.load() && !handle->cancelled()) {
      nlohmann::json response;
      base::Status status;
      {
        base::nvtx::ScopedRange range("zmq_token_poll_rpc",
                                      base::nvtx::kColorProcess);
        status = request_response(
            {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kToken)},
             {"request_id", handle->remote_request_id()},
             {"timeout_ms",
              std::max(1, std::min(config.engine_zmq_timeout_ms, 1000))}},
            &response);
      }
      if (!status) {
        handle->finish("", true, status.get_err_msg());
        return;
      }
      if (!response.value("ok", false)) {
        handle->finish("", true, response.value("error", "zmq_token_failed"));
        return;
      }
      const auto& event = response.at("event");
      const auto type = zmq_rpc_message_type_from_string(
          event.value("type", std::string()));
      if (type == ZmqRpcMessageType::kToken) {
        handle->push_token(event.value("text", ""));
        continue;
      }
      if (type == ZmqRpcMessageType::kFinal) {
        handle->finish(event.value("text", ""),
                       event.value("failed", false),
                       event.value("error", ""));
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  BenchConfig config;
  ZmqRpcConfig rpc;
  std::atomic<bool> started{false};
  std::atomic<bool> stopping{false};
  mutable std::mutex mu;
  std::vector<std::thread> polling_threads;
};

ZmqOnlineEnginePool::ZmqOnlineEnginePool(const BenchConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ZmqOnlineEnginePool::~ZmqOnlineEnginePool() { stop(); }

void ZmqOnlineEnginePool::start() {
  impl_->stopping.store(false);
  nlohmann::json response;
  auto status = impl_->request_response(
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kHealth)}},
      &response);
  if (!status) {
    LOG(ERROR) << "failed to connect to ZMQ EngineCore at "
               << impl_->rpc.endpoint << ": " << status.get_err_msg();
  } else if (!response.value("ok", false)) {
    LOG(ERROR) << "ZMQ EngineCore health check failed: "
               << response.value("error", "unknown");
  }
  impl_->started.store(true);
}

void ZmqOnlineEnginePool::stop() {
  impl_->stopping.store(true);
  std::vector<std::thread> local;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    local.swap(impl_->polling_threads);
  }
  for (auto& thread : local) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

std::shared_ptr<OnlineRequestHandle> ZmqOnlineEnginePool::submit(
    const OnlineGenerateRequest& request, std::string* error) {
  if (!impl_->started.load()) {
    start();
  }
  nlohmann::json response;
  OnlineGenerateRequest remote_request = request;
  remote_request.stream = true;
  auto status = impl_->request_response(
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kGenerate)},
       {"request", online_generate_request_to_json(remote_request)}},
      &response);
  if (!status) {
    if (error != nullptr) {
      *error = status.get_err_msg();
    }
    return nullptr;
  }
  if (!response.value("ok", false)) {
    if (error != nullptr) {
      *error = response.value("error", "zmq_submit_failed");
    }
    return nullptr;
  }

  auto handle = std::make_shared<ZmqRemoteRequestHandle>();
  handle->set_remote_request_id(response.value("request_id", -1));
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->polling_threads.emplace_back(
        [impl = impl_.get(), handle]() { impl->poll_tokens(handle); });
  }
  return handle;
}

void ZmqOnlineEnginePool::cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
                                 const std::string& reason) {
  if (!handle) {
    return;
  }
  handle->mark_cancelled();
  nlohmann::json response;
  auto status = impl_->request_response(
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kCancel)},
       {"request_id", handle->request_id()},
       {"reason", reason}},
      &response);
  if (!status) {
    LOG(WARNING) << "ZMQ cancel failed: " << status.get_err_msg();
  }
}

int32_t ZmqOnlineEnginePool::default_timeout_ms() const {
  return impl_->config.request_timeout_ms;
}

nlohmann::json ZmqOnlineEnginePool::metrics_json() const {
  nlohmann::json response;
  auto status = impl_->request_response(
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kMetrics)}},
      &response);
  if (!status) {
    return {{"error", status.get_err_msg()},
            {"process", {{"role", kOnlineProcessRoleZmqHttpApi},
                         {"engine_zmq_endpoint", impl_->rpc.endpoint}}}};
  }
  if (!response.value("ok", false)) {
    return {{"error", response.value("error", "zmq_metrics_failed")},
            {"process", {{"role", kOnlineProcessRoleZmqHttpApi},
                         {"engine_zmq_endpoint", impl_->rpc.endpoint}}}};
  }
  nlohmann::json metrics = response.value("metrics", nlohmann::json::object());
  metrics["process"]["role"] = kOnlineProcessRoleZmqHttpApi;
  metrics["process"]["engine_zmq_endpoint"] = impl_->rpc.endpoint;
  metrics["process"]["prefill_zmq_endpoint"] = impl_->config.prefill_zmq_endpoint;
  metrics["engine_pool"]["type"] = "zmq";
  return metrics;
}

}  // namespace serving
