#include "serving/serving_online_engine_pool.h"

#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

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

SingleOnlineEnginePool::SingleOnlineEnginePool(ServingRuntime* app,
                                               const BenchConfig& config)
    : engine_(std::make_unique<OnlineServingEngine>(app, config)),
      model_id_(config.model_id.empty() ? "paged-batch-engine" : config.model_id) {}

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

  void poll_tokens() {
    // This socket is created, used and destroyed on its owning I/O thread.
    std::unique_ptr<ZmqSocket> socket;
    while (true) {
      std::vector<std::shared_ptr<ZmqRemoteRequestHandle>> snapshot;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return stopping.load() || !requests.empty(); });
        if (stopping.load()) break;
        for (auto& entry : requests) snapshot.push_back(entry.second);
      }
      bool progress = false;
      for (const auto& handle : snapshot) {
        if (stopping.load()) break;
        bool terminal = handle->cancelled();
        if (!terminal) {
          try {
            nlohmann::json response;
            auto status = socket ? base::error::Success() : make_zmq_req_socket(rpc, &socket);
            if (status) status = socket->send_json({{"type", "token"},
                {"request_id", handle->remote_request_id()}, {"max_events", 32}});
            if (status) status = socket->recv_json(&response);
            if (!status) {
              // A token may have been consumed before a lost reply. Do not replay.
              socket.reset();
              std::lock_guard<std::mutex> lock(mu);
              failure = "worker_transport_failed: " + status.get_err_msg();
              for (auto& entry : requests) entry.second->finish("", true, failure);
              requests.clear();
              return; // fail the model channel once, not one timeout per request
            }
            if (!response.value("ok", false))
              throw std::runtime_error(response.value("error", "zmq_token_failed"));
            const auto events = response.contains("events") ? response.at("events")
                : nlohmann::json::array({response.at("event")});
            if (!events.is_array() || events.empty())
              throw std::runtime_error("invalid token event batch");
            for (const auto& event : events) {
              const auto type = event.value("type", std::string());
              if (type == "token") {
                handle->push_token(event.value("text", ""));
                progress = true;
              } else if (type == "final") {
                handle->finish(event.value("text", ""), event.value("failed", false),
                               event.value("error", ""));
                terminal = true;
                progress = true;
                break;
              } else if (type != "none") {
                throw std::runtime_error("invalid token event");
              }
            }
          } catch (const std::exception& error) {
            handle->finish("", true, error.what());
            terminal = true;
          }
        }
        if (terminal) {
          std::lock_guard<std::mutex> lock(mu);
          requests.erase(handle->remote_request_id());
        }
      }
      if (!progress) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::milliseconds(5), [&] { return stopping.load(); });
      }
    }
    std::lock_guard<std::mutex> lock(mu);
    for (auto& entry : requests) entry.second->finish("", true, "server_stopping");
    requests.clear();
  }

  BenchConfig config;
  ZmqRpcConfig rpc;
  std::atomic<bool> started{false};
  std::atomic<bool> stopping{false};
  mutable std::mutex mu;
  std::mutex lifecycle_mu;
  std::condition_variable cv;
  std::unordered_map<int64_t, std::shared_ptr<ZmqRemoteRequestHandle>> requests;
  std::thread polling_thread;
  std::string failure;
};

ZmqOnlineEnginePool::ZmqOnlineEnginePool(const BenchConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ZmqOnlineEnginePool::~ZmqOnlineEnginePool() { stop(); }

void ZmqOnlineEnginePool::start() {
  std::lock_guard<std::mutex> lock(impl_->lifecycle_mu);
  if (impl_->started.load()) return;
  if (impl_->stopping.load()) throw std::runtime_error("engine pool has stopped");
  nlohmann::json response;
  auto status = impl_->request_response(
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kHealth)}},
      &response);
  if (!status) {
    throw std::runtime_error("EngineCore health failed at " + impl_->rpc.endpoint + ": " + status.get_err_msg());
  } else if (!response.value("ok", false)) {
    throw std::runtime_error(response.value("error", "EngineCore not ready"));
  }
  if (!impl_->config.model_id.empty() && response.value("model_id", "") != impl_->config.model_id)
    throw std::runtime_error("EngineCore model identity mismatch");
  impl_->polling_thread = std::thread([impl = impl_.get()] { impl->poll_tokens(); });
  impl_->started.store(true);
}

void ZmqOnlineEnginePool::stop() {
  std::lock_guard<std::mutex> lock(impl_->lifecycle_mu);
  {
    std::lock_guard<std::mutex> state_lock(impl_->mu);
    impl_->stopping.store(true);
    impl_->started.store(false);
  }
  impl_->cv.notify_all();
  if (impl_->polling_thread.joinable()) impl_->polling_thread.join();
}

std::shared_ptr<OnlineRequestHandle> ZmqOnlineEnginePool::submit(
    const OnlineGenerateRequest& request, std::string* error) {
  if (!impl_->started.load()) {
    start();
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->failure.empty()) {
      if (error) *error = impl_->failure;
      return nullptr;
    }
  }
  nlohmann::json response;
  OnlineGenerateRequest remote_request = request;
  remote_request.stream = true;
  if (remote_request.timeout_ms <= 0) remote_request.timeout_ms = impl_->config.request_timeout_ms;
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
  handle->set_remote_request_id(response.value("request_id", int64_t{-1}));
  handle->model_id = request.model_id;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->stopping.load() && impl_->failure.empty()) {
      impl_->requests.emplace(handle->remote_request_id(), handle);
      impl_->cv.notify_one();
      return handle;
    }
  }
  cancel(handle, "server_stopping");
  if (error) *error = "server_stopping";
  return nullptr;
}

void ZmqOnlineEnginePool::cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
                                 const std::string& reason) {
  if (!handle) {
    return;
  }
  handle->mark_cancelled();
  handle->finish("", true, reason);
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->requests.erase(handle->request_id());
  }
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
  metrics["process"]["prefill_zmq_endpoint"] = impl_->config.ipc.prefill_zmq_endpoint;
  metrics["engine_pool"]["type"] = "zmq";
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    metrics["engine_pool"]["pending_remote_requests"] = impl_->requests.size();
    metrics["engine_pool"]["polling_threads"] = impl_->started.load() && impl_->failure.empty() ? 1 : 0;
    metrics["engine_pool"]["failure"] = impl_->failure;
  }
  return metrics;
}

MultiModelOnlineEnginePool::MultiModelOnlineEnginePool(const BenchConfig& config)
    : timeout_ms_(config.request_timeout_ms) {
  std::ifstream input(config.model_registry_path);
  if (!input) throw std::invalid_argument("Cannot open model registry");
  nlohmann::json registry; input >> registry;
  std::map<std::string, bool> endpoints;
  for (const auto& model : registry.at("models")) {
    const std::string id = model.at("id"), endpoint = model.at("endpoint");
    if (id.empty() || endpoint.empty() || pools_.count(id) || endpoints.count(endpoint))
      throw std::invalid_argument("Duplicate/empty model ID or endpoint");
    BenchConfig worker = config;
    worker.model_id = id;
    worker.ipc.engine_zmq_endpoint = endpoint;
    pools_[id] = std::make_unique<ZmqOnlineEnginePool>(worker);
    endpoints[endpoint] = true;
    if (default_model_.empty()) default_model_ = id;
  }
  default_model_ = registry.value("default_model", default_model_);
  if (!pools_.count(default_model_)) throw std::invalid_argument("Invalid default model");
}
MultiModelOnlineEnginePool::~MultiModelOnlineEnginePool() { stop(); }
void MultiModelOnlineEnginePool::start() { for (auto& p : pools_) p.second->start(); }
void MultiModelOnlineEnginePool::stop() { for (auto& p : pools_) p.second->stop(); }
int32_t MultiModelOnlineEnginePool::default_timeout_ms() const { return timeout_ms_; }
std::shared_ptr<OnlineRequestHandle> MultiModelOnlineEnginePool::submit(
    const OnlineGenerateRequest& request, std::string* error) {
  OnlineGenerateRequest routed = request;
  if (routed.model_id.empty()) routed.model_id = default_model_;
  const auto pool = pools_.find(routed.model_id);
  if (pool == pools_.end()) { if (error) *error = "unknown_model"; return nullptr; }
  auto handle = pool->second->submit(routed, error);
  if (handle) handle->model_id = routed.model_id;
  return handle;
}
void MultiModelOnlineEnginePool::cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
    const std::string& reason) {
  if (!handle) return;
  const auto pool = pools_.find(handle->model_id);
  if (pool != pools_.end()) pool->second->cancel(handle, reason);
}
nlohmann::json MultiModelOnlineEnginePool::models_json() const {
  nlohmann::json models = nlohmann::json::array();
  for (const auto& pool : pools_) models.push_back({{"id", pool.first}, {"object", "model"}, {"owned_by", "pbe"}});
  return {{"object", "list"}, {"data", models}};
}
nlohmann::json MultiModelOnlineEnginePool::metrics_json() const {
  nlohmann::json models = nlohmann::json::object();
  for (const auto& pool : pools_) models[pool.first] = pool.second->metrics_json();
  return {{"engine_pool", {{"type", "multi-model-process"}, {"engine_count", pools_.size()}}}, {"models", models}};
}
}  // namespace serving
