#include "serving/serving_online_engine_pool.h"

#include <utility>

#include "serving/serving_benchmark_app.h"
#include "serving/serving_config.h"

namespace serving {

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
  body["engine_pool"]["type"] = "single";
  body["engine_pool"]["engine_count"] = 1;
  return body;
}

}  // namespace serving
