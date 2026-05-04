#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_

#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "serving/serving_online_engine.h"

namespace serving {

class ServingBenchmarkApp;

class OnlineEnginePool {
 public:
  virtual ~OnlineEnginePool() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest& request,
                                                      std::string* error) = 0;
  virtual void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
                      const std::string& reason) = 0;
  virtual int32_t default_timeout_ms() const = 0;
  virtual nlohmann::json metrics_json() const = 0;
};

class SingleOnlineEnginePool final : public OnlineEnginePool {
 public:
  SingleOnlineEnginePool(ServingBenchmarkApp* app, const BenchConfig& config);
  ~SingleOnlineEnginePool() override;

  void start() override;
  void stop() override;
  std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest& request,
                                              std::string* error) override;
  void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
              const std::string& reason) override;
  int32_t default_timeout_ms() const override;
  nlohmann::json metrics_json() const override;

 private:
  std::unique_ptr<OnlineServingEngine> engine_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_
