#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

#include "serving/serving_online_engine.h"

namespace serving {

class ServingRuntime;

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
  virtual std::string default_model_id() const { return "paged-batch-engine"; }
  virtual nlohmann::json models_json() const {
    return {{"object", "list"}, {"data", nlohmann::json::array({
        {{"id", default_model_id()}, {"object", "model"}, {"owned_by", "pbe"}}})}};
  }
};

class SingleOnlineEnginePool final : public OnlineEnginePool {
 public:
  SingleOnlineEnginePool(ServingRuntime* app, const BenchConfig& config);
  ~SingleOnlineEnginePool() override;

  void start() override;
  void stop() override;
  std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest& request,
                                              std::string* error) override;
  void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
              const std::string& reason) override;
  int32_t default_timeout_ms() const override;
  nlohmann::json metrics_json() const override;
  std::string default_model_id() const override { return model_id_; }

 private:
  std::unique_ptr<OnlineServingEngine> engine_;
  std::string model_id_;
};

class MultiModelOnlineEnginePool final : public OnlineEnginePool {
 public:
  explicit MultiModelOnlineEnginePool(const BenchConfig& config);
  ~MultiModelOnlineEnginePool() override;
  void start() override;
  void stop() override;
  std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest&, std::string*) override;
  void cancel(const std::shared_ptr<OnlineRequestHandle>&, const std::string&) override;
  int32_t default_timeout_ms() const override;
  nlohmann::json metrics_json() const override;
  nlohmann::json models_json() const override;
  std::string default_model_id() const override { return default_model_; }
 private:
  std::map<std::string, std::unique_ptr<OnlineEnginePool>> pools_;
  std::string default_model_;
  int32_t timeout_ms_ = 0;
};

class ZmqOnlineEnginePool final : public OnlineEnginePool {
 public:
  explicit ZmqOnlineEnginePool(const BenchConfig& config);
  ~ZmqOnlineEnginePool() override;

  void start() override;
  void stop() override;
  std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest& request,
                                              std::string* error) override;
  void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
              const std::string& reason) override;
  int32_t default_timeout_ms() const override;
  nlohmann::json metrics_json() const override;
  std::string default_model_id() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_POOL_H_
