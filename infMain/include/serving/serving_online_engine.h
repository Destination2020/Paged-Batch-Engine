#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_H_

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "serving/generation_config.h"
#include "serving/gpu_worker.h"
#include "serving/scheduler.h"
#include "serving/serving_config.h"

namespace serving {

class ServingBenchmarkApp;

struct OnlineGenerateRequest {
  std::string prompt;
  GenerationConfig generation_config;
  bool stream = false;
  int32_t timeout_ms = 0;
};

class OnlineRequestHandle {
 public:
  void set_request_id(int64_t request_id);
  int64_t request_id() const;
  void mark_cancelled();
  bool cancelled() const;
  void push_token(const std::string& token_text);
  void finish(std::string full_text, bool failed, std::string error);
  bool wait_next_token(std::string* token_text, bool* finished, bool* failed,
                       std::string* error);
  bool wait_next_token_for(int32_t timeout_ms, std::string* token_text,
                           bool* finished, bool* failed, std::string* error);
  bool wait_full_text(int32_t timeout_ms, std::string* text, bool* failed,
                      std::string* error);

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::string> token_texts_;
  size_t next_idx_ = 0;
  int64_t request_id_ = -1;
  std::string full_text_;
  bool finished_ = false;
  bool failed_ = false;
  bool cancelled_ = false;
  std::string error_;
};

class OnlineServingEngine {
 public:
  OnlineServingEngine(ServingBenchmarkApp* app, const BenchConfig& config);
  ~OnlineServingEngine();

  void start();
  void stop();

  std::shared_ptr<OnlineRequestHandle> submit(const OnlineGenerateRequest& request,
                                              std::string* error);
  int32_t default_timeout_ms() const;
  nlohmann::json metrics_json() const;
  void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
              const std::string& reason);

 private:
  struct PendingSubmission {
    std::vector<int32_t> prompt_tokens;
    GenerationConfig generation_config;
    std::vector<std::string> stop;
    std::shared_ptr<OnlineRequestHandle> handle;
  };

  void run_loop();
  void run_pd_loop();
  void run_remote_pd_batch_loop();
  void run_pd_submission(PendingSubmission submission);
  void flush_submissions();
  void flush_cancellations();
  void run_step(void* stream);
  void publish_sampled_tokens(const SchedulerOutput& output,
                              const MixedBatchMetadata& batch,
                              const SampledTokenView& sampled_tokens);
  void publish_finished();

  ServingBenchmarkApp* app_ = nullptr;
  BenchConfig config_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<GpuWorker> gpu_worker_;
  std::thread worker_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<PendingSubmission> submissions_;
  std::vector<std::pair<int64_t, std::string>> cancellations_;
  int32_t pending_submissions_ = 0;
  bool stopping_ = false;
  int64_t next_pd_request_id_ = 0;
  int32_t pd_active_requests_ = 0;
  int64_t pd_completed_requests_ = 0;
  int64_t pd_failed_requests_ = 0;
  int64_t pd_generated_tokens_ = 0;
  std::unordered_map<int64_t, std::shared_ptr<OnlineRequestHandle>> handles_;
  std::unordered_map<int64_t, std::vector<std::string>> stop_by_request_;
  std::unordered_map<int64_t, std::string> streamed_text_by_request_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_ENGINE_H_
