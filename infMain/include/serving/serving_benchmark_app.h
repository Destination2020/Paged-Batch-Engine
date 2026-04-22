// Generic continuous-batching benchmark app scaffold.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_
#define KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "serving/mixed_batch.h"
#include "serving/scheduler.h"
#include "serving/serving_capacity.h"
#include "serving/serving_config.h"
#include "serving/serving_metrics.h"

namespace serving {

// ServingBenchmarkApp owns the common offline serving lifecycle:
// argument parsing, prompt submission, warmup, scheduler loop, profiling,
// request metrics, and final summary. Model-specific demos only need to
// implement the virtual hooks below.
class ServingBenchmarkApp {
 public:
  virtual ~ServingBenchmarkApp() = default;

  int run(int argc, char* argv[]);

 protected:
  virtual const char* usage_name() const = 0;
  virtual bool initialize_model(const std::string& model_path,
                                const std::string& tokenizer_path,
                                const BenchConfig& bench_config) = 0;
  virtual int32_t max_model_batch_size() const = 0;
  virtual base::KVCacheManager* kv_cache_manager() const = 0;
  virtual ServingCapacityInfo serving_capacity_info() const = 0;
  virtual void* model_stream() const = 0;
  virtual std::vector<int32_t> encode_prompt(const std::string& user_prompt) const = 0;
  virtual std::string decode_tokens(const std::vector<int32_t>& token_ids) const = 0;
  virtual bool is_sentence_ending(int32_t token) const = 0;
  virtual base::Status forward_mixed_batch(const MixedBatchMetadata& batch) const = 0;
  virtual base::Status forward_decode_batch(const MixedBatchMetadata& batch) const = 0;
  virtual SampledTokenView batch_sample(const MixedBatchMetadata& batch) const = 0;

  virtual std::vector<std::string> default_prompts() const;
  virtual std::string postprocess_decoded_text(std::string text) const;

 private:
  using Clock = std::chrono::steady_clock;

  bool parse_args(int argc, char* argv[]);
  void collect_prompts(int argc, char* argv[]);
  void prepare_benchmark_config();
  void run_warmup();
  void create_scheduler();
  void submit_all_requests();
  void submit_requests_to(Scheduler& scheduler,
                          int32_t max_new_tokens,
                          bool quiet) const;
  void run_serving_loop();
  void run_serving_step(void* stream);
  StepProfile build_step_profile(const MixedBatchMetadata& batch,
                                 size_t sampled_token_count,
                                 Clock::time_point schedule_start,
                                 Clock::time_point schedule_end,
                                 Clock::time_point build_start,
                                 Clock::time_point build_end,
                                 Clock::time_point forward_start,
                                 Clock::time_point forward_end,
                                 Clock::time_point sample_start,
                                 Clock::time_point sample_end,
                                 Clock::time_point process_start,
                                 Clock::time_point process_end,
                                 Clock::time_point step_start,
                                 Clock::time_point step_end) const;
  void process_finished(const std::vector<SequenceState>& finished);
  void record_step_profile(const StepProfile& profile,
                           bool decode_only_step,
                           int32_t finished_count);
  void maybe_print_step_profile(const StepProfile& profile,
                                const MixedBatchMetadata& batch,
                                size_t sampled_token_count) const;
  void print_done(double duration, double throughput) const;

  std::string model_path_;
  std::string tokenizer_path_;
  BenchConfig bench_config_;
  std::vector<std::string> prompts_;
  std::unique_ptr<Scheduler> scheduler_;
  SummaryStats summary_;
  int32_t total_decode_steps_ = 0;
  int32_t step_ = 0;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_BENCHMARK_APP_H_
