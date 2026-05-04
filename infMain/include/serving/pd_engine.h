#ifndef KUIPER_INCLUDE_SERVING_PD_ENGINE_H_
#define KUIPER_INCLUDE_SERVING_PD_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "serving/generation_config.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_handoff.h"
#include "serving/pd_worker.h"
#include "serving/scheduler.h"

namespace serving {

class ServingBenchmarkApp;

struct InProcPDEngineConfig {
  SchedulerConfig scheduler_config;
  KVPoolDescriptor kv_pool;
};

struct PrefillSubmitResult {
  int64_t request_id = -1;
};

struct DecodeReadySubmitResult {
  int64_t request_id = -1;
};

class InProcPrefillEngine final {
 public:
  InProcPrefillEngine(ServingBenchmarkApp* app,
                      base::KVCacheManager* kv_manager,
                      InProcPDEngineConfig config);

  base::Status submit_request(std::vector<int32_t> prompt_tokens,
                              GenerationConfig generation_config,
                              PrefillSubmitResult* result);
  SchedulerOutput schedule_step();
  base::Status execute_step(const SchedulerOutput& output,
                            void* stream,
                            PDWorkerStepOutput* step_output);
  base::Status execute_and_process_step(const SchedulerOutput& output,
                                        void* stream,
                                        PDWorkerStepOutput* step_output);
  Scheduler* scheduler() { return scheduler_.get(); }
  base::KVCacheManager* kv_manager() const { return kv_manager_; }
  const KVPoolDescriptor& kv_pool() const { return config_.kv_pool; }

 private:
  ServingBenchmarkApp* app_ = nullptr;
  base::KVCacheManager* kv_manager_ = nullptr;
  InProcPDEngineConfig config_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<PrefillWorker> worker_;
};

class InProcDecodeEngine final {
 public:
  InProcDecodeEngine(ServingBenchmarkApp* app,
                     base::KVCacheManager* kv_manager,
                     InProcPDEngineConfig config);

  base::Status submit_decode_ready_request(std::vector<int32_t> prompt_tokens,
                                           GenerationConfig generation_config,
                                           int64_t* request_id);
  base::Status submit_decode_ready_request(const DecodeKVReservation& reservation,
                                           std::vector<int32_t> prompt_tokens,
                                           GenerationConfig generation_config,
                                           DecodeReadySubmitResult* result);
  SchedulerOutput schedule_step();
  base::Status execute_step(const SchedulerOutput& output,
                            void* stream,
                            PDWorkerStepOutput* step_output);

  Scheduler* scheduler() { return scheduler_.get(); }
  base::KVCacheManager* kv_manager() const { return kv_manager_; }
  const KVPoolDescriptor& kv_pool() const { return config_.kv_pool; }

 private:
  ServingBenchmarkApp* app_ = nullptr;
  base::KVCacheManager* kv_manager_ = nullptr;
  InProcPDEngineConfig config_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<DecodeWorker> worker_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_ENGINE_H_
