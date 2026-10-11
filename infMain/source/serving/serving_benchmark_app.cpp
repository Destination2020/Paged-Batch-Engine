#include "serving/serving_benchmark_app.h"
#include "serving/serving_config_parser.h"
#include "serving/serving_online_server.h"
#include "serving/serving_zmq_engine_core.h"
#include "serving/serving_zmq_rpc.h"
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <cuda_runtime_api.h>
#if defined(KUIPER_ENABLE_NCCL)
#include <nccl.h>
#endif
#include "base/nvtx_utils.h"
#include "base/alloc.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_coordinator.h"
#include "serving/pd_handoff_builder.h"

namespace serving {

namespace {

using Duration = std::chrono::duration<double, std::milli>;

std::string format_double(double value, int precision = 3) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << value;
  return os.str();
}

std::string scheduling_policy_name(SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::kFCFS: return "fcfs";
    case SchedulingPolicy::kPriority: return "priority";
  }
  return "unknown";
}

const char* preemption_policy_name(PreemptionPolicy policy) {
  switch (policy) {
    case PreemptionPolicy::kRecompute: return "recompute";
    case PreemptionPolicy::kCheckpoint: return "checkpoint";
    case PreemptionPolicy::kAuto: return "auto";
  }
  return "unknown";
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  if (values.size() == 1) return values.front();
  const double rank = (static_cast<double>(values.size()) - 1.0) * p;
  const auto lower = static_cast<size_t>(std::floor(rank));
  const auto upper = static_cast<size_t>(std::ceil(rank));
  if (lower == upper) return values[lower];
  const double weight = rank - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double average(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

int32_t round_down_to_multiple(int32_t value, int32_t multiple) {
  if (multiple <= 1) {
    return value;
  }
  return (value / multiple) * multiple;
}

int32_t round_up_to_multiple(int32_t value, int32_t multiple) {
  if (multiple <= 1) {
    return value;
  }
  return ((value + multiple - 1) / multiple) * multiple;
}

AutoScheduleEstimate estimate_auto_schedule(
    const ServingCapacityInfo& info,
    const PromptTokenStats& prompt_stats,
    int32_t max_new_tokens,
    double kv_cache_memory_utilization) {
  AutoScheduleEstimate estimate;
  const int32_t block_size = std::max(1, info.block_size);
  const int32_t free_kv_blocks = std::max(0, info.free_kv_blocks);
  // The KV pool itself is already sized from kv_cache_memory_utilization during
  // model initialization, so do not apply the same fraction a second time here.
  estimate.usable_free_kv_blocks = free_kv_blocks;
  estimate.kv_token_budget = estimate.usable_free_kv_blocks * block_size;

  if (info.serving_workspace_token_capacity > 0) {
    estimate.device_workspace_bytes_budget = info.serving_workspace_reserved_bytes;
    estimate.workspace_token_budget = info.serving_workspace_token_capacity;
  } else {
    estimate.device_workspace_bytes_budget = static_cast<size_t>(
        static_cast<double>(info.device_free_memory_bytes) * kv_cache_memory_utilization);
    estimate.workspace_token_budget =
        (info.workspace_bytes_per_token > 0 && estimate.device_workspace_bytes_budget > 0)
            ? static_cast<int32_t>(
                  std::min<size_t>(static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                                   estimate.device_workspace_bytes_budget /
                                       info.workspace_bytes_per_token))
            : estimate.kv_token_budget;
  }

  estimate.capacity_token_budget = std::max(
      1, std::min({std::max(1, estimate.kv_token_budget),
                   std::max(1, estimate.workspace_token_budget)}));
  if (info.serving_workspace_token_capacity > 0) {
    estimate.capacity_token_budget =
        std::min(estimate.capacity_token_budget,
                 info.serving_workspace_token_capacity);
  }
  estimate.scheduler_seq_window = std::max(
      1, std::min(info.max_batch_size, std::max(1, prompt_stats.prompt_count)));
  const int32_t representative_prompt_tokens = std::max(
      block_size,
      prompt_stats.p95_prompt_tokens > 0 ? prompt_stats.p95_prompt_tokens
                                         : std::max(prompt_stats.p50_prompt_tokens, block_size));
  const bool decode_heavy_workload =
      max_new_tokens >= std::max(block_size, prompt_stats.p50_prompt_tokens);
  estimate.target_decode_concurrency = decode_heavy_workload
                                           ? std::max(1, estimate.scheduler_seq_window - 1)
                                           : std::max(1, estimate.scheduler_seq_window / 2);
  estimate.prefill_reserved_seqs =
      std::max(1, estimate.scheduler_seq_window - estimate.target_decode_concurrency);

  const int32_t desired_prefill_chunk_tokens =
      decode_heavy_workload
          ? std::max(
                block_size,
                round_down_to_multiple(
                    std::max(
                        block_size,
                        static_cast<int32_t>(std::floor(
                            static_cast<double>(
                                std::max(prompt_stats.p50_prompt_tokens, block_size)) *
                            0.75))),
                    block_size))
          : std::max(block_size, round_up_to_multiple(
                                      static_cast<int32_t>(std::ceil(
                                          representative_prompt_tokens * 0.75)),
                                      block_size));
  estimate.prompt_prefill_chunk_target = std::max(
      block_size, round_up_to_multiple(desired_prefill_chunk_tokens, block_size));
  estimate.target_decode_token_window =
      decode_heavy_workload
          ? std::max(block_size, round_up_to_multiple(
                                     estimate.scheduler_seq_window *
                                         representative_prompt_tokens,
                                     block_size))
          : std::max(block_size, round_up_to_multiple(
                                     estimate.target_decode_concurrency *
                                         representative_prompt_tokens,
                                     block_size));
  estimate.scheduler_token_budget =
      std::max(estimate.scheduler_seq_window, estimate.target_decode_token_window);
  estimate.safety_token_cap = kAutoMaxBatchedTokensSafetyCap;
  estimate.prefill_safety_token_cap = kAutoPrefillChunkCapSafetyCap;

  const int32_t effective_token_budget = std::max(
      1, std::min({estimate.capacity_token_budget,
                   std::max(1, estimate.scheduler_token_budget)}));
  const int32_t block_aligned_cap =
      effective_token_budget >= block_size
          ? round_down_to_multiple(effective_token_budget, block_size)
          : effective_token_budget;
  estimate.raw_max_batched_tokens = std::max(
      1, std::min(estimate.safety_token_cap, std::max(1, block_aligned_cap)));
  estimate.raw_prefill_chunk_cap =
      std::max(1, std::min({estimate.prefill_safety_token_cap,
                            estimate.prompt_prefill_chunk_target,
                            estimate.raw_max_batched_tokens,
                            estimate.capacity_token_budget}));
  return estimate;
}

void print_step_profile(const StepProfile& profile) {
  std::cout << "STEP_PROFILE"
            << " step=" << profile.step
            << " requests=" << profile.num_requests
            << " tokens=" << profile.num_tokens
            << " decode_tokens=" << profile.num_decode_tokens
            << " prefill_tokens=" << profile.num_prefill_tokens
            << " sampled_tokens=" << profile.num_sampled_tokens
            << " waiting_queue=" << profile.waiting_queue_size
            << " running_queue=" << profile.running_queue_size
            << " preemptions=" << profile.preemptions
            << " schedule_ms=" << format_double(profile.schedule_ms)
            << " build_metadata_ms=" << format_double(profile.build_metadata_ms)
            << " forward_ms=" << format_double(profile.forward_ms)
            << " sample_ms=" << format_double(profile.sample_ms)
            << " process_outputs_ms=" << format_double(profile.process_outputs_ms)
            << " step_ms=" << format_double(profile.step_ms)
            << "\n";
}

void print_request_metric(const SequenceState& seq) {
  std::cout << "REQUEST_METRIC"
            << " client_request_id=" << seq.client_request_id
            << " status=" << (seq.failed ? "failed" : "ok")
            << " prompt_tokens=" << seq.prompt_tokens.size()
            << " generated_tokens=" << seq.generated_tokens
            << " output_tokens=" << seq.output_tokens.size()
            << " ttft_ms=" << format_double(seq.ttft_ms())
            << " itl_ms=" << format_double(seq.itl_ms())
            << " latency_ms=" << format_double(seq.latency_ms())
            << " finish_reason=" << (seq.finish_reason.empty() ? "completed" : seq.finish_reason)
            << "\n";
}

void print_config_summary(const BenchConfig& config) {
  std::cout << "CONFIG_SUMMARY"
            << " max_new_tokens=" << config.max_new_tokens
            << " max_batched_tokens_request=" << config.max_num_batched_tokens_request
            << " max_batched_tokens_resolved=" << config.max_num_batched_tokens
            << " prefill_chunk_cap_request=" << config.prefill_chunk_cap_request
            << " prefill_chunk_cap_resolved=" << config.prefill_chunk_cap
            << " scheduling_policy=" << scheduling_policy_name(config.scheduling_policy)
            << " preemption_policy=" << preemption_policy_name(config.preemption_policy)
            << " long_prefill_token_threshold=" << config.long_prefill_token_threshold
            << " max_partial_prefills=" << config.max_partial_prefills
            << " max_long_partial_prefills=" << config.max_long_partial_prefills
            << " device_id=" << config.device_id
            << " request_timeout_ms=" << config.request_timeout_ms
            << " max_prompt_tokens=" << config.max_prompt_tokens
            << " pd_mode=" << config.pd_mode
            << " online_process_role=" << config.online_process_role
            << " engine_zmq_endpoint=" << config.engine_zmq_endpoint
            << " prefill_zmq_endpoint=" << config.prefill_zmq_endpoint
            << " engine_zmq_timeout_ms=" << config.engine_zmq_timeout_ms
            << " prefill_device_id=" << config.prefill_device_id
            << " decode_device_id=" << config.decode_device_id
            << " http_worker_threads=" << config.http_worker_threads
            << " http_listen_backlog=" << config.http_listen_backlog
            << " kv_cache_memory_utilization="
            << format_double(config.kv_cache_memory_utilization, 3)
            << " kv_cache_blocks_per_layer_request="
            << config.kv_cache_blocks_per_layer
            << " warmup_rounds=" << config.warmup_rounds
            << " radix_cache_config_explicit="
            << (config.radix_cache_config_explicit ? 1 : 0)
            << " radix_cache_enabled=" << (config.radix_cache_enabled ? 1 : 0)
            << " host_cache_enabled=" << (config.host_cache_enabled ? 1 : 0)
            << " host_cache_bytes=" << config.host_cache_bytes
            << " host_cache_pages=" << config.host_cache_pages
            << " host_cache_inflight_pages=" << config.host_cache_inflight_pages
            << " host_demote_after_warmup=" << (config.host_demote_after_warmup ? 1 : 0)
            << " prompt_count=" << config.prompt_token_stats.prompt_count
            << " prompt_min_tokens=" << config.prompt_token_stats.min_prompt_tokens
            << " prompt_p50_tokens=" << config.prompt_token_stats.p50_prompt_tokens
            << " prompt_p95_tokens=" << config.prompt_token_stats.p95_prompt_tokens
            << " prompt_max_tokens=" << config.prompt_token_stats.max_prompt_tokens
            << " prompt_total_tokens=" << config.prompt_token_stats.total_prompt_tokens
            << " prompt_mean_tokens=" << format_double(config.prompt_token_stats.mean_prompt_tokens)
            << " capacity_max_batch_size=" << config.capacity_info.max_batch_size
            << " capacity_block_size=" << config.capacity_info.block_size
            << " capacity_total_kv_blocks=" << config.capacity_info.total_kv_blocks
            << " capacity_free_kv_blocks=" << config.capacity_info.free_kv_blocks
            << " capacity_layer_num=" << config.capacity_info.layer_num
            << " capacity_model_dim=" << config.capacity_info.model_dim
            << " capacity_head_num=" << config.capacity_info.head_num
            << " capacity_kv_head_num=" << config.capacity_info.kv_head_num
            << " capacity_head_size=" << config.capacity_info.head_size
            << " capacity_kv_dim=" << config.capacity_info.kv_dim
            << " capacity_hidden_dim=" << config.capacity_info.hidden_dim
            << " capacity_vocab_size=" << config.capacity_info.vocab_size
            << " capacity_device_free_memory_bytes="
            << config.capacity_info.device_free_memory_bytes
            << " capacity_device_total_memory_bytes="
            << config.capacity_info.device_total_memory_bytes
            << " capacity_kv_bytes_per_token=" << config.capacity_info.kv_bytes_per_token
            << " capacity_kv_bytes_per_block_per_layer="
            << config.capacity_info.kv_bytes_per_block_per_layer
            << " capacity_total_kv_pool_bytes="
            << config.capacity_info.total_kv_pool_bytes
            << " capacity_workspace_bytes_per_token="
            << config.capacity_info.workspace_bytes_per_token
            << " capacity_serving_workspace_token_capacity="
            << config.capacity_info.serving_workspace_token_capacity
            << " capacity_serving_workspace_reserved_bytes="
            << config.capacity_info.serving_workspace_reserved_bytes
            << " auto_usable_free_kv_blocks="
            << config.auto_estimate.usable_free_kv_blocks
            << " auto_kv_token_budget=" << config.auto_estimate.kv_token_budget
            << " auto_device_workspace_bytes_budget="
            << config.auto_estimate.device_workspace_bytes_budget
            << " auto_workspace_token_budget=" << config.auto_estimate.workspace_token_budget
            << " auto_capacity_token_budget=" << config.auto_estimate.capacity_token_budget
            << " auto_scheduler_seq_window=" << config.auto_estimate.scheduler_seq_window
            << " auto_target_decode_concurrency="
            << config.auto_estimate.target_decode_concurrency
            << " auto_prefill_reserved_seqs="
            << config.auto_estimate.prefill_reserved_seqs
            << " auto_prompt_prefill_chunk_target="
            << config.auto_estimate.prompt_prefill_chunk_target
            << " auto_scheduler_token_budget="
            << config.auto_estimate.scheduler_token_budget
            << " auto_safety_token_cap=" << config.auto_estimate.safety_token_cap
            << " auto_prefill_safety_token_cap="
            << config.auto_estimate.prefill_safety_token_cap
            << "\n";
}

void print_final_summary(const SummaryStats& stats,
                         double wall_ms,
                         double throughput_tokens_per_s,
                         const base::RadixCacheStats& radix_stats,
                         int32_t radix_cache_nodes,
                         int32_t radix_cache_splits,
                         int32_t radix_cache_evictable_blocks) {
  const double ttft_mean_ms = average(stats.request_ttft_ms);
  const double itl_mean_ms = average(stats.request_itl_ms);
  const double latency_mean_ms = average(stats.request_latency_ms);
  std::cout << "FINAL_SUMMARY"
            << " total_steps=" << stats.total_steps
            << " active_steps=" << stats.active_steps
            << " decode_only_steps=" << stats.decode_only_steps
            << " completed_requests=" << stats.completed_requests
            << " failed_requests=" << stats.failed_requests
            << " total_decode_tokens=" << stats.total_decode_tokens
            << " total_prefill_tokens=" << stats.total_prefill_tokens
            << " total_batched_tokens=" << stats.total_batched_tokens
            << " scheduler_no_progress_steps=" << stats.scheduler_no_progress_steps
            << " scheduler_preemptions=" << stats.scheduler_preemptions
            << " scheduler_priority_preemptions=" << stats.scheduler_priority_preemptions
            << " scheduler_decode_kv_preemptions=" << stats.scheduler_decode_kv_preemptions
            << " scheduler_waiting_rejections=" << stats.scheduler_waiting_rejections
            << " scheduler_stalled_prefill_failures=" << stats.scheduler_stalled_prefill_failures
            << " scheduler_cache_transfer_completions=" << stats.cache_transfer_completions
            << " scheduler_cache_restore_wait_steps=" << stats.cache_restore_wait_steps
            << " avg_waiting_queue=" << format_double(stats.active_steps > 0 ? static_cast<double>(stats.waiting_queue_samples) / stats.active_steps : 0.0)
            << " avg_running_queue=" << format_double(stats.active_steps > 0 ? static_cast<double>(stats.running_queue_samples) / stats.active_steps : 0.0)
            << " max_waiting_queue=" << stats.max_waiting_queue
            << " max_running_queue=" << stats.max_running_queue
            << " wall_ms=" << format_double(wall_ms)
            << " throughput_tps=" << format_double(throughput_tokens_per_s)
            << " avg_schedule_ms=" << format_double(stats.active_steps > 0 ? stats.total_schedule_ms / stats.active_steps : 0.0)
            << " avg_build_metadata_ms=" << format_double(stats.active_steps > 0 ? stats.total_build_metadata_ms / stats.active_steps : 0.0)
            << " avg_forward_ms=" << format_double(stats.active_steps > 0 ? stats.total_forward_ms / stats.active_steps : 0.0)
            << " avg_sample_ms=" << format_double(stats.active_steps > 0 ? stats.total_sample_ms / stats.active_steps : 0.0)
            << " avg_process_outputs_ms=" << format_double(stats.active_steps > 0 ? stats.total_process_outputs_ms / stats.active_steps : 0.0)
            << " avg_step_ms=" << format_double(stats.active_steps > 0 ? stats.total_step_ms / stats.active_steps : 0.0)
            << " ttft_ms=" << format_double(ttft_mean_ms)
            << " ttft_p50_ms=" << format_double(percentile(stats.request_ttft_ms, 0.50))
            << " ttft_p95_ms=" << format_double(percentile(stats.request_ttft_ms, 0.95))
            << " ttft_p99_ms=" << format_double(percentile(stats.request_ttft_ms, 0.99))
            << " token_gap_count=" << stats.token_gap_ms.size()
            << " token_gap_p50_ms=" << format_double(percentile(stats.token_gap_ms, 0.50))
            << " token_gap_p95_ms=" << format_double(percentile(stats.token_gap_ms, 0.95))
            << " token_gap_p99_ms=" << format_double(percentile(stats.token_gap_ms, 0.99))
            << " request_mean_token_gap_p95_ms=" << format_double(percentile(stats.request_itl_ms, 0.95))
            << " itl_ms=" << format_double(itl_mean_ms)
            << " itl_p50_ms=" << format_double(percentile(stats.request_itl_ms, 0.50))
            << " itl_p95_ms=" << format_double(percentile(stats.request_itl_ms, 0.95))
            << " itl_p99_ms=" << format_double(percentile(stats.request_itl_ms, 0.99))
            << " latency_ms=" << format_double(latency_mean_ms)
            << " latency_p50_ms=" << format_double(percentile(stats.request_latency_ms, 0.50))
            << " latency_p95_ms=" << format_double(percentile(stats.request_latency_ms, 0.95))
            << " latency_p99_ms=" << format_double(percentile(stats.request_latency_ms, 0.99))
            << " radix_cache_lookups=" << radix_stats.lookup_requests
            << " radix_cache_hits=" << radix_stats.cache_hits
            << " radix_cache_misses=" << radix_stats.cache_misses
            << " radix_cache_tokens_reused=" << radix_stats.tokens_reused
            << " radix_cache_publish_requests=" << radix_stats.publish_requests
            << " radix_cache_published_blocks=" << radix_stats.published_blocks
            << " radix_cache_evictions=" << radix_stats.evictions
            << " radix_cache_evicted_blocks=" << radix_stats.evicted_blocks
            << " host_demoted_blocks=" << radix_stats.host_demoted_blocks
            << " host_restore_requests=" << radix_stats.host_restore_requests
            << " host_restored_blocks=" << radix_stats.host_restored_blocks
            << " host_restore_failures=" << radix_stats.host_restore_failures
            << " radix_cache_nodes=" << radix_cache_nodes
            << " radix_cache_splits=" << radix_cache_splits
            << " radix_cache_evictable_blocks=" << radix_cache_evictable_blocks
            << "\n";
}

}  // namespace

int ServingBenchmarkApp::run(int argc, char* argv[]) {
  if (!parse_args(argc, argv)) {
    return -1;
  }
  if (bench_config_.online_server &&
      bench_config_.online_process_role == kOnlineProcessRoleZmqHttpApi) {
    return serving::run_online_server(nullptr, bench_config_);
  }
  if (!initialize_model(model_path_, tokenizer_path_, bench_config_)) {
    return -1;
  }

  prepare_benchmark_config();
  run_warmup();
  if (bench_config_.host_demote_after_warmup) {
    CHECK(kv_cache_manager()->host_cache_enabled())
        << "--host-demote-after-warmup requires --host-cache=1";
    int32_t admitted = 0;
    while (true) {
      const auto batch = kv_cache_manager()->demote_radix_cache_to_host();
      admitted += batch;
      CHECK(kv_cache_manager()->drain_cache_transfers())
          << "host cache warmup demotion could not establish physical completion";
      if (batch == 0) break;
    }
    LOG(INFO) << "Host cache warmup demotion admitted_pages=" << admitted
              << " host_demoted_pages="
              << kv_cache_manager()->radix_cache_stats().host_demoted_blocks;
  }
  kv_cache_manager()->reset_radix_cache_stats();
  if (bench_config_.online_server) {
    if (is_zmq_engine_core_role(bench_config_.online_process_role)) {
      auto status = serving::run_zmq_engine_core_server(this, bench_config_);
      if (!status) {
        LOG(ERROR) << status.get_err_msg();
        return -1;
      }
      return 0;
    }
    return serving::run_online_server(this);
  }
  if (is_dual_gpu_pd_mode(bench_config_.pd_mode)) {
    return run_dual_gpu_pd_offline();
  }
  create_scheduler();
  submit_all_requests();
  run_serving_loop();
  return 0;
}

int ServingBenchmarkApp::run_dual_gpu_pd_offline() {
  if (!pd_dual_gpu_supported()) {
    LOG(ERROR) << "pd-mode=" << bench_config_.pd_mode
               << " requested, but this app does not provide P/D models";
    return -1;
  }
  if (!bench_config_.quiet) {
    std::cout << "\n=== Starting " << bench_config_.pd_mode
              << " PD offline ===" << std::endl;
  }
  summary_.request_ttft_ms.reserve(prompts_.size());
  summary_.request_itl_ms.reserve(prompts_.size());
  summary_.request_latency_ms.reserve(prompts_.size());
  const auto start = Clock::now();
  for (size_t i = 0; i < prompts_.size(); ++i) {
    const auto request_start = Clock::now();
    auto tokens = encode_prompt(prompts_[i]);
    GenerationConfig config(bench_config_.max_new_tokens);
    config.sampling = bench_config_.sampling;
    config.ignore_eos = bench_config_.ignore_eos;
    PDGenerationResult generation = run_dual_gpu_pd_generation(std::move(tokens), config);
    const auto request_end = Clock::now();
    if (generation.failed) {
      ++summary_.failed_requests;
    } else {
      ++summary_.completed_requests;
      total_decode_steps_ += static_cast<int32_t>(generation.output_tokens.size());
    }
    const double latency = Duration(request_end - request_start).count();
    summary_.request_latency_ms.push_back(latency);
    std::cout << "PD_REQUEST_METRIC"
              << " client_request_id=" << i
              << " status=" << (generation.failed ? "failed" : "ok")
              << " output_tokens=" << generation.output_tokens.size()
              << " latency_ms=" << format_double(latency)
              << " finish_reason="
              << (generation.error.empty() ? "completed" : generation.error)
              << "\n";
    if (!bench_config_.quiet) {
      std::cout << "\n--- Request " << i
                << (generation.failed ? " failed" : " finished") << " ---\n";
      if (generation.failed) {
        std::cout << "reason: " << generation.error << "\n";
      } else {
        std::cout << postprocess_decoded_text(decode_tokens(generation.output_tokens))
                  << "\n";
      }
    }
  }
  const auto end = Clock::now();
  const double duration = std::chrono::duration<double>(end - start).count();
  const double wall_ms = Duration(end - start).count();
  const double throughput = duration > 0.0 ? total_decode_steps_ / duration : 0.0;
  print_done(duration, throughput);
  if (bench_config_.print_final_summary) {
    const auto* kv_manager = pd_decode_kv_cache_manager();
    print_final_summary(summary_, wall_ms, throughput,
                        kv_manager->radix_cache_stats(),
                        kv_manager->radix_cache_node_count(),
                        kv_manager->radix_cache_split_count(),
                        kv_manager->radix_cache_evictable_blocks());
  }
  return summary_.failed_requests == 0 ? 0 : -1;
}

std::vector<std::string> ServingBenchmarkApp::default_prompts() const {
  return {"What is AI?", "Write a haiku about coding.",
          "Explain quantum computing briefly."};
}

bool ServingBenchmarkApp::parse_args(int argc, char* argv[]) {
  bool api_only_short_form = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--online-process-role=zmq-http-api") {
      api_only_short_form = true;
      break;
    }
  }
  if (argc < 3 && !api_only_short_form) {
    LOG(INFO) << "Usage: " << usage_name()
              << " <model.bin> <tokenizer.json> [prompt1] [prompt2] ..."
              << " [--max-new-tokens=N] [--max-batched-tokens=N]"
              << " [--temperature=0.0] [--top-k=0] [--top-p=1.0] [--seed=N] [--ignore-eos=0|1]"
              << " [--prefill-chunk-cap=N] [--scheduling-policy=fcfs|priority]"
              << " [--preemption-policy=recompute|checkpoint|auto]"
              << " [--long-prefill-token-threshold=N] [--max-partial-prefills=N]"
              << " [--max-long-partial-prefills=N]"
              << " [--device-id=N] [--kv-cache-memory-utilization=0.8]"
              << " [--kv-cache-blocks-per-layer=N] [--quiet=0|1]"
              << " [--pd-mode=off|dual-gpu-p2p|dual-gpu-nccl|dual-gpu-nccl-layer|remote-zmq-cpu|remote-zmq-nccl|remote-zmq-nccl-layer]"
              << " [--prefill-device-id=N]"
              << " [--decode-device-id=N]"
              << " [--warmup-rounds=N]"
              << " [--step-profile=0|1] [--step-trace=0|1] [--final-summary=0|1]"
              << " [--online-server=0|1] [--listen-host=127.0.0.1]"
              << " [--listen-port=8080] [--http-worker-threads=N]"
              << " [--http-listen-backlog=N] [--max-queue-size=N]"
              << " [--request-timeout-ms=N] [--max-prompt-tokens=N]"
              << " [--online-process-role=inproc|zmq-http-api|zmq-engine-core|zmq-prefill-engine-core|zmq-decode-engine-core]"
              << " [--engine-zmq-endpoint=tcp://127.0.0.1:19090]"
              << " [--prefill-zmq-endpoint=tcp://127.0.0.1:19091]"
              << " [--engine-zmq-timeout-ms=N]"
              << " [--data-service-endpoint=/path/to/pbe-data.sock]";
    return false;
  }

  if (api_only_short_form) {
    model_path_.clear();
    tokenizer_path_.clear();
    bench_config_ = parse_bench_config(argc, argv, 1);
    bench_config_.online_server = true;
  } else {
    model_path_ = argv[1];
    tokenizer_path_ = argv[2];
    bench_config_ = parse_bench_config(argc, argv, 3);
  }
  collect_prompts(argc, argv);
  return true;
}

void ServingBenchmarkApp::collect_prompts(int argc, char* argv[]) {
  prompts_.clear();
  if (argc > 3) {
    for (int i = 3; i < argc; ++i) {
      if (is_cli_flag(argv[i])) {
        continue;
      }
      prompts_.emplace_back(argv[i]);
    }
  }
  if (prompts_.empty()) {
    prompts_ = default_prompts();
  }
}

void ServingBenchmarkApp::prepare_benchmark_config() {
  PromptTokenStats stats;
  if (!prompts_.empty()) {
    std::vector<double> token_counts;
    token_counts.reserve(prompts_.size());
    for (const auto& prompt : prompts_) {
      const int32_t token_count = static_cast<int32_t>(encode_prompt(prompt).size());
      stats.total_prompt_tokens += token_count;
      token_counts.push_back(static_cast<double>(token_count));
    }

    std::sort(token_counts.begin(), token_counts.end());
    stats.prompt_count = static_cast<int32_t>(token_counts.size());
    stats.min_prompt_tokens = static_cast<int32_t>(token_counts.front());
    stats.p50_prompt_tokens = static_cast<int32_t>(std::round(percentile(token_counts, 0.50)));
    stats.p95_prompt_tokens = static_cast<int32_t>(std::round(percentile(token_counts, 0.95)));
    stats.max_prompt_tokens = static_cast<int32_t>(token_counts.back());
    stats.mean_prompt_tokens =
        static_cast<double>(stats.total_prompt_tokens) / static_cast<double>(stats.prompt_count);
  }
  bench_config_.prompt_token_stats = stats;
  bench_config_.capacity_info = serving_capacity_info();
  bench_config_.auto_estimate =
      estimate_auto_schedule(bench_config_.capacity_info,
                             bench_config_.prompt_token_stats,
                             bench_config_.max_new_tokens,
                             bench_config_.kv_cache_memory_utilization);

  if (bench_config_.auto_max_num_batched_tokens) {
    bench_config_.max_num_batched_tokens = bench_config_.auto_estimate.raw_max_batched_tokens;
  } else if (bench_config_.auto_prefill_chunk_cap) {
    bench_config_.max_num_batched_tokens = std::min(
        bench_config_.max_num_batched_tokens,
        std::max(bench_config_.prefill_chunk_cap,
                 bench_config_.auto_estimate.raw_max_batched_tokens));
  } else {
    bench_config_.max_num_batched_tokens = std::min(
        bench_config_.max_num_batched_tokens,
        std::max(1, bench_config_.capacity_info.max_batch_size *
                        bench_config_.prefill_chunk_cap));
  }
  // Explicit CLI values are ceilings too: another resident model can cause
  // init_mem to allocate a smaller workspace than the requested batch size.
  if (bench_config_.capacity_info.serving_workspace_token_capacity > 0) {
    bench_config_.max_num_batched_tokens = std::min(
        bench_config_.max_num_batched_tokens,
        bench_config_.capacity_info.serving_workspace_token_capacity);
  }
  if (bench_config_.auto_prefill_chunk_cap) {
    bench_config_.prefill_chunk_cap =
        std::min(bench_config_.auto_estimate.raw_prefill_chunk_cap,
                 bench_config_.max_num_batched_tokens);
  }
  bench_config_.max_num_batched_tokens = std::max(1, bench_config_.max_num_batched_tokens);
  bench_config_.prefill_chunk_cap =
      std::max(1, std::min(bench_config_.prefill_chunk_cap,
                           bench_config_.max_num_batched_tokens));

  if (bench_config_.print_final_summary) {
    print_config_summary(bench_config_);
  }
}

void ServingBenchmarkApp::run_warmup() {
  if (bench_config_.warmup_rounds <= 0 || prompts_.empty()) {
    return;
  }

  SchedulerConfig sched_config;
  sched_config.max_num_seqs = max_model_batch_size();
  sched_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  sched_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  sched_config.policy = bench_config_.scheduling_policy;
  sched_config.preemption_policy = bench_config_.preemption_policy;
  sched_config.long_prefill_token_threshold = bench_config_.long_prefill_token_threshold;
  sched_config.max_partial_prefills = bench_config_.max_partial_prefills;
  sched_config.max_long_partial_prefills = bench_config_.max_long_partial_prefills;

  void* stream = model_stream();
  for (int32_t round = 0; round < bench_config_.warmup_rounds; ++round) {
    Scheduler warmup_scheduler(sched_config, kv_cache_manager());
    submit_requests_to(warmup_scheduler, 1, true);
    warmup_scheduler.reset_request_arrival_times();

    while (warmup_scheduler.has_active_requests()) {
      auto sched_out = warmup_scheduler.schedule_step();
      if (sched_out.total_tokens == 0) {
        continue;
      }

      const bool decode_only_step =
          sched_out.num_decode_seqs > 0 && sched_out.num_prefill_seqs == 0;
      auto batch = decode_only_step ? warmup_scheduler.build_decode_batch(sched_out, stream)
                                    : warmup_scheduler.build_mixed_batch(sched_out, stream);
      auto status = decode_only_step ? forward_decode_batch(batch)
                                     : forward_mixed_batch(batch);
      CHECK(status) << (decode_only_step ? "warmup forward_decode_batch failed: "
                                         : "warmup forward_mixed_batch failed: ")
                    << status.get_err_msg();

      auto sampled_tokens = batch_sample(batch, sched_out);
      warmup_scheduler.process_outputs(
          sched_out, batch, sampled_tokens,
          [&](int32_t token) { return is_sentence_ending(token); });
      auto finished = warmup_scheduler.pop_finished();
      (void)finished;
    }
  }
}

void ServingBenchmarkApp::create_scheduler() {
  SchedulerConfig sched_config;
  sched_config.max_num_seqs = max_model_batch_size();
  sched_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  sched_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  sched_config.policy = bench_config_.scheduling_policy;
  sched_config.preemption_policy = bench_config_.preemption_policy;
  sched_config.long_prefill_token_threshold = bench_config_.long_prefill_token_threshold;
  sched_config.max_partial_prefills = bench_config_.max_partial_prefills;
  sched_config.max_long_partial_prefills = bench_config_.max_long_partial_prefills;

  scheduler_ = std::make_unique<Scheduler>(sched_config, kv_cache_manager());
}

void ServingBenchmarkApp::submit_all_requests() {
  submit_requests_to(*scheduler_, bench_config_.max_new_tokens, bench_config_.quiet);
}

void ServingBenchmarkApp::submit_requests_to(Scheduler& scheduler,
                                             int32_t max_new_tokens,
                                             bool quiet) const {
  if (!quiet) {
    std::cout << "=== Submitting " << prompts_.size() << " requests ===" << std::endl;
  }
  for (size_t i = 0; i < prompts_.size(); ++i) {
    auto tokens = encode_prompt(prompts_[i]);
    if (!quiet) {
      std::cout << "Request " << i << ": \"" << prompts_[i]
                << "\" (" << tokens.size() << " tokens)" << std::endl;
    }
    GenerationConfig generation(max_new_tokens);
    generation.sampling = bench_config_.sampling;
    generation.ignore_eos = bench_config_.ignore_eos;
    scheduler.add_request(std::move(tokens), generation);
  }
}

void ServingBenchmarkApp::run_serving_loop() {
  if (!bench_config_.quiet) {
    std::cout << "\n=== Starting continuous batching ===" << std::endl;
  }

  scheduler_->reset_request_arrival_times();
  const auto start = Clock::now();
  void* stream = model_stream();
  summary_.request_ttft_ms.reserve(prompts_.size());
  summary_.request_itl_ms.reserve(prompts_.size());
  summary_.request_latency_ms.reserve(prompts_.size());

  while (scheduler_->has_active_requests()) {
    run_serving_step(stream);
  }

  const auto end = Clock::now();
  const double duration = std::chrono::duration<double>(end - start).count();
  const double wall_ms = Duration(end - start).count();
  const double throughput = duration > 0.0 ? total_decode_steps_ / duration : 0.0;

  print_done(duration, throughput);
  if (bench_config_.print_final_summary) {
    const auto* kv_manager = kv_cache_manager();
    print_final_summary(summary_, wall_ms, throughput,
                        kv_manager->radix_cache_stats(),
                        kv_manager->radix_cache_node_count(),
                        kv_manager->radix_cache_split_count(),
                        kv_manager->radix_cache_evictable_blocks());
  }
}

void ServingBenchmarkApp::run_serving_step(void* stream) {
  const std::string step_name = "step " + std::to_string(step_);
  base::nvtx::ScopedRange step_range(step_name, base::nvtx::kColorStep);
  const auto step_start = Clock::now();

  const auto schedule_start = Clock::now();
  SchedulerOutput sched_out;
  {
    base::nvtx::ScopedRange range("schedule", base::nvtx::kColorSchedule);
    sched_out = scheduler_->schedule_step();
  }
  const auto schedule_end = Clock::now();
  if (sched_out.total_tokens == 0) {
    record_no_progress_step(sched_out);
    if (kv_cache_manager()->cache_transfers_pending()) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ++step_;
    return;
  }

  const auto build_start = Clock::now();
  const bool decode_only_step =
      sched_out.num_decode_seqs > 0 && sched_out.num_prefill_seqs == 0;
  MixedBatchMetadata batch;
  {
    const char* build_name = decode_only_step ? "build_decode_batch" : "build_mixed_batch";
    base::nvtx::ScopedRange range(build_name, base::nvtx::kColorMetadata);
    batch = decode_only_step ? scheduler_->build_decode_batch(sched_out, stream)
                             : scheduler_->build_mixed_batch(sched_out, stream);
  }
  const auto build_end = Clock::now();

  const auto forward_start = Clock::now();
  base::Status status;
  {
    const char* forward_name = decode_only_step ? "forward_decode_batch"
                                                : "forward_mixed_batch";
    base::nvtx::ScopedRange range(forward_name, base::nvtx::kColorForward);
    status = decode_only_step ? forward_decode_batch(batch)
                              : forward_mixed_batch(batch);
  }
  const auto forward_end = Clock::now();
  CHECK(status) << (decode_only_step ? "forward_decode_batch failed: "
                                     : "forward_mixed_batch failed: ")
                << status.get_err_msg();

  const auto sample_start = Clock::now();
  auto sampled_tokens = [&]() {
    base::nvtx::ScopedRange range("batch_sample", base::nvtx::kColorSample);
    return batch_sample(batch, sched_out);
  }();
  const auto sample_end = Clock::now();
  total_decode_steps_ += static_cast<int32_t>(sampled_tokens.size());

  const auto process_start = Clock::now();
  {
    base::nvtx::ScopedRange range("process_outputs", base::nvtx::kColorProcess);
    scheduler_->process_outputs(
        sched_out, batch, sampled_tokens,
        [&](int32_t token) { return is_sentence_ending(token); });
  }
  const auto process_end = Clock::now();

  auto finished = scheduler_->pop_finished();
  process_finished(finished);

  const auto step_end = Clock::now();
  StepProfile profile = build_step_profile(
      sched_out, batch, sampled_tokens.size(), schedule_start, schedule_end, build_start, build_end,
      forward_start, forward_end, sample_start, sample_end, process_start, process_end,
      step_start, step_end);
  record_step_profile(profile, decode_only_step, static_cast<int32_t>(finished.size()));
  maybe_print_step_profile(profile, batch, sampled_tokens.size());

  ++step_;
}

StepProfile ServingBenchmarkApp::build_step_profile(
    const SchedulerOutput& sched_out,
    const MixedBatchMetadata& batch,
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
    Clock::time_point step_end) const {
  StepProfile profile;
  profile.step = step_;
  profile.num_requests = batch.num_requests;
  profile.num_tokens = batch.num_tokens;
  profile.num_decode_tokens = batch.num_decode_tokens;
  profile.num_prefill_tokens = batch.num_prefill_tokens;
  profile.num_sampled_tokens = static_cast<int32_t>(sampled_token_count);
  profile.waiting_queue_size = sched_out.waiting_queue_size;
  profile.running_queue_size = sched_out.running_queue_size;
  profile.preemptions = sched_out.preemptions;
  profile.schedule_ms = Duration(schedule_end - schedule_start).count();
  profile.build_metadata_ms = Duration(build_end - build_start).count();
  profile.forward_ms = Duration(forward_end - forward_start).count();
  profile.sample_ms = Duration(sample_end - sample_start).count();
  profile.process_outputs_ms = Duration(process_end - process_start).count();
  profile.step_ms = Duration(step_end - step_start).count();
  return profile;
}

void ServingBenchmarkApp::process_finished(const std::vector<SequenceState>& finished) {
  for (const auto& seq : finished) {
    if (seq.failed) {
      summary_.failed_requests++;
    } else {
      summary_.completed_requests++;
    }
    summary_.request_ttft_ms.push_back(seq.ttft_ms());
    if (seq.generated_tokens > 1) {
      summary_.request_itl_ms.push_back(seq.itl_ms());
    }
    const auto gaps = seq.token_gaps_ms();
    summary_.token_gap_ms.insert(summary_.token_gap_ms.end(), gaps.begin(), gaps.end());
    summary_.request_latency_ms.push_back(seq.latency_ms());
    print_request_metric(seq);

    if (!bench_config_.quiet) {
      std::string text = postprocess_decoded_text(decode_tokens(seq.output_tokens));
      std::cout << "\n--- Request " << seq.client_request_id
                << (seq.failed ? " failed" : " finished") << " ---\n";
      if (seq.failed) {
        std::cout << "reason: " << seq.finish_reason << "\n";
      } else {
        std::cout << text << "\n";
      }
    }
  }
}

void ServingBenchmarkApp::record_no_progress_step(const SchedulerOutput& sched_out) {
  summary_.total_steps++;
  summary_.waiting_queue_samples += sched_out.waiting_queue_size;
  summary_.running_queue_samples += sched_out.running_queue_size;
  summary_.max_waiting_queue = std::max(summary_.max_waiting_queue,
                                        sched_out.waiting_queue_size);
  summary_.max_running_queue = std::max(summary_.max_running_queue,
                                        sched_out.running_queue_size);
  const auto& scheduler_metrics = scheduler_->metrics();
  summary_.scheduler_no_progress_steps = scheduler_metrics.no_progress_steps;
  summary_.scheduler_preemptions = scheduler_metrics.preemptions;
  summary_.scheduler_priority_preemptions = scheduler_metrics.priority_preemptions;
  summary_.scheduler_decode_kv_preemptions = scheduler_metrics.decode_kv_preemptions;
  summary_.scheduler_waiting_rejections = scheduler_metrics.waiting_rejections;
  summary_.scheduler_stalled_prefill_failures = scheduler_metrics.stalled_prefill_failures;
  summary_.cache_transfer_completions = scheduler_metrics.cache_transfer_completions;
  summary_.cache_restore_wait_steps = scheduler_metrics.cache_restore_wait_steps;
}

void ServingBenchmarkApp::record_step_profile(const StepProfile& profile,
                                              bool decode_only_step,
                                              int32_t /*finished_count*/) {
  summary_.total_steps++;
  summary_.active_steps++;
  summary_.decode_only_steps += decode_only_step ? 1 : 0;
  summary_.total_decode_tokens += profile.num_sampled_tokens;
  summary_.total_prefill_tokens += profile.num_prefill_tokens;
  summary_.total_batched_tokens += profile.num_tokens;
  summary_.total_schedule_ms += profile.schedule_ms;
  summary_.total_build_metadata_ms += profile.build_metadata_ms;
  summary_.total_forward_ms += profile.forward_ms;
  summary_.total_sample_ms += profile.sample_ms;
  summary_.total_process_outputs_ms += profile.process_outputs_ms;
  summary_.total_step_ms += profile.step_ms;
  summary_.waiting_queue_samples += profile.waiting_queue_size;
  summary_.running_queue_samples += profile.running_queue_size;
  summary_.max_waiting_queue = std::max(summary_.max_waiting_queue, profile.waiting_queue_size);
  summary_.max_running_queue = std::max(summary_.max_running_queue, profile.running_queue_size);
  const auto& scheduler_metrics = scheduler_->metrics();
  summary_.scheduler_no_progress_steps = scheduler_metrics.no_progress_steps;
  summary_.scheduler_preemptions = scheduler_metrics.preemptions;
  summary_.scheduler_priority_preemptions = scheduler_metrics.priority_preemptions;
  summary_.scheduler_decode_kv_preemptions = scheduler_metrics.decode_kv_preemptions;
  summary_.scheduler_waiting_rejections = scheduler_metrics.waiting_rejections;
  summary_.scheduler_stalled_prefill_failures = scheduler_metrics.stalled_prefill_failures;
  summary_.cache_transfer_completions = scheduler_metrics.cache_transfer_completions;
  summary_.cache_restore_wait_steps = scheduler_metrics.cache_restore_wait_steps;
}

void ServingBenchmarkApp::maybe_print_step_profile(const StepProfile& profile,
                                                   const MixedBatchMetadata& batch,
                                                   size_t sampled_token_count) const {
  if (bench_config_.print_step_profile) {
    print_step_profile(profile);
  }

  if (bench_config_.print_step_trace && !bench_config_.quiet) {
    std::cout << "[step " << step_ << "]"
              << " requests=" << batch.num_requests
              << " tokens=" << batch.num_tokens
              << " decode=" << batch.num_decode_tokens
              << " prefill=" << batch.num_prefill_tokens
              << " sampled=" << sampled_token_count
              << " waiting_queue=" << profile.waiting_queue_size
              << " running_queue=" << profile.running_queue_size
              << " preemptions=" << profile.preemptions
              << " schedule_ms=" << format_double(profile.schedule_ms)
              << " build_ms=" << format_double(profile.build_metadata_ms)
              << " forward_ms=" << format_double(profile.forward_ms)
              << " sample_ms=" << format_double(profile.sample_ms)
              << " process_ms=" << format_double(profile.process_outputs_ms)
              << "\n";
  }
}

void ServingBenchmarkApp::print_done(double duration, double throughput) const {
  if (bench_config_.quiet) {
    return;
  }
  std::cout << "\n=== Done ===" << std::endl;
  std::cout << "Steps: " << step_ << std::endl;
  std::cout << "Total decode tokens: " << total_decode_steps_ << std::endl;
  std::cout << "Duration: " << duration << "s" << std::endl;
  std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
}

}  // namespace serving
