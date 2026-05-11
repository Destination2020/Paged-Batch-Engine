#include "serving/serving_benchmark_app.h"
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

bool parse_bool_flag(std::string_view value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

bool pd_layer_diag_enabled() {
  const char* value = std::getenv("KUIPER_PD_LAYER_DIAG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void pd_layer_diag(const std::string& message) {
  if (pd_layer_diag_enabled()) {
    std::cerr << "PD_LAYER_DIAG " << message << std::endl;
  }
}

std::string scheduling_policy_name(SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::kFCFS: return "fcfs";
    case SchedulingPolicy::kPriority: return "priority";
  }
  return "unknown";
}

bool parse_scheduling_policy(std::string_view value, SchedulingPolicy* policy) {
  CHECK_NE(policy, nullptr);
  if (value == "fcfs" || value == "FCFS") {
    *policy = SchedulingPolicy::kFCFS;
    return true;
  }
  if (value == "priority" || value == "Priority" || value == "PRIORITY") {
    *policy = SchedulingPolicy::kPriority;
    return true;
  }
  return false;
}

bool parse_toggle_flag(std::string_view value, bool* out) {
  CHECK_NE(out, nullptr);
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
      value == "on" || value == "enable" || value == "enabled") {
    *out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "no" ||
      value == "off" || value == "disable" || value == "disabled") {
    *out = false;
    return true;
  }
  return false;
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool is_cli_flag(std::string_view arg) {
  return starts_with(arg, "--");
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

BenchConfig parse_bench_config(int argc, char* argv[], int prompt_start_index) {
  BenchConfig config;
  for (int i = prompt_start_index; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (!starts_with(arg, "--")) {
      continue;
    }

    auto read_int = [&](std::string_view prefix, int32_t& out) {
      if (!starts_with(arg, prefix)) {
        return false;
      }
      const std::string value(arg.substr(prefix.size()));
      out = std::stoi(value);
      return true;
    };

    auto read_auto_int = [&](std::string_view prefix, int32_t& out, std::string& requested,
                             bool& is_auto, int32_t auto_value) {
      if (!starts_with(arg, prefix)) {
        return false;
      }
      const std::string value(arg.substr(prefix.size()));
      requested = value;
      if (value == "auto" || value == "AUTO") {
        is_auto = true;
        out = auto_value;
      } else {
        is_auto = false;
        out = std::stoi(value);
      }
      return true;
    };

    auto read_double = [&](std::string_view prefix, double& out) {
      if (!starts_with(arg, prefix)) {
        return false;
      }
      const std::string value(arg.substr(prefix.size()));
      out = std::stod(value);
      return true;
    };

    auto read_bool = [&](std::string_view prefix, bool& out) {
      if (!starts_with(arg, prefix)) {
        return false;
      }
      out = parse_bool_flag(arg.substr(prefix.size()));
      return true;
    };

    auto read_toggle = [&](std::string_view prefix, bool& out, bool& seen) {
      if (!starts_with(arg, prefix)) {
        return false;
      }
      const std::string_view value = arg.substr(prefix.size());
      CHECK(parse_toggle_flag(value, &out))
          << "Invalid value for " << prefix << value
          << ". Expected one of: on/off, 1/0, true/false.";
      seen = true;
      return true;
    };

    if (read_int("--max-new-tokens=", config.max_new_tokens) ||
        read_auto_int("--max-batched-tokens=", config.max_num_batched_tokens,
                      config.max_num_batched_tokens_request,
                      config.auto_max_num_batched_tokens,
                      kAutoMaxBatchedTokensSafetyCap) ||
        read_auto_int("--prefill-chunk-cap=", config.prefill_chunk_cap,
                      config.prefill_chunk_cap_request,
                      config.auto_prefill_chunk_cap,
                      kAutoPrefillChunkCapSafetyCap) ||
        read_int("--long-prefill-token-threshold=", config.long_prefill_token_threshold) ||
        read_int("--max-partial-prefills=", config.max_partial_prefills) ||
        read_int("--max-long-partial-prefills=", config.max_long_partial_prefills) ||
        read_int("--warmup-rounds=", config.warmup_rounds) ||
        read_double("--kv-cache-memory-utilization=", config.kv_cache_memory_utilization) ||
        read_double("--gpu-memory-utilization=", config.kv_cache_memory_utilization) ||
        read_toggle("--radix-cache=", config.radix_cache_enabled,
                    config.radix_cache_config_explicit) ||
        read_toggle("--enable-radix-cache=", config.radix_cache_enabled,
                    config.radix_cache_config_explicit) ||
        read_bool("--quiet=", config.quiet) ||
        read_bool("--step-profile=", config.print_step_profile) ||
        read_bool("--step-trace=", config.print_step_trace) ||
        read_bool("--final-summary=", config.print_final_summary) ||
        read_bool("--online-server=", config.online_server) ||
        read_int("--listen-port=", config.listen_port) ||
        read_int("--http-worker-threads=", config.http_worker_threads) ||
        read_int("--http-listen-backlog=", config.http_listen_backlog) ||
        read_int("--max-queue-size=", config.max_queue_size) ||
        read_int("--request-timeout-ms=", config.request_timeout_ms) ||
        read_int("--max-prompt-tokens=", config.max_prompt_tokens) ||
        read_int("--device-id=", config.device_id) ||
        read_int("--prefill-device-id=", config.prefill_device_id) ||
        read_int("--decode-device-id=", config.decode_device_id) ||
        read_int("--engine-zmq-timeout-ms=", config.engine_zmq_timeout_ms)) {
      continue;
    }
    const std::string_view online_process_role_prefix = "--online-process-role=";
    if (starts_with(arg, online_process_role_prefix)) {
      config.online_process_role =
          std::string(arg.substr(online_process_role_prefix.size()));
      continue;
    }
    const std::string_view engine_zmq_endpoint_prefix = "--engine-zmq-endpoint=";
    if (starts_with(arg, engine_zmq_endpoint_prefix)) {
      config.engine_zmq_endpoint =
          std::string(arg.substr(engine_zmq_endpoint_prefix.size()));
      continue;
    }
    const std::string_view prefill_zmq_endpoint_prefix = "--prefill-zmq-endpoint=";
    if (starts_with(arg, prefill_zmq_endpoint_prefix)) {
      config.prefill_zmq_endpoint =
          std::string(arg.substr(prefill_zmq_endpoint_prefix.size()));
      continue;
    }
    const std::string_view pd_mode_prefix = "--pd-mode=";
    if (starts_with(arg, pd_mode_prefix)) {
      config.pd_mode = std::string(arg.substr(pd_mode_prefix.size()));
      continue;
    }
    const std::string_view listen_host_prefix = "--listen-host=";
    if (starts_with(arg, listen_host_prefix)) {
      config.listen_host = std::string(arg.substr(listen_host_prefix.size()));
      continue;
    }
    const std::string_view scheduling_policy_prefix = "--scheduling-policy=";
    if (starts_with(arg, scheduling_policy_prefix)) {
      const std::string_view value = arg.substr(scheduling_policy_prefix.size());
      CHECK(parse_scheduling_policy(value, &config.scheduling_policy))
          << "Invalid --scheduling-policy=" << value << ". Expected fcfs or priority.";
      continue;
    }
  }
  config.max_new_tokens = std::max(1, config.max_new_tokens);
  config.max_num_batched_tokens = std::max(1, config.max_num_batched_tokens);
  config.prefill_chunk_cap = std::max(1, config.prefill_chunk_cap);
  config.long_prefill_token_threshold = std::max(0, config.long_prefill_token_threshold);
  config.max_partial_prefills = std::max(0, config.max_partial_prefills);
  config.max_long_partial_prefills = std::max(0, config.max_long_partial_prefills);
  config.warmup_rounds = std::max(0, config.warmup_rounds);
  config.http_worker_threads = std::max(0, config.http_worker_threads);
  config.http_listen_backlog = std::max(1, config.http_listen_backlog);
  config.kv_cache_memory_utilization =
      std::min(1.0, std::max(0.01, config.kv_cache_memory_utilization));
  config.listen_port = std::max(1, std::min(65535, config.listen_port));
  config.max_queue_size = std::max(1, config.max_queue_size);
  config.request_timeout_ms = std::max(0, config.request_timeout_ms);
  config.max_prompt_tokens = std::max(0, config.max_prompt_tokens);
  config.device_id = std::max(0, config.device_id);
  config.prefill_device_id = std::max(0, config.prefill_device_id);
  config.decode_device_id = std::max(0, config.decode_device_id);
  config.engine_zmq_timeout_ms = std::max(1, config.engine_zmq_timeout_ms);
  if (config.online_process_role.empty()) {
    config.online_process_role = kOnlineProcessRoleInProc;
  }
  CHECK(config.online_process_role == kOnlineProcessRoleInProc ||
        config.online_process_role == kOnlineProcessRoleZmqHttpApi ||
        config.online_process_role == kOnlineProcessRoleZmqEngineCore ||
        config.online_process_role == kOnlineProcessRoleZmqPrefillEngineCore ||
        config.online_process_role == kOnlineProcessRoleZmqDecodeEngineCore)
      << "Invalid --online-process-role=" << config.online_process_role
      << ". Expected inproc, zmq-http-api, zmq-engine-core, "
      << "zmq-prefill-engine-core, or zmq-decode-engine-core.";
  if (config.pd_mode.empty()) {
    config.pd_mode = "off";
  }
  if (config.pd_mode == "off") {
    config.prefill_device_id = config.device_id;
    config.decode_device_id = config.device_id;
  }
  return config;
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

base::Status copy_bytes_to_host(const void* src,
                                size_t byte_size,
                                base::DeviceType device_type,
                                void* stream,
                                std::string* dst) {
  CHECK_NE(dst, nullptr);
  dst->assign(byte_size, '\0');
  if (byte_size == 0) {
    return base::error::Success();
  }
  if (device_type == base::DeviceType::kDeviceCUDA) {
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    allocator->memcpy(src, dst->data(), byte_size, base::MemcpyKind::kMemcpyCUDA2CPU,
                      stream, true);
  } else {
    std::memcpy(dst->data(), src, byte_size);
  }
  return base::error::Success();
}

base::Status copy_bytes_from_host(const std::string& src,
                                  void* dst,
                                  size_t expected_byte_size,
                                  base::DeviceType device_type,
                                  void* stream) {
  if (src.size() != expected_byte_size) {
    return base::error::InvalidArgument(
        "remote kv payload byte size mismatch(expected=" +
        std::to_string(expected_byte_size) + ", actual=" +
        std::to_string(src.size()) + ")");
  }
  if (expected_byte_size == 0) {
    return base::error::Success();
  }
  if (device_type == base::DeviceType::kDeviceCUDA) {
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    allocator->memcpy(src.data(), dst, expected_byte_size,
                      base::MemcpyKind::kMemcpyCPU2CUDA, stream, true);
  } else {
    std::memcpy(dst, src.data(), expected_byte_size);
  }
  return base::error::Success();
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
            << " warmup_rounds=" << config.warmup_rounds
            << " radix_cache_config_explicit="
            << (config.radix_cache_config_explicit ? 1 : 0)
            << " radix_cache_enabled=" << (config.radix_cache_enabled ? 1 : 0)
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

KVPoolDescriptor ServingBenchmarkApp::pd_prefill_kv_pool() const {
  const ServingCapacityInfo info = pd_prefill_serving_capacity_info();
  KVPoolDescriptor pool;
  pool.device_id = bench_config_.prefill_device_id;
  pool.layer_num = info.layer_num;
  pool.block_size = info.block_size;
  pool.kv_head_num = info.kv_head_num;
  pool.head_size = info.head_size;
  pool.dtype = info.runtime_data_type;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

KVPoolDescriptor ServingBenchmarkApp::pd_decode_kv_pool() const {
  const ServingCapacityInfo info = pd_decode_serving_capacity_info();
  KVPoolDescriptor pool;
  pool.device_id = bench_config_.decode_device_id;
  pool.layer_num = info.layer_num;
  pool.block_size = info.block_size;
  pool.kv_head_num = info.kv_head_num;
  pool.head_size = info.head_size;
  pool.dtype = info.runtime_data_type;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

ServingBenchmarkApp::PDGenerationResult ServingBenchmarkApp::run_dual_gpu_pd_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  return run_dual_gpu_pd_generation_with_mode(
      bench_config_.pd_mode, std::move(prompt_tokens), std::move(generation_config),
      on_token);
}

RemotePrefillResult ServingBenchmarkApp::run_remote_prefill_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config) const {
  RemotePrefillResult result;
  const bool nccl_handoff = bench_config_.pd_mode == "remote-zmq-nccl";
  result.prompt_tokens = prompt_tokens;
  generation_config.normalize();
  if (prompt_tokens.empty()) {
    result.failed = true;
    result.error = "empty_prompt";
    return result;
  }

  SchedulerConfig prefill_config;
  prefill_config.max_num_seqs = max_model_batch_size();
  prefill_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  prefill_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  prefill_config.policy = bench_config_.scheduling_policy;
  prefill_config.long_prefill_token_threshold =
      bench_config_.long_prefill_token_threshold;
  prefill_config.max_partial_prefills = bench_config_.max_partial_prefills;
  prefill_config.max_long_partial_prefills =
      bench_config_.max_long_partial_prefills;

  Scheduler prefill_scheduler(prefill_config, kv_cache_manager());
  const int64_t client_id =
      prefill_scheduler.add_request(prompt_tokens, generation_config);

  base::RequestId src_request_id = -1;
  int32_t first_token = -1;
  bool completed_prefill = false;
  while (prefill_scheduler.has_active_requests() && !completed_prefill) {
    SchedulerOutput output = prefill_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch =
        prefill_scheduler.build_mixed_batch(output, model_stream());
    base::Status status = forward_mixed_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      return result;
    }
    SampledTokenView sampled = batch_sample(batch, output);
    int32_t sample_idx = 0;
    for (size_t i = 0; i < output.scheduled_seqs.size() && !completed_prefill; ++i) {
      auto* seq = output.scheduled_seqs[i];
      if (seq == nullptr) {
        continue;
      }
      const bool is_prefill_row = i >= static_cast<size_t>(output.num_decode_seqs);
      const bool prompt_finished =
          is_prefill_row &&
          seq->computed_tokens + seq->scheduled_tokens ==
              static_cast<int32_t>(prompt_tokens.size());
      if (prompt_finished) {
        if (sample_idx >= sampled.size()) {
          result.failed = true;
          result.error = "remote_prefill_missing_first_token_sample";
          return result;
        }
        if (seq->client_request_id == client_id) {
          src_request_id = seq->request_id;
          first_token = sampled.tokens[sample_idx];
          completed_prefill = true;
        }
        ++sample_idx;
      }
    }
    if (completed_prefill) {
      break;
    }
    prefill_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return is_sentence_ending(token); });
    auto finished_now = prefill_scheduler.pop_finished();
    for (const auto& seq : finished_now) {
      if (seq.client_request_id != client_id) {
        continue;
      }
      result.output_tokens = seq.output_tokens;
      result.failed = seq.failed;
      result.error = seq.finish_reason;
      if (!result.output_tokens.empty()) {
        result.first_tokens = {result.output_tokens.front()};
        result.first_token = result.output_tokens.front();
      }
      return result;
    }
  }

  if (!completed_prefill || src_request_id < 0 || first_token < 0) {
    result.failed = true;
    result.error = "remote_prefill_did_not_produce_first_token";
    return result;
  }

  result.computed_tokens = static_cast<int32_t>(prompt_tokens.size());
  result.first_token = first_token;
  result.first_tokens = {first_token};
  result.src_pool = pd_prefill_kv_pool();
  result.layers.reserve(result.src_pool.layer_num);
  result.client_request_id.value =
      "remote-zmq-" + std::to_string(
                         static_cast<int64_t>(
                             std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count()));
  result.handoff_id.value =
      static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
  if (!result.handoff_id.valid()) {
    result.handoff_id.value = 1;
  }

  const int32_t required_blocks =
      (result.computed_tokens + result.src_pool.block_size - 1) /
      result.src_pool.block_size;
  for (int32_t layer_idx = 0; layer_idx < result.src_pool.layer_num; ++layer_idx) {
    RemoteKVLayerPayload layer;
    layer.layer_idx = layer_idx;
    const auto& block_ids = kv_cache_manager()->get_block_ids(src_request_id, layer_idx);
    if (static_cast<int32_t>(block_ids.size()) < required_blocks) {
      result.failed = true;
      result.error = "remote_prefill_source_block_count_insufficient";
      break;
    }
    layer.src_block_ids.assign(block_ids.begin(), block_ids.begin() + required_blocks);
    if (nccl_handoff) {
      result.layers.push_back(std::move(layer));
      continue;
    }
    auto& allocator = kv_cache_manager()->allocator_mut(layer_idx);
    for (int32_t block_idx = 0; block_idx < required_blocks; ++block_idx) {
      const int32_t src_block_id = layer.src_block_ids[block_idx];
      const auto ptrs = allocator.get_block_payload_ptrs(src_block_id);
      RemoteKVBlockPayload block;
      block.src_block_id = src_block_id;
      base::Status status = copy_bytes_to_host(
          ptrs.key, ptrs.key_value_bytes, allocator.device_type(), model_stream(),
          &block.key);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      status = copy_bytes_to_host(
          ptrs.value, ptrs.key_value_bytes, allocator.device_type(), model_stream(),
          &block.value);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      if (ptrs.scale_bytes > 0) {
        status = copy_bytes_to_host(
            ptrs.key_scale, ptrs.scale_bytes, allocator.device_type(),
            model_stream(), &block.key_scale);
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
        status = copy_bytes_to_host(
            ptrs.value_scale, ptrs.scale_bytes, allocator.device_type(),
            model_stream(), &block.value_scale);
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
      }
      layer.blocks.push_back(std::move(block));
    }
    if (result.failed) {
      break;
    }
    result.layers.push_back(std::move(layer));
  }

  if (nccl_handoff && !result.failed) {
    std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
    pending_remote_prefills_[result.handoff_id.value] = {src_request_id};
  } else if (kv_cache_manager()->is_valid_request(src_request_id)) {
    kv_cache_manager()->free_request(src_request_id);
  }
  return result;
}

base::Status ServingBenchmarkApp::run_remote_nccl_kv_send(
    const KVBlockManifest& manifest,
    const std::string& nccl_unique_id) const {
#if !defined(KUIPER_ENABLE_NCCL)
  (void)manifest;
  (void)nccl_unique_id;
  return base::error::FunctionNotImplement(
      "NCCL support is not enabled. Reconfigure with "
      "-DKUIPER_ENABLE_NCCL=ON for remote-zmq-nccl modes.");
#else
  PendingRemotePrefill pending;
  {
    std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
    auto it = pending_remote_prefills_.find(manifest.handoff_id.value);
    if (it == pending_remote_prefills_.end()) {
      return base::error::InvalidArgument(
          "remote nccl prefill handoff id not found");
    }
    pending = it->second;
  }
  if (pending.request_id < 0 || !kv_cache_manager()->is_valid_request(pending.request_id)) {
    return base::error::InvalidArgument(
        "remote nccl prefill source request is invalid");
  }
  for (const auto& mapping : manifest.layer_mappings) {
    const auto& blocks =
        kv_cache_manager()->get_block_ids(pending.request_id, mapping.layer_idx);
    if (blocks.size() < mapping.src_block_ids.size()) {
      return base::error::InvalidArgument(
          "remote nccl prefill source block count is insufficient");
    }
    for (size_t i = 0; i < mapping.src_block_ids.size(); ++i) {
      if (blocks[i] != mapping.src_block_ids[i]) {
        return base::error::InvalidArgument(
            "remote nccl prefill source block mapping changed");
      }
    }
  }

  RemoteNcclKVTransferOptions options;
  options.role = RemoteNcclKVTransferRole::kProducer;
  options.kv_manager = kv_cache_manager();
  options.device_id = manifest.src_pool.device_id;
  options.nccl_unique_id = nccl_unique_id;
  options.stream = model_stream();
  options.need_sync = true;
  base::Status status = run_remote_nccl_kv_block_transfer(manifest, options);
  {
    std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
    pending_remote_prefills_.erase(manifest.handoff_id.value);
  }
  if (kv_cache_manager()->is_valid_request(pending.request_id)) {
    kv_cache_manager()->free_request(pending.request_id);
  }
  return status;
#endif
}

void ServingBenchmarkApp::retain_remote_prefill(
    HandoffId handoff_id, base::RequestId request_id) const {
  if (!handoff_id.valid() || request_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
  pending_remote_prefills_[handoff_id.value] = {request_id};
}

void ServingBenchmarkApp::release_remote_prefill(HandoffId handoff_id) const {
  if (!handoff_id.valid()) {
    return;
  }
  PendingRemotePrefill pending;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
    auto it = pending_remote_prefills_.find(handoff_id.value);
    if (it != pending_remote_prefills_.end()) {
      pending = it->second;
      pending_remote_prefills_.erase(it);
      found = true;
    }
  }
  if (found && pending.request_id >= 0 &&
      kv_cache_manager()->is_valid_request(pending.request_id)) {
    kv_cache_manager()->free_request(pending.request_id);
  }
}

ServingBenchmarkApp::PDGenerationResult
ServingBenchmarkApp::run_remote_zmq_cpu_pd_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  PDGenerationResult result;
  generation_config.normalize();
  if (prompt_tokens.empty()) {
    result.failed = true;
    result.error = "empty_prompt";
    return result;
  }

  ZmqRpcConfig rpc = make_prefill_zmq_rpc_config(bench_config_);
  nlohmann::json response;
  base::Status status = zmq_request_response(
      rpc,
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefill)},
       {"prompt_tokens", prompt_tokens},
       {"generation_config", generation_config_to_json(generation_config)}},
      &response);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    return result;
  }
  if (!response.value("ok", false)) {
    result.failed = true;
    result.error = response.value("error", "remote_prefill_failed");
    return result;
  }

  RemotePrefillResult prefill =
      remote_prefill_result_from_json(response.at("prefill_result"));
  if (prefill.failed) {
    result.failed = true;
    result.error = prefill.error;
    result.output_tokens = prefill.output_tokens;
    return result;
  }
  if (prefill.first_token < 0 || prefill.computed_tokens <= 0) {
    result.failed = true;
    result.error = "remote_prefill_missing_decode_ready_metadata";
    return result;
  }
  if (on_token && !is_sentence_ending(prefill.first_token)) {
    on_token(prefill.first_token);
  }

  DecodeKVReservationRequest reservation_request;
  reservation_request.client_request_id.value = "remote-zmq-cpu";
  reservation_request.handoff_id.value = 1;
  reservation_request.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  reservation_request.computed_tokens = prefill.computed_tokens;
  reservation_request.first_token = prefill.first_token;
  reservation_request.src_pool = prefill.src_pool;
  reservation_request.src_block_ids_per_layer.resize(prefill.src_pool.layer_num);
  for (const auto& layer : prefill.layers) {
    if (layer.layer_idx >= 0 &&
        layer.layer_idx < static_cast<int32_t>(
                              reservation_request.src_block_ids_per_layer.size())) {
      reservation_request.src_block_ids_per_layer[layer.layer_idx] =
          layer.src_block_ids;
    }
  }

  DecodeKVReservationManager reservation_manager(kv_cache_manager(),
                                                 pd_decode_kv_pool());
  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  status = reservation_manager.reserve(reservation_request, &reservation, &manifest);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    return result;
  }

  for (const auto& layer : prefill.layers) {
    if (layer.layer_idx < 0 || layer.layer_idx >= pd_decode_kv_pool().layer_num) {
      result.failed = true;
      result.error = "remote_prefill_layer_idx_out_of_range";
      break;
    }
    auto& allocator = kv_cache_manager()->allocator_mut(layer.layer_idx);
    if (layer.blocks.size() !=
        reservation.dst_block_ids_per_layer[layer.layer_idx].size()) {
      result.failed = true;
      result.error = "remote_prefill_block_count_mismatch";
      break;
    }
    for (size_t block_idx = 0; block_idx < layer.blocks.size(); ++block_idx) {
      const int32_t dst_block_id =
          reservation.dst_block_ids_per_layer[layer.layer_idx][block_idx];
      const auto ptrs = allocator.get_block_payload_ptrs(dst_block_id);
      status = copy_bytes_from_host(layer.blocks[block_idx].key, ptrs.key,
                                    ptrs.key_value_bytes,
                                    allocator.device_type(), model_stream());
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      status = copy_bytes_from_host(layer.blocks[block_idx].value, ptrs.value,
                                    ptrs.key_value_bytes,
                                    allocator.device_type(), model_stream());
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      if (ptrs.scale_bytes > 0) {
        status = copy_bytes_from_host(layer.blocks[block_idx].key_scale,
                                      ptrs.key_scale, ptrs.scale_bytes,
                                      allocator.device_type(), model_stream());
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
        status = copy_bytes_from_host(layer.blocks[block_idx].value_scale,
                                      ptrs.value_scale, ptrs.scale_bytes,
                                      allocator.device_type(), model_stream());
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
      }
    }
    if (result.failed) {
      break;
    }
  }
  if (result.failed) {
    reservation_manager.release(&reservation);
    return result;
  }

  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    reservation_manager.release(&reservation);
    return result;
  }

  SchedulerConfig decode_config;
  decode_config.max_num_seqs = max_model_batch_size();
  decode_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  decode_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  decode_config.policy = bench_config_.scheduling_policy;
  decode_config.long_prefill_token_threshold =
      bench_config_.long_prefill_token_threshold;
  decode_config.max_partial_prefills = bench_config_.max_partial_prefills;
  decode_config.max_long_partial_prefills = bench_config_.max_long_partial_prefills;

  Scheduler decode_scheduler(decode_config, kv_cache_manager());
  const base::RequestId decode_request_id = reservation.decode_request_id;
  const int64_t client_id = decode_scheduler.add_decode_ready_request(
      decode_request_id, prompt_tokens, generation_config,
      reservation.reserved_tokens, prefill.first_token);
  reservation = {};

  while (decode_scheduler.has_active_requests()) {
    SchedulerOutput output = decode_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch =
        decode_scheduler.build_decode_batch(output, model_stream());
    status = forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (kv_cache_manager()->is_valid_request(decode_request_id)) {
        kv_cache_manager()->free_request(decode_request_id);
      }
      return result;
    }
    SampledTokenView sampled = batch_sample(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return is_sentence_ending(token); });
  }

  for (const auto& seq : decode_scheduler.pop_finished()) {
    if (seq.client_request_id != client_id) {
      continue;
    }
    result.output_tokens = seq.output_tokens;
    result.failed = seq.failed;
    result.error = seq.finish_reason;
    return result;
  }
  result.failed = true;
  result.error = "remote_decode_result_missing";
  return result;
}

ServingBenchmarkApp::PDGenerationResult
ServingBenchmarkApp::run_remote_zmq_nccl_pd_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  PDGenerationResult result;
#if !defined(KUIPER_ENABLE_NCCL)
  (void)prompt_tokens;
  (void)generation_config;
  (void)on_token;
  result.failed = true;
  result.error =
      "NCCL support is not enabled. Reconfigure with "
      "-DKUIPER_ENABLE_NCCL=ON for remote-zmq-nccl modes.";
  return result;
#else
  generation_config.normalize();
  if (prompt_tokens.empty()) {
    result.failed = true;
    result.error = "empty_prompt";
    return result;
  }

  ZmqRpcConfig rpc = make_prefill_zmq_rpc_config(bench_config_);
  nlohmann::json response;
  base::Status status = zmq_request_response(
      rpc,
      {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kPrefill)},
       {"prompt_tokens", prompt_tokens},
       {"generation_config", generation_config_to_json(generation_config)}},
      &response);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    return result;
  }
  if (!response.value("ok", false)) {
    result.failed = true;
    result.error = response.value("error", "remote_prefill_failed");
    return result;
  }

  RemotePrefillResult prefill =
      remote_prefill_result_from_json(response.at("prefill_result"));
  if (prefill.failed) {
    result.failed = true;
    result.error = prefill.error;
    result.output_tokens = prefill.output_tokens;
    return result;
  }
  if (prefill.first_token < 0 || prefill.computed_tokens <= 0 ||
      !prefill.handoff_id.valid()) {
    result.failed = true;
    result.error = "remote_prefill_missing_nccl_metadata";
    return result;
  }
  auto release_prefill = [&]() {
    ZmqRpcConfig release_rpc = make_prefill_zmq_rpc_config(bench_config_);
    nlohmann::json release_response;
    zmq_request_response(
        release_rpc,
        {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kKvRelease)},
         {"handoff_id", prefill.handoff_id.value}},
        &release_response);
  };
  if (on_token && !is_sentence_ending(prefill.first_token)) {
    on_token(prefill.first_token);
  }

  DecodeKVReservationRequest reservation_request;
  reservation_request.client_request_id =
      prefill.client_request_id.empty()
          ? GlobalRequestId{"remote-zmq-nccl"}
          : prefill.client_request_id;
  reservation_request.handoff_id = prefill.handoff_id;
  reservation_request.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  reservation_request.computed_tokens = prefill.computed_tokens;
  reservation_request.first_token = prefill.first_token;
  reservation_request.src_pool = prefill.src_pool;
  reservation_request.src_block_ids_per_layer.resize(prefill.src_pool.layer_num);
  for (const auto& layer : prefill.layers) {
    if (layer.layer_idx >= 0 &&
        layer.layer_idx < static_cast<int32_t>(
                              reservation_request.src_block_ids_per_layer.size())) {
      reservation_request.src_block_ids_per_layer[layer.layer_idx] =
          layer.src_block_ids;
    }
  }

  DecodeKVReservationManager reservation_manager(kv_cache_manager(),
                                                 pd_decode_kv_pool());
  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  status = reservation_manager.reserve(reservation_request, &reservation, &manifest);
  if (!status) {
    release_prefill();
    result.failed = true;
    result.error = status.get_err_msg();
    return result;
  }

  ncclUniqueId unique_id;
  ncclResult_t nccl_status = ncclGetUniqueId(&unique_id);
  if (nccl_status != ncclSuccess) {
    reservation_manager.release(&reservation);
    release_prefill();
    result.failed = true;
    result.error = std::string("ncclGetUniqueId failed: ") +
                   ncclGetErrorString(nccl_status);
    return result;
  }
  const std::string unique_id_bytes(
      reinterpret_cast<const char*>(&unique_id), sizeof(unique_id));

  auto send_future = std::async(std::launch::async, [this, manifest, unique_id_bytes]() {
    ZmqRpcConfig transfer_rpc = make_prefill_zmq_rpc_config(bench_config_);
    nlohmann::json transfer_response;
    base::Status transfer_status = zmq_request_response(
        transfer_rpc,
        {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kKvTransfer)},
         {"backend", "nccl"},
         {"manifest", kv_block_manifest_to_json(manifest)},
         {"nccl_unique_id", binary_to_hex_json(unique_id_bytes)}},
        &transfer_response);
    if (!transfer_status) {
      return transfer_status;
    }
    if (!transfer_response.value("ok", false)) {
      return base::error::InternalError(
          transfer_response.value("error", "remote_nccl_prefill_send_failed"));
    }
    return base::error::Success();
  });

  RemoteNcclKVTransferOptions recv_options;
  recv_options.role = RemoteNcclKVTransferRole::kConsumer;
  recv_options.kv_manager = kv_cache_manager();
  recv_options.device_id = manifest.dst_pool.device_id;
  recv_options.nccl_unique_id = unique_id_bytes;
  recv_options.stream = model_stream();
  recv_options.need_sync = true;
  status = run_remote_nccl_kv_block_transfer(manifest, recv_options);
  const base::Status send_status = send_future.get();
  if (!status || !send_status) {
    reservation_manager.release(&reservation);
    release_prefill();
    result.failed = true;
    result.error = !status ? status.get_err_msg() : send_status.get_err_msg();
    return result;
  }

  SchedulerConfig decode_config;
  decode_config.max_num_seqs = max_model_batch_size();
  decode_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  decode_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  decode_config.policy = bench_config_.scheduling_policy;
  decode_config.long_prefill_token_threshold =
      bench_config_.long_prefill_token_threshold;
  decode_config.max_partial_prefills = bench_config_.max_partial_prefills;
  decode_config.max_long_partial_prefills = bench_config_.max_long_partial_prefills;

  Scheduler decode_scheduler(decode_config, kv_cache_manager());
  const base::RequestId decode_request_id = reservation.decode_request_id;
  const int64_t client_id = decode_scheduler.add_decode_ready_request(
      decode_request_id, prompt_tokens, generation_config,
      reservation.reserved_tokens, prefill.first_token);
  reservation = {};

  while (decode_scheduler.has_active_requests()) {
    SchedulerOutput output = decode_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch =
        decode_scheduler.build_decode_batch(output, model_stream());
    status = forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (kv_cache_manager()->is_valid_request(decode_request_id)) {
        kv_cache_manager()->free_request(decode_request_id);
      }
      return result;
    }
    SampledTokenView sampled = batch_sample(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return is_sentence_ending(token); });
  }

  for (const auto& seq : decode_scheduler.pop_finished()) {
    if (seq.client_request_id != client_id) {
      continue;
    }
    result.output_tokens = seq.output_tokens;
    result.failed = seq.failed;
    result.error = seq.finish_reason;
    return result;
  }
  result.failed = true;
  result.error = "remote_nccl_decode_result_missing";
  return result;
#endif
}

ServingBenchmarkApp::PDGenerationResult ServingBenchmarkApp::run_dual_gpu_pd_generation_with_mode(
    const std::string& pd_mode,
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  PDGenerationResult result;
  generation_config.normalize();
  if (!pd_dual_gpu_supported()) {
    result.failed = true;
    result.error = "pd_dual_gpu_not_supported";
    return result;
  }
  if (prompt_tokens.empty()) {
    result.failed = true;
    result.error = "empty_prompt";
    return result;
  }

  SchedulerConfig prefill_config;
  prefill_config.max_num_seqs = max_model_batch_size();
  prefill_config.max_num_batched_tokens = bench_config_.max_num_batched_tokens;
  prefill_config.prefill_chunk_cap = bench_config_.prefill_chunk_cap;
  prefill_config.policy = bench_config_.scheduling_policy;
  prefill_config.long_prefill_token_threshold = bench_config_.long_prefill_token_threshold;
  prefill_config.max_partial_prefills = bench_config_.max_partial_prefills;
  prefill_config.max_long_partial_prefills = bench_config_.max_long_partial_prefills;

  if (pd_mode == "dual-gpu-nccl-layer") {
#if !defined(KUIPER_ENABLE_NCCL)
    result.failed = true;
    result.error =
        "NCCL support is not enabled. Reconfigure with "
        "-DKUIPER_ENABLE_NCCL=ON for dual-gpu-nccl-layer mode.";
    return result;
#else
    const int32_t prompt_token_count = static_cast<int32_t>(prompt_tokens.size());
    const base::RequestId prefill_request_id =
        pd_prefill_kv_cache_manager()->register_request();
    cudaSetDevice(bench_config_.decode_device_id);
    DecodeKVReservationManager reservation_manager(pd_decode_kv_cache_manager(),
                                                   pd_decode_kv_pool());
    DecodeKVReservationRequest reservation_request;
    reservation_request.client_request_id.value = "offline-layer-0";
    reservation_request.handoff_id.value = 1;
    reservation_request.prompt_tokens = prompt_token_count;
    reservation_request.computed_tokens = prompt_token_count;
    reservation_request.first_token = -1;
    reservation_request.src_pool = pd_prefill_kv_pool();
    reservation_request.src_block_ids_per_layer.resize(reservation_request.src_pool.layer_num);
    const int32_t required_blocks =
        (prompt_token_count + reservation_request.src_pool.block_size - 1) /
        reservation_request.src_pool.block_size;
    for (int32_t layer_idx = 0; layer_idx < reservation_request.src_pool.layer_num; ++layer_idx) {
      reservation_request.src_block_ids_per_layer[layer_idx].assign(required_blocks, 0);
    }
    DecodeKVReservation reservation;
    KVBlockManifest unused_manifest;
    base::Status status =
        reservation_manager.reserve(reservation_request, &reservation, &unused_manifest);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }

    NcclLayerKVTransferConnector layer_connector(
        pd_prefill_kv_cache_manager(),
        pd_decode_kv_cache_manager(),
        bench_config_.prefill_device_id,
        bench_config_.decode_device_id,
        pd_prefill_stream(),
        pd_transfer_stream(),
        false);
    LayerKVTransferRequest layer_request;
    layer_request.client_request_id.value = "offline-layer-0";
    layer_request.handoff_id.value = 1;
    layer_request.src_request_id = prefill_request_id;
    layer_request.dst_request_id = reservation.decode_request_id;
    layer_request.prompt_tokens = prompt_token_count;
    layer_request.computed_tokens = prompt_token_count;
    layer_request.src_pool = pd_prefill_kv_pool();
    layer_request.dst_pool = pd_decode_kv_pool();

    status = layer_connector.prepare(layer_request);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      reservation_manager.release(&reservation);
      if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }

    cudaSetDevice(bench_config_.prefill_device_id);
    Scheduler prefill_metadata_builder(prefill_config, pd_prefill_kv_cache_manager());
    MixedBatchMetadata last_prefill_batch;
    SchedulerOutput last_prefill_output;
    int32_t computed_tokens = 0;
    while (computed_tokens < prompt_token_count) {
      const int32_t chunk = std::min({
          prompt_token_count - computed_tokens,
          prefill_config.max_num_batched_tokens,
          prefill_config.prefill_chunk_cap});
      if (chunk <= 0) {
        result.failed = true;
        result.error = "layer_prefill_zero_chunk";
        reservation_manager.release(&reservation);
        if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }

      cudaSetDevice(bench_config_.prefill_device_id);
      SchedulerOutput prefill_output;
      SequenceState prefill_seq;
      prefill_seq.request_id = prefill_request_id;
      prefill_seq.client_request_id = 0;
      prefill_seq.prompt_tokens = prompt_tokens;
      prefill_seq.generation_config = generation_config;
      prefill_seq.computed_tokens = computed_tokens;
      prefill_seq.scheduled_tokens = chunk;
      prefill_seq.status = SequenceStatus::kRunning;
      prefill_output.scheduled_seqs.push_back(&prefill_seq);
      prefill_output.num_tokens_per_seq.push_back(chunk);
      prefill_output.total_tokens = chunk;
      prefill_output.num_prefill_seqs = 1;
      MixedBatchMetadata prefill_batch =
          prefill_metadata_builder.build_mixed_batch(prefill_output, pd_prefill_stream());

      set_pd_prefill_layer_kv_connector(
          &layer_connector, LayerKVConnectorRole::kProducer);
      status = pd_forward_prefill_batch(prefill_batch);
      set_pd_prefill_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        layer_connector.cancel(status.get_err_msg());
        reservation_manager.release(&reservation);
        if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }

      computed_tokens += chunk;
      if (computed_tokens == prompt_token_count) {
        last_prefill_batch = prefill_batch;
        last_prefill_output = prefill_output;
      }
    }

    for (int32_t layer_idx = 0; layer_idx < pd_prefill_kv_pool().layer_num; ++layer_idx) {
      status = layer_connector.save_kv_layer(layer_idx);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        layer_connector.cancel(status.get_err_msg());
        reservation_manager.release(&reservation);
        if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }
    }

    while (true) {
      KVTransferStatus transfer_status = layer_connector.poll();
      if (transfer_status.ok()) {
        break;
      }
      if (transfer_status.state == KVTransferState::kFailed ||
          transfer_status.state == KVTransferState::kCancelled) {
        result.failed = true;
        result.error = transfer_status.error;
        reservation_manager.release(&reservation);
        if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }
    }

    SampledTokenView first_sampled =
        pd_batch_sample_prefill(last_prefill_batch, last_prefill_output);
    if (first_sampled.empty()) {
      result.failed = true;
      result.error = "layer_prefill_missing_first_token";
      reservation_manager.release(&reservation);
      if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }
    const int32_t first_token = first_sampled.tokens[0];
    if (on_token && !is_sentence_ending(first_token)) {
      on_token(first_token);
    }
    if (pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
      pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
    }

    SchedulerConfig decode_config = prefill_config;
    cudaSetDevice(bench_config_.decode_device_id);
    Scheduler decode_scheduler(decode_config, pd_decode_kv_cache_manager());
    const base::RequestId decode_request_id = reservation.decode_request_id;
    const int64_t d_client_id = decode_scheduler.add_decode_ready_request(
        reservation.decode_request_id, prompt_tokens, generation_config,
        reservation.reserved_tokens, first_token);
    reservation = {};

    set_pd_decode_layer_kv_connector(
        &layer_connector, LayerKVConnectorRole::kConsumer);
    while (decode_scheduler.has_active_requests()) {
      cudaSetDevice(bench_config_.decode_device_id);
      SchedulerOutput output = decode_scheduler.schedule_step();
      if (output.total_tokens == 0) {
        continue;
      }
      MixedBatchMetadata batch = decode_scheduler.build_decode_batch(output, pd_decode_stream());
      status = pd_forward_decode_batch(batch);
      if (!status) {
        set_pd_decode_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);
        result.failed = true;
        result.error = status.get_err_msg();
        if (pd_decode_kv_cache_manager()->is_valid_request(decode_request_id)) {
          pd_decode_kv_cache_manager()->free_request(decode_request_id);
        }
        reservation = {};
        return result;
      }
      SampledTokenView sampled = pd_batch_sample_decode(batch, output);
      for (int32_t i = 0; i < sampled.size(); ++i) {
        if (on_token && !is_sentence_ending(sampled.tokens[i])) {
          on_token(sampled.tokens[i]);
        }
      }
      decode_scheduler.process_outputs(
          output, batch, sampled,
          [&](int32_t token) { return is_sentence_ending(token); });
    }
    set_pd_decode_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);

    auto finished = decode_scheduler.pop_finished();
    for (const auto& seq : finished) {
      if (seq.client_request_id != d_client_id) {
        continue;
      }
      result.output_tokens = seq.output_tokens;
      result.failed = seq.failed;
      result.error = seq.finish_reason;
      return result;
    }
    result.failed = true;
    result.error = "decode_result_missing";
    return result;
#endif
  }

  cudaSetDevice(bench_config_.prefill_device_id);
  Scheduler prefill_scheduler(prefill_config, pd_prefill_kv_cache_manager());
  const int64_t p_client_id = prefill_scheduler.add_request(prompt_tokens, generation_config);
  base::RequestId src_request_id = -1;
  const SequenceState* prefill_seq = nullptr;

  while (prefill_scheduler.has_active_requests()) {
    cudaSetDevice(bench_config_.prefill_device_id);
    SchedulerOutput output = prefill_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch = prefill_scheduler.build_mixed_batch(output, pd_prefill_stream());
    base::Status status = pd_forward_prefill_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      return result;
    }
    SampledTokenView sampled = pd_batch_sample_prefill(batch, output);
    prefill_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return is_sentence_ending(token); });
    auto finished_now = prefill_scheduler.pop_finished();
    for (const auto& seq : finished_now) {
      if (seq.client_request_id != p_client_id) {
        continue;
      }
      result.output_tokens = seq.output_tokens;
      result.failed = seq.failed;
      result.error = seq.finish_reason;
      if (on_token) {
        for (int32_t token : result.output_tokens) {
          on_token(token);
        }
      }
      return result;
    }
    for (const auto* seq : output.scheduled_seqs) {
      if (seq != nullptr && seq->client_request_id == p_client_id &&
          !seq->is_prefill() && !seq->output_tokens.empty()) {
        src_request_id = seq->request_id;
        prefill_seq = seq;
        break;
      }
    }
    if (prefill_seq != nullptr) {
      break;
    }
  }
  if (prefill_seq == nullptr || src_request_id < 0 ||
      prefill_seq->output_tokens.empty()) {
    result.failed = true;
    result.error = "prefill_did_not_produce_first_token";
    return result;
  }

  const int32_t first_token = prefill_seq->output_tokens.front();
  if (on_token && !is_sentence_ending(first_token)) {
    on_token(first_token);
  }

  DecodeKVReservationManager reservation_manager(pd_decode_kv_cache_manager(),
                                                 pd_decode_kv_pool());
  PDHandoffBuilder handoff_builder(pd_prefill_kv_cache_manager(), pd_prefill_kv_pool());
  PrefillHandoffBuildRequest build_request;
  build_request.scheduled_request_index = 0;
  build_request.client_request_id.value = "offline-" + std::to_string(p_client_id);
  build_request.handoff_id.value = static_cast<uint64_t>(p_client_id + 1);
  DecodeKVReservationRequest reservation_request;
  SchedulerOutput handoff_output;
  handoff_output.scheduled_seqs.push_back(const_cast<SequenceState*>(prefill_seq));
  handoff_output.num_tokens_per_seq.push_back(0);
  base::Status status = handoff_builder.build_decode_reservation_request(
      handoff_output, build_request, &reservation_request);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  cudaSetDevice(bench_config_.decode_device_id);
  status = reservation_manager.reserve(reservation_request, &reservation, &manifest);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }

  std::unique_ptr<KVTransferConnector> connector;
  if (pd_mode == "dual-gpu-nccl") {
#if !defined(KUIPER_ENABLE_NCCL)
    result.failed = true;
    result.error =
        "NCCL support is not enabled. Reconfigure with "
        "-DKUIPER_ENABLE_NCCL=ON for dual-gpu-nccl mode.";
    reservation_manager.release(&reservation);
    pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
#else
    connector = std::make_unique<NcclKVBlockTransferConnector>(
        pd_prefill_kv_cache_manager(),
        pd_decode_kv_cache_manager(),
        bench_config_.prefill_device_id,
        bench_config_.decode_device_id,
        pd_transfer_stream(),
        false);
#endif
  } else {
    connector = std::make_unique<CudaP2PKVTransferConnector>(
        pd_prefill_kv_cache_manager(),
        pd_decode_kv_cache_manager(),
        bench_config_.prefill_device_id,
        bench_config_.decode_device_id,
        pd_transfer_stream(),
        false);
  }
  PDCoordinator coordinator(connector.get());
  PDHandoffState handoff_state;
  status = coordinator.start_prefill_handoff(manifest, &handoff_state);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    reservation_manager.release(&reservation);
    pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }
  while (!handoff_state.terminal()) {
    status = coordinator.advance(&handoff_state);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      reservation_manager.release(&reservation);
      pd_prefill_kv_cache_manager()->free_request(src_request_id);
      return result;
    }
  }
  if (pd_transfer_stream() != nullptr) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(pd_transfer_stream()));
  }
  if (!handoff_state.ready_for_decode()) {
    result.failed = true;
    result.error = handoff_state.error.empty() ? "pd_handoff_failed" : handoff_state.error;
    reservation_manager.release(&reservation);
    pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }

  pd_prefill_kv_cache_manager()->free_request(src_request_id);

  SchedulerConfig decode_config = prefill_config;
  cudaSetDevice(bench_config_.decode_device_id);
  Scheduler decode_scheduler(decode_config, pd_decode_kv_cache_manager());
  const int64_t d_client_id = decode_scheduler.add_decode_ready_request(
      reservation.decode_request_id, prompt_tokens, generation_config,
      reservation.reserved_tokens, first_token);

  while (decode_scheduler.has_active_requests()) {
    cudaSetDevice(bench_config_.decode_device_id);
    SchedulerOutput output = decode_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch = decode_scheduler.build_decode_batch(output, pd_decode_stream());
    status = pd_forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (pd_decode_kv_cache_manager()->is_valid_request(reservation.decode_request_id)) {
        pd_decode_kv_cache_manager()->free_request(reservation.decode_request_id);
      }
      reservation = {};
      return result;
    }
    SampledTokenView sampled = pd_batch_sample_decode(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return is_sentence_ending(token); });
  }

  auto finished = decode_scheduler.pop_finished();
  reservation = {};
  for (const auto& seq : finished) {
    if (seq.client_request_id != d_client_id) {
      continue;
    }
    result.output_tokens = seq.output_tokens;
    result.failed = seq.failed;
    result.error = seq.finish_reason;
    return result;
  }
  result.failed = true;
  result.error = "decode_result_missing";
  return result;
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
    PDGenerationResult generation = run_dual_gpu_pd_generation(
        std::move(tokens), GenerationConfig(bench_config_.max_new_tokens));
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

std::string ServingBenchmarkApp::postprocess_decoded_text(std::string text) const {
  return text;
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
              << " [--prefill-chunk-cap=N] [--scheduling-policy=fcfs|priority]"
              << " [--long-prefill-token-threshold=N] [--max-partial-prefills=N]"
              << " [--max-long-partial-prefills=N]"
              << " [--device-id=N] [--kv-cache-memory-utilization=0.8] [--quiet=0|1]"
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
              << " [--engine-zmq-timeout-ms=N]";
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
    scheduler.add_request(std::move(tokens), GenerationConfig(max_new_tokens));
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
