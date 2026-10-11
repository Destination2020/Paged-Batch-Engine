#include "serving/serving_config_parser.h"
#include "serving/serving_zmq_rpc.h"
#include <glog/logging.h>
#include <algorithm>
#include <stdexcept>
namespace serving {
namespace {
bool parse_bool_flag(std::string_view value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
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

bool parse_preemption_policy(std::string_view value, PreemptionPolicy* policy) {
  if (!policy) return false;
  if (value == "recompute") *policy = PreemptionPolicy::kRecompute;
  else if (value == "checkpoint") *policy = PreemptionPolicy::kCheckpoint;
  else if (value == "auto") *policy = PreemptionPolicy::kAuto;
  else return false;
  return true;
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

    auto read_size = [&](std::string_view prefix, size_t& out) {
      if (!starts_with(arg, prefix)) return false;
      out = static_cast<size_t>(std::stoull(std::string(arg.substr(prefix.size()))));
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

    if (starts_with(arg, "--seed=")) {
      config.sampling.seed = std::stoull(std::string(arg.substr(7)));
      continue;
    }
    if (read_double("--temperature=", config.sampling.temperature) ||
        read_double("--top-p=", config.sampling.top_p) ||
        read_int("--top-k=", config.sampling.top_k) ||
        read_bool("--ignore-eos=", config.ignore_eos)) {
      config.sampling.normalize();
      continue;
    }
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
        read_int("--kv-cache-blocks-per-layer=", config.kv_cache_blocks_per_layer) ||
        read_double("--kv-cache-memory-utilization=", config.kv_cache_memory_utilization) ||
        read_double("--gpu-memory-utilization=", config.kv_cache_memory_utilization) ||
        read_toggle("--radix-cache=", config.radix_cache_enabled,
                    config.radix_cache_config_explicit) ||
        read_toggle("--enable-radix-cache=", config.radix_cache_enabled,
                    config.radix_cache_config_explicit) ||
        read_bool("--host-cache=", config.host_cache_enabled) ||
        read_size("--host-cache-bytes=", config.host_cache_bytes) ||
        read_size("--host-cache-pages=", config.host_cache_pages) ||
        read_size("--host-cache-inflight-pages=", config.host_cache_inflight_pages) ||
        read_bool("--host-demote-after-warmup=", config.host_demote_after_warmup) ||
        read_bool("--quiet=", config.quiet) ||
        read_bool("--step-profile=", config.print_step_profile) ||
        read_bool("--step-trace=", config.print_step_trace) ||
        read_bool("--final-summary=", config.print_final_summary) ||
        read_bool("--online-server=", config.online_server) ||
        read_int("--listen-port=", config.listen_port) ||
        read_int("--http-worker-threads=", config.http_worker_threads) ||
        read_int("--http-listen-backlog=", config.http_listen_backlog) ||
        read_int("--http-io-timeout-ms=", config.http_io_timeout_ms) ||
        read_int("--http-max-pending-connections=", config.http_max_pending_connections) ||
        read_int("--max-queue-size=", config.max_queue_size) ||
        read_int("--request-timeout-ms=", config.request_timeout_ms) ||
        read_int("--max-prompt-tokens=", config.max_prompt_tokens) ||
        read_int("--kv-cache-budget-mib=", config.kv_cache_budget_mib) ||
        read_int("--private-memory-budget-mib=", config.private_memory_budget_mib) ||
        read_int("--shared-kv-hard-limit-mib=", config.shared_kv_hard_limit_mib) ||
        read_int("--shared-kv-chunk-mib=", config.shared_kv_chunk_mib) ||
        read_int("--max-output-tokens=", config.max_output_tokens) ||
        read_int("--max-active-requests=", config.max_active_requests) ||
        read_int("--device-id=", config.device_id) ||
        read_int("--prefill-device-id=", config.prefill_device_id) ||
        read_int("--decode-device-id=", config.decode_device_id) ||
        read_int("--engine-zmq-timeout-ms=", config.engine_zmq_timeout_ms)) {
      continue;
    }
    auto read_string = [&](std::string_view prefix, std::string& value) {
      if (!starts_with(arg, prefix)) return false;
      value = std::string(arg.substr(prefix.size()));
      return true;
    };
    if (read_string("--model-id=", config.model_id) ||
        read_string("--model-registry=", config.model_registry_path) ||
        read_string("--shared-kv-endpoint=", config.shared_kv_endpoint)) continue;
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
    const std::string_view data_service_endpoint_prefix =
        "--data-service-endpoint=";
    if (starts_with(arg, data_service_endpoint_prefix)) {
      config.data_service_endpoint =
          std::string(arg.substr(data_service_endpoint_prefix.size()));
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
    const std::string_view preemption_policy_prefix = "--preemption-policy=";
    if (starts_with(arg, preemption_policy_prefix)) {
      const auto value = arg.substr(preemption_policy_prefix.size());
      CHECK(parse_preemption_policy(value, &config.preemption_policy))
          << "Invalid --preemption-policy=" << value
          << ". Expected recompute, checkpoint, or auto.";
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
  config.kv_cache_blocks_per_layer =
      std::max(0, config.kv_cache_blocks_per_layer);
  config.http_worker_threads = std::max(0, config.http_worker_threads);
  config.http_listen_backlog = std::max(1, config.http_listen_backlog);
  if (config.http_io_timeout_ms <= 0 || config.http_max_pending_connections <= 0)
    throw std::invalid_argument("HTTP I/O timeout and pending connection limit must be positive");
  config.kv_cache_memory_utilization =
      std::min(1.0, std::max(0.01, config.kv_cache_memory_utilization));
  if (config.host_cache_enabled) {
    CHECK_GT(config.host_cache_bytes, 0u) << "--host-cache-bytes must be positive";
    CHECK_GT(config.host_cache_pages, 0u) << "--host-cache-pages must be positive";
    CHECK_GT(config.host_cache_inflight_pages, 0u)
        << "--host-cache-inflight-pages must be positive";
  }
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

}
