#include "serving/serving_runtime.h"
#include "serving/pd_executor.h"
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

}
PDExecutor::PDGenerationResult PDExecutor::run_dual_gpu_pd_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  return run_dual_gpu_pd_generation_with_mode(
      runtime_.bench_config().pd.pd_mode, std::move(prompt_tokens), std::move(generation_config),
      on_token);
}

RemotePrefillResult PDExecutor::run_remote_prefill_generation(
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config) const {
  RemotePrefillResult result;
  const bool nccl_handoff = runtime_.bench_config().pd.pd_mode == "remote-zmq-nccl";
  result.prompt_tokens = prompt_tokens;
  generation_config.normalize();
  if (prompt_tokens.empty()) {
    result.failed = true;
    result.error = "empty_prompt";
    return result;
  }

  SchedulerConfig prefill_config;
  prefill_config.max_num_seqs = runtime_.max_model_batch_size();
  prefill_config.max_num_batched_tokens = runtime_.bench_config().max_num_batched_tokens;
  prefill_config.prefill_chunk_cap = runtime_.bench_config().prefill_chunk_cap;
  prefill_config.policy = runtime_.bench_config().scheduling_policy;
  prefill_config.preemption_policy = runtime_.bench_config().preemption_policy;
  prefill_config.long_prefill_token_threshold =
      runtime_.bench_config().long_prefill_token_threshold;
  prefill_config.max_partial_prefills = runtime_.bench_config().max_partial_prefills;
  prefill_config.max_long_partial_prefills =
      runtime_.bench_config().max_long_partial_prefills;

  Scheduler prefill_scheduler(prefill_config, runtime_.kv_cache_manager());
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
        prefill_scheduler.build_mixed_batch(output, runtime_.model_stream());
    base::Status status = runtime_.forward_mixed_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      return result;
    }
    SampledTokenView sampled = runtime_.batch_sample(batch, output);
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
        [&](int32_t token) { return runtime_.is_sentence_ending(token); });
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
  result.src_pool = runtime_.pd_prefill_kv_pool();
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
    const auto& block_ids = runtime_.kv_cache_manager()->get_block_ids(src_request_id, layer_idx);
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
    auto& allocator = runtime_.kv_cache_manager()->allocator_mut(layer_idx);
    for (int32_t block_idx = 0; block_idx < required_blocks; ++block_idx) {
      const int32_t src_block_id = layer.src_block_ids[block_idx];
      const auto ptrs = allocator.get_block_payload_ptrs(src_block_id);
      RemoteKVBlockPayload block;
      block.src_block_id = src_block_id;
      base::Status status = copy_bytes_to_host(
          ptrs.key, ptrs.key_value_bytes, allocator.device_type(), runtime_.model_stream(),
          &block.key);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      status = copy_bytes_to_host(
          ptrs.value, ptrs.key_value_bytes, allocator.device_type(), runtime_.model_stream(),
          &block.value);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      if (ptrs.scale_bytes > 0) {
        status = copy_bytes_to_host(
            ptrs.key_scale, ptrs.scale_bytes, allocator.device_type(),
            runtime_.model_stream(), &block.key_scale);
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
        status = copy_bytes_to_host(
            ptrs.value_scale, ptrs.scale_bytes, allocator.device_type(),
            runtime_.model_stream(), &block.value_scale);
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
  } else if (runtime_.kv_cache_manager()->is_valid_request(src_request_id)) {
    runtime_.kv_cache_manager()->free_request(src_request_id);
  }
  return result;
}

base::Status PDExecutor::run_remote_nccl_kv_send(
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
  if (pending.request_id < 0 || !runtime_.kv_cache_manager()->is_valid_request(pending.request_id)) {
    return base::error::InvalidArgument(
        "remote nccl prefill source request is invalid");
  }
  for (const auto& mapping : manifest.layer_mappings) {
    const auto& blocks =
        runtime_.kv_cache_manager()->get_block_ids(pending.request_id, mapping.layer_idx);
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
  options.kv_manager = runtime_.kv_cache_manager();
  options.device_id = manifest.src_pool.device_id;
  options.nccl_unique_id = nccl_unique_id;
  options.stream = runtime_.model_stream();
  options.need_sync = true;
  base::Status status = run_remote_nccl_kv_block_transfer(manifest, options);
  {
    std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
    pending_remote_prefills_.erase(manifest.handoff_id.value);
  }
  if (runtime_.kv_cache_manager()->is_valid_request(pending.request_id)) {
    runtime_.kv_cache_manager()->free_request(pending.request_id);
  }
  return status;
#endif
}

void PDExecutor::retain_remote_prefill(
    HandoffId handoff_id, base::RequestId request_id) const {
  if (!handoff_id.valid() || request_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(pending_remote_prefills_mu_);
  pending_remote_prefills_[handoff_id.value] = {request_id};
}

void PDExecutor::release_remote_prefill(HandoffId handoff_id) const {
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
      runtime_.kv_cache_manager()->is_valid_request(pending.request_id)) {
    runtime_.kv_cache_manager()->free_request(pending.request_id);
  }
}

PDExecutor::PDGenerationResult
PDExecutor::run_remote_zmq_cpu_pd_generation(
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

  ZmqRpcConfig rpc = make_prefill_zmq_rpc_config(runtime_.bench_config());
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
  if (on_token && !runtime_.is_sentence_ending(prefill.first_token)) {
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

  DecodeKVReservationManager reservation_manager(runtime_.kv_cache_manager(),
                                                 runtime_.pd_decode_kv_pool());
  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  status = reservation_manager.reserve(reservation_request, &reservation, &manifest);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    return result;
  }

  for (const auto& layer : prefill.layers) {
    if (layer.layer_idx < 0 || layer.layer_idx >= runtime_.pd_decode_kv_pool().layer_num) {
      result.failed = true;
      result.error = "remote_prefill_layer_idx_out_of_range";
      break;
    }
    auto& allocator = runtime_.kv_cache_manager()->allocator_mut(layer.layer_idx);
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
                                    allocator.device_type(), runtime_.model_stream());
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      status = copy_bytes_from_host(layer.blocks[block_idx].value, ptrs.value,
                                    ptrs.key_value_bytes,
                                    allocator.device_type(), runtime_.model_stream());
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        break;
      }
      if (ptrs.scale_bytes > 0) {
        status = copy_bytes_from_host(layer.blocks[block_idx].key_scale,
                                      ptrs.key_scale, ptrs.scale_bytes,
                                      allocator.device_type(), runtime_.model_stream());
        if (!status) {
          result.failed = true;
          result.error = status.get_err_msg();
          break;
        }
        status = copy_bytes_from_host(layer.blocks[block_idx].value_scale,
                                      ptrs.value_scale, ptrs.scale_bytes,
                                      allocator.device_type(), runtime_.model_stream());
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
  decode_config.max_num_seqs = runtime_.max_model_batch_size();
  decode_config.max_num_batched_tokens = runtime_.bench_config().max_num_batched_tokens;
  decode_config.prefill_chunk_cap = runtime_.bench_config().prefill_chunk_cap;
  decode_config.policy = runtime_.bench_config().scheduling_policy;
  decode_config.preemption_policy = runtime_.bench_config().preemption_policy;
  decode_config.long_prefill_token_threshold =
      runtime_.bench_config().long_prefill_token_threshold;
  decode_config.max_partial_prefills = runtime_.bench_config().max_partial_prefills;
  decode_config.max_long_partial_prefills = runtime_.bench_config().max_long_partial_prefills;

  Scheduler decode_scheduler(decode_config, runtime_.kv_cache_manager());
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
        decode_scheduler.build_decode_batch(output, runtime_.model_stream());
    status = runtime_.forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (runtime_.kv_cache_manager()->is_valid_request(decode_request_id)) {
        runtime_.kv_cache_manager()->free_request(decode_request_id);
      }
      return result;
    }
    SampledTokenView sampled = runtime_.batch_sample(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !runtime_.is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return runtime_.is_sentence_ending(token); });
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

PDExecutor::PDGenerationResult
PDExecutor::run_remote_zmq_nccl_pd_generation(
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

  ZmqRpcConfig rpc = make_prefill_zmq_rpc_config(runtime_.bench_config());
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
    ZmqRpcConfig release_rpc = make_prefill_zmq_rpc_config(runtime_.bench_config());
    nlohmann::json release_response;
    zmq_request_response(
        release_rpc,
        {{"type", zmq_rpc_message_type_name(ZmqRpcMessageType::kKvRelease)},
         {"handoff_id", prefill.handoff_id.value}},
        &release_response);
  };
  if (on_token && !runtime_.is_sentence_ending(prefill.first_token)) {
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

  DecodeKVReservationManager reservation_manager(runtime_.kv_cache_manager(),
                                                 runtime_.pd_decode_kv_pool());
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
    ZmqRpcConfig transfer_rpc = make_prefill_zmq_rpc_config(runtime_.bench_config());
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
  recv_options.kv_manager = runtime_.kv_cache_manager();
  recv_options.device_id = manifest.dst_pool.device_id;
  recv_options.nccl_unique_id = unique_id_bytes;
  recv_options.stream = runtime_.model_stream();
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
  decode_config.max_num_seqs = runtime_.max_model_batch_size();
  decode_config.max_num_batched_tokens = runtime_.bench_config().max_num_batched_tokens;
  decode_config.prefill_chunk_cap = runtime_.bench_config().prefill_chunk_cap;
  decode_config.policy = runtime_.bench_config().scheduling_policy;
  decode_config.preemption_policy = runtime_.bench_config().preemption_policy;
  decode_config.long_prefill_token_threshold =
      runtime_.bench_config().long_prefill_token_threshold;
  decode_config.max_partial_prefills = runtime_.bench_config().max_partial_prefills;
  decode_config.max_long_partial_prefills = runtime_.bench_config().max_long_partial_prefills;

  Scheduler decode_scheduler(decode_config, runtime_.kv_cache_manager());
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
        decode_scheduler.build_decode_batch(output, runtime_.model_stream());
    status = runtime_.forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (runtime_.kv_cache_manager()->is_valid_request(decode_request_id)) {
        runtime_.kv_cache_manager()->free_request(decode_request_id);
      }
      return result;
    }
    SampledTokenView sampled = runtime_.batch_sample(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !runtime_.is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return runtime_.is_sentence_ending(token); });
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

PDExecutor::PDGenerationResult PDExecutor::run_dual_gpu_pd_generation_with_mode(
    const std::string& pd_mode,
    std::vector<int32_t> prompt_tokens,
    GenerationConfig generation_config,
    const std::function<void(int32_t)>& on_token) const {
  PDGenerationResult result;
  generation_config.normalize();
  if (!runtime_.pd_dual_gpu_supported()) {
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
  prefill_config.max_num_seqs = runtime_.max_model_batch_size();
  prefill_config.max_num_batched_tokens = runtime_.bench_config().max_num_batched_tokens;
  prefill_config.prefill_chunk_cap = runtime_.bench_config().prefill_chunk_cap;
  prefill_config.policy = runtime_.bench_config().scheduling_policy;
  prefill_config.preemption_policy = runtime_.bench_config().preemption_policy;
  prefill_config.long_prefill_token_threshold = runtime_.bench_config().long_prefill_token_threshold;
  prefill_config.max_partial_prefills = runtime_.bench_config().max_partial_prefills;
  prefill_config.max_long_partial_prefills = runtime_.bench_config().max_long_partial_prefills;

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
        runtime_.pd_prefill_kv_cache_manager()->register_request();
    cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
    DecodeKVReservationManager reservation_manager(runtime_.pd_decode_kv_cache_manager(),
                                                   runtime_.pd_decode_kv_pool());
    DecodeKVReservationRequest reservation_request;
    reservation_request.client_request_id.value = "offline-layer-0";
    reservation_request.handoff_id.value = 1;
    reservation_request.prompt_tokens = prompt_token_count;
    reservation_request.computed_tokens = prompt_token_count;
    reservation_request.first_token = -1;
    reservation_request.src_pool = runtime_.pd_prefill_kv_pool();
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
      if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }

    NcclLayerKVTransferConnector layer_connector(
        runtime_.pd_prefill_kv_cache_manager(),
        runtime_.pd_decode_kv_cache_manager(),
        runtime_.bench_config().pd.prefill_device_id,
        runtime_.bench_config().pd.decode_device_id,
        runtime_.pd_prefill_stream(),
        runtime_.pd_transfer_stream(),
        false);
    LayerKVTransferRequest layer_request;
    layer_request.client_request_id.value = "offline-layer-0";
    layer_request.handoff_id.value = 1;
    layer_request.src_request_id = prefill_request_id;
    layer_request.dst_request_id = reservation.decode_request_id;
    layer_request.prompt_tokens = prompt_token_count;
    layer_request.computed_tokens = prompt_token_count;
    layer_request.src_pool = runtime_.pd_prefill_kv_pool();
    layer_request.dst_pool = runtime_.pd_decode_kv_pool();

    status = layer_connector.prepare(layer_request);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      reservation_manager.release(&reservation);
      if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }

    cudaSetDevice(runtime_.bench_config().pd.prefill_device_id);
    Scheduler prefill_metadata_builder(prefill_config, runtime_.pd_prefill_kv_cache_manager());
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
        if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }

      cudaSetDevice(runtime_.bench_config().pd.prefill_device_id);
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
          prefill_metadata_builder.build_mixed_batch(prefill_output, runtime_.pd_prefill_stream());

      runtime_.set_pd_prefill_layer_kv_connector(
          &layer_connector, LayerKVConnectorRole::kProducer);
      status = runtime_.pd_forward_prefill_batch(prefill_batch);
      runtime_.set_pd_prefill_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        layer_connector.cancel(status.get_err_msg());
        reservation_manager.release(&reservation);
        if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }

      computed_tokens += chunk;
      if (computed_tokens == prompt_token_count) {
        last_prefill_batch = prefill_batch;
        last_prefill_output = prefill_output;
      }
    }

    for (int32_t layer_idx = 0; layer_idx < runtime_.pd_prefill_kv_pool().layer_num; ++layer_idx) {
      status = layer_connector.save_kv_layer(layer_idx);
      if (!status) {
        result.failed = true;
        result.error = status.get_err_msg();
        layer_connector.cancel(status.get_err_msg());
        reservation_manager.release(&reservation);
        if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
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
        if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
          runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
        }
        return result;
      }
    }

    SampledTokenView first_sampled =
        runtime_.pd_batch_sample_prefill(last_prefill_batch, last_prefill_output);
    if (first_sampled.empty()) {
      result.failed = true;
      result.error = "layer_prefill_missing_first_token";
      reservation_manager.release(&reservation);
      if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
        runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
      }
      return result;
    }
    const int32_t first_token = first_sampled.tokens[0];
    if (on_token && !runtime_.is_sentence_ending(first_token)) {
      on_token(first_token);
    }
    if (runtime_.pd_prefill_kv_cache_manager()->is_valid_request(prefill_request_id)) {
      runtime_.pd_prefill_kv_cache_manager()->free_request(prefill_request_id);
    }

    SchedulerConfig decode_config = prefill_config;
    cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
    Scheduler decode_scheduler(decode_config, runtime_.pd_decode_kv_cache_manager());
    const base::RequestId decode_request_id = reservation.decode_request_id;
    const int64_t d_client_id = decode_scheduler.add_decode_ready_request(
        reservation.decode_request_id, prompt_tokens, generation_config,
        reservation.reserved_tokens, first_token);
    reservation = {};

    runtime_.set_pd_decode_layer_kv_connector(
        &layer_connector, LayerKVConnectorRole::kConsumer);
    while (decode_scheduler.has_active_requests()) {
      cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
      SchedulerOutput output = decode_scheduler.schedule_step();
      if (output.total_tokens == 0) {
        continue;
      }
      MixedBatchMetadata batch = decode_scheduler.build_decode_batch(output, runtime_.pd_decode_stream());
      status = runtime_.pd_forward_decode_batch(batch);
      if (!status) {
        runtime_.set_pd_decode_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);
        result.failed = true;
        result.error = status.get_err_msg();
        if (runtime_.pd_decode_kv_cache_manager()->is_valid_request(decode_request_id)) {
          runtime_.pd_decode_kv_cache_manager()->free_request(decode_request_id);
        }
        reservation = {};
        return result;
      }
      SampledTokenView sampled = runtime_.pd_batch_sample_decode(batch, output);
      for (int32_t i = 0; i < sampled.size(); ++i) {
        if (on_token && !runtime_.is_sentence_ending(sampled.tokens[i])) {
          on_token(sampled.tokens[i]);
        }
      }
      decode_scheduler.process_outputs(
          output, batch, sampled,
          [&](int32_t token) { return runtime_.is_sentence_ending(token); });
    }
    runtime_.set_pd_decode_layer_kv_connector(nullptr, LayerKVConnectorRole::kDisabled);

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

  cudaSetDevice(runtime_.bench_config().pd.prefill_device_id);
  Scheduler prefill_scheduler(prefill_config, runtime_.pd_prefill_kv_cache_manager());
  const int64_t p_client_id = prefill_scheduler.add_request(prompt_tokens, generation_config);
  base::RequestId src_request_id = -1;
  const SequenceState* prefill_seq = nullptr;

  while (prefill_scheduler.has_active_requests()) {
    cudaSetDevice(runtime_.bench_config().pd.prefill_device_id);
    SchedulerOutput output = prefill_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch = prefill_scheduler.build_mixed_batch(output, runtime_.pd_prefill_stream());
    base::Status status = runtime_.pd_forward_prefill_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      return result;
    }
    SampledTokenView sampled = runtime_.pd_batch_sample_prefill(batch, output);
    prefill_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return runtime_.is_sentence_ending(token); });
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
  if (on_token && !runtime_.is_sentence_ending(first_token)) {
    on_token(first_token);
  }

  DecodeKVReservationManager reservation_manager(runtime_.pd_decode_kv_cache_manager(),
                                                 runtime_.pd_decode_kv_pool());
  PDHandoffBuilder handoff_builder(runtime_.pd_prefill_kv_cache_manager(), runtime_.pd_prefill_kv_pool());
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
    runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }

  DecodeKVReservation reservation;
  KVBlockManifest manifest;
  cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
  status = reservation_manager.reserve(reservation_request, &reservation, &manifest);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
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
    runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
#else
    connector = std::make_unique<NcclKVBlockTransferConnector>(
        runtime_.pd_prefill_kv_cache_manager(),
        runtime_.pd_decode_kv_cache_manager(),
        runtime_.bench_config().pd.prefill_device_id,
        runtime_.bench_config().pd.decode_device_id,
        runtime_.pd_transfer_stream(),
        false);
#endif
  } else {
    connector = std::make_unique<CudaP2PKVTransferConnector>(
        runtime_.pd_prefill_kv_cache_manager(),
        runtime_.pd_decode_kv_cache_manager(),
        runtime_.bench_config().pd.prefill_device_id,
        runtime_.bench_config().pd.decode_device_id,
        runtime_.pd_transfer_stream(),
        false);
  }
  PDCoordinator coordinator(connector.get());
  PDHandoffState handoff_state;
  status = coordinator.start_prefill_handoff(manifest, &handoff_state);
  if (!status) {
    result.failed = true;
    result.error = status.get_err_msg();
    reservation_manager.release(&reservation);
    runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }
  while (!handoff_state.terminal()) {
    status = coordinator.advance(&handoff_state);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      reservation_manager.release(&reservation);
      runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
      return result;
    }
  }
  if (runtime_.pd_transfer_stream() != nullptr) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(runtime_.pd_transfer_stream()));
  }
  if (!handoff_state.ready_for_decode()) {
    result.failed = true;
    result.error = handoff_state.error.empty() ? "pd_handoff_failed" : handoff_state.error;
    reservation_manager.release(&reservation);
    runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);
    return result;
  }

  runtime_.pd_prefill_kv_cache_manager()->free_request(src_request_id);

  SchedulerConfig decode_config = prefill_config;
  cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
  Scheduler decode_scheduler(decode_config, runtime_.pd_decode_kv_cache_manager());
  const int64_t d_client_id = decode_scheduler.add_decode_ready_request(
      reservation.decode_request_id, prompt_tokens, generation_config,
      reservation.reserved_tokens, first_token);

  while (decode_scheduler.has_active_requests()) {
    cudaSetDevice(runtime_.bench_config().pd.decode_device_id);
    SchedulerOutput output = decode_scheduler.schedule_step();
    if (output.total_tokens == 0) {
      continue;
    }
    MixedBatchMetadata batch = decode_scheduler.build_decode_batch(output, runtime_.pd_decode_stream());
    status = runtime_.pd_forward_decode_batch(batch);
    if (!status) {
      result.failed = true;
      result.error = status.get_err_msg();
      if (runtime_.pd_decode_kv_cache_manager()->is_valid_request(reservation.decode_request_id)) {
        runtime_.pd_decode_kv_cache_manager()->free_request(reservation.decode_request_id);
      }
      reservation = {};
      return result;
    }
    SampledTokenView sampled = runtime_.pd_batch_sample_decode(batch, output);
    for (int32_t i = 0; i < sampled.size(); ++i) {
      if (on_token && !runtime_.is_sentence_ending(sampled.tokens[i])) {
        on_token(sampled.tokens[i]);
      }
    }
    decode_scheduler.process_outputs(
        output, batch, sampled,
        [&](int32_t token) { return runtime_.is_sentence_ending(token); });
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

} // namespace serving
