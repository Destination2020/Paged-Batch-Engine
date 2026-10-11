#include "serving/serving_zmq_rpc.h"

#if defined(KUIPER_ENABLE_ZMQ)
#include <zmq.h>
#endif

#include <utility>

#include "base/nvtx_utils.h"

namespace serving {
bool is_zmq_online_process_role(const std::string& role) {
  return role == kOnlineProcessRoleZmqHttpApi ||
         role == kOnlineProcessRoleZmqEngineCore ||
         role == kOnlineProcessRoleZmqDecodeEngineCore ||
         role == kOnlineProcessRoleZmqPrefillEngineCore;
}

bool is_zmq_engine_core_role(const std::string& role) {
  return role == kOnlineProcessRoleZmqEngineCore ||
         role == kOnlineProcessRoleZmqDecodeEngineCore ||
         role == kOnlineProcessRoleZmqPrefillEngineCore;
}

ZmqRpcConfig make_zmq_rpc_config(const BenchConfig& config) {
  ZmqRpcConfig rpc;
  rpc.endpoint = config.ipc.engine_zmq_endpoint;
  rpc.timeout_ms = config.ipc.engine_zmq_timeout_ms;
  return rpc;
}

ZmqRpcConfig make_prefill_zmq_rpc_config(const BenchConfig& config) {
  ZmqRpcConfig rpc;
  rpc.endpoint = config.ipc.prefill_zmq_endpoint;
  rpc.timeout_ms = config.ipc.engine_zmq_timeout_ms;
  return rpc;
}

nlohmann::json generation_config_to_json(const GenerationConfig& config) {
  nlohmann::json json;
  json["max_new_tokens"] = config.max_new_tokens;
  json["min_new_tokens"] = config.min_new_tokens;
  json["ignore_eos"] = config.ignore_eos;
  json["priority"] = config.priority;
  json["sampling"] = {
      {"seed", config.sampling.seed},
      {"temperature", config.sampling.temperature},
      {"top_p", config.sampling.top_p},
      {"top_k", config.sampling.top_k},
      {"repetition_penalty", config.sampling.repetition_penalty},
      {"stop", config.sampling.stop},
  };
  return json;
}

GenerationConfig generation_config_from_json(const nlohmann::json& json) {
  GenerationConfig config;
  config.max_new_tokens = json.value("max_new_tokens", config.max_new_tokens);
  config.min_new_tokens = json.value("min_new_tokens", config.min_new_tokens);
  config.ignore_eos = json.value("ignore_eos", config.ignore_eos);
  config.priority = json.value("priority", config.priority);
  if (json.contains("sampling")) {
    const auto& sampling = json.at("sampling");
    config.sampling.temperature =
        sampling.value("temperature", config.sampling.temperature);
    config.sampling.seed = sampling.value("seed", config.sampling.seed);
    config.sampling.top_p = sampling.value("top_p", config.sampling.top_p);
    config.sampling.top_k = sampling.value("top_k", config.sampling.top_k);
    config.sampling.repetition_penalty =
        sampling.value("repetition_penalty", config.sampling.repetition_penalty);
    if (sampling.contains("stop") && sampling.at("stop").is_array()) {
      config.sampling.stop.clear();
      for (const auto& item : sampling.at("stop")) {
        if (item.is_string()) {
          config.sampling.stop.push_back(item.get<std::string>());
        }
      }
    }
  }
  config.normalize();
  return config;
}

nlohmann::json online_generate_request_to_json(
    const OnlineGenerateRequest& request) {
  nlohmann::json json;
  json["model_id"] = request.model_id;
  json["prompt"] = request.prompt;
  json["generation_config"] = generation_config_to_json(request.generation_config);
  json["stream"] = request.stream;
  json["timeout_ms"] = request.timeout_ms;
  return json;
}

OnlineGenerateRequest online_generate_request_from_json(
    const nlohmann::json& json) {
  OnlineGenerateRequest request;
  request.model_id = json.value("model_id", "");
  request.prompt = json.value("prompt", "");
  if (json.contains("generation_config")) {
    request.generation_config =
        generation_config_from_json(json.at("generation_config"));
  }
  request.stream = json.value("stream", false);
  request.timeout_ms = json.value("timeout_ms", 0);
  return request;
}

nlohmann::json kv_pool_descriptor_to_json(const KVPoolDescriptor& pool) {
  return {
      {"device_id", pool.device_id},
      {"layer_num", pool.layer_num},
      {"block_size", pool.block_size},
      {"kv_head_num", pool.kv_head_num},
      {"head_size", pool.head_size},
      {"dtype", static_cast<int32_t>(pool.dtype)},
      {"storage_mode", static_cast<int32_t>(pool.storage_mode)},
  };
}

KVPoolDescriptor kv_pool_descriptor_from_json(const nlohmann::json& json) {
  KVPoolDescriptor pool;
  pool.device_id = json.value("device_id", pool.device_id);
  pool.layer_num = json.value("layer_num", pool.layer_num);
  pool.block_size = json.value("block_size", pool.block_size);
  pool.kv_head_num = json.value("kv_head_num", pool.kv_head_num);
  pool.head_size = json.value("head_size", pool.head_size);
  pool.dtype = static_cast<base::DataType>(
      json.value("dtype", static_cast<int32_t>(pool.dtype)));
  pool.storage_mode = static_cast<base::BlockStorageMode>(
      json.value("storage_mode", static_cast<int32_t>(pool.storage_mode)));
  return pool;
}

nlohmann::json kv_block_manifest_to_json(const KVBlockManifest& manifest) {
  nlohmann::json json;
  json["client_request_id"] = manifest.client_request_id.value;
  json["handoff_id"] = manifest.handoff_id.value;
  json["prompt_tokens"] = manifest.prompt_tokens;
  json["computed_tokens"] = manifest.computed_tokens;
  json["first_token"] = manifest.first_token;
  json["src_pool"] = kv_pool_descriptor_to_json(manifest.src_pool);
  json["dst_pool"] = kv_pool_descriptor_to_json(manifest.dst_pool);
  json["layer_mappings"] = nlohmann::json::array();
  for (const auto& mapping : manifest.layer_mappings) {
    json["layer_mappings"].push_back({
        {"layer_idx", mapping.layer_idx},
        {"src_block_ids", mapping.src_block_ids},
        {"dst_block_ids", mapping.dst_block_ids},
    });
  }
  return json;
}

KVBlockManifest kv_block_manifest_from_json(const nlohmann::json& json) {
  KVBlockManifest manifest;
  manifest.client_request_id.value =
      json.value("client_request_id", std::string());
  manifest.handoff_id.value = json.value("handoff_id", uint64_t{0});
  manifest.prompt_tokens = json.value("prompt_tokens", manifest.prompt_tokens);
  manifest.computed_tokens =
      json.value("computed_tokens", manifest.computed_tokens);
  manifest.first_token = json.value("first_token", manifest.first_token);
  if (json.contains("src_pool")) {
    manifest.src_pool = kv_pool_descriptor_from_json(json.at("src_pool"));
  }
  if (json.contains("dst_pool")) {
    manifest.dst_pool = kv_pool_descriptor_from_json(json.at("dst_pool"));
  }
  if (json.contains("layer_mappings") && json.at("layer_mappings").is_array()) {
    for (const auto& mapping_json : json.at("layer_mappings")) {
      KVBlockMapping mapping;
      mapping.layer_idx = mapping_json.value("layer_idx", mapping.layer_idx);
      if (mapping_json.contains("src_block_ids")) {
        mapping.src_block_ids =
            mapping_json.at("src_block_ids").get<std::vector<int32_t>>();
      }
      if (mapping_json.contains("dst_block_ids")) {
        mapping.dst_block_ids =
            mapping_json.at("dst_block_ids").get<std::vector<int32_t>>();
      }
      manifest.layer_mappings.push_back(std::move(mapping));
    }
  }
  return manifest;
}

nlohmann::json layer_kv_transfer_request_to_json(
    const LayerKVTransferRequest& request) {
  nlohmann::json json;
  json["client_request_id"] = request.client_request_id.value;
  json["handoff_id"] = request.handoff_id.value;
  json["src_request_id"] = request.src_request_id;
  json["dst_request_id"] = request.dst_request_id;
  json["prompt_tokens"] = request.prompt_tokens;
  json["computed_tokens"] = request.computed_tokens;
  json["src_pool"] = kv_pool_descriptor_to_json(request.src_pool);
  json["dst_pool"] = kv_pool_descriptor_to_json(request.dst_pool);
  return json;
}

LayerKVTransferRequest layer_kv_transfer_request_from_json(
    const nlohmann::json& json) {
  LayerKVTransferRequest request;
  request.client_request_id.value =
      json.value("client_request_id", std::string());
  request.handoff_id.value = json.value("handoff_id", uint64_t{0});
  request.src_request_id = json.value("src_request_id", request.src_request_id);
  request.dst_request_id = json.value("dst_request_id", request.dst_request_id);
  request.prompt_tokens = json.value("prompt_tokens", request.prompt_tokens);
  request.computed_tokens =
      json.value("computed_tokens", request.computed_tokens);
  if (json.contains("src_pool")) {
    request.src_pool = kv_pool_descriptor_from_json(json.at("src_pool"));
  }
  if (json.contains("dst_pool")) {
    request.dst_pool = kv_pool_descriptor_from_json(json.at("dst_pool"));
  }
  return request;
}

nlohmann::json remote_prefill_result_to_json(const RemotePrefillResult& result) {
  nlohmann::json json;
  json["client_request_id"] = result.client_request_id.value;
  json["handoff_id"] = result.handoff_id.value;
  json["prompt_tokens"] = result.prompt_tokens;
  json["output_tokens"] = result.output_tokens;
  json["first_tokens"] = result.first_tokens;
  json["failed"] = result.failed;
  json["error"] = result.error;
  json["computed_tokens"] = result.computed_tokens;
  json["first_token"] = result.first_token;
  json["src_pool"] = kv_pool_descriptor_to_json(result.src_pool);
  json["layers"] = nlohmann::json::array();
  for (const auto& layer : result.layers) {
    nlohmann::json layer_json;
    layer_json["layer_idx"] = layer.layer_idx;
    layer_json["src_block_ids"] = layer.src_block_ids;
    layer_json["blocks"] = nlohmann::json::array();
    for (const auto& block : layer.blocks) {
      layer_json["blocks"].push_back({
          {"src_block_id", block.src_block_id},
          {"key", binary_to_hex_json(block.key)},
          {"value", binary_to_hex_json(block.value)},
          {"key_scale", binary_to_hex_json(block.key_scale)},
          {"value_scale", binary_to_hex_json(block.value_scale)},
      });
    }
    json["layers"].push_back(std::move(layer_json));
  }
  return json;
}

RemotePrefillResult remote_prefill_result_from_json(const nlohmann::json& json) {
  RemotePrefillResult result;
  result.client_request_id.value =
      json.value("client_request_id", std::string());
  result.handoff_id.value = json.value("handoff_id", uint64_t{0});
  if (json.contains("prompt_tokens")) {
    result.prompt_tokens = json.at("prompt_tokens").get<std::vector<int32_t>>();
  }
  if (json.contains("output_tokens")) {
    result.output_tokens = json.at("output_tokens").get<std::vector<int32_t>>();
  }
  if (json.contains("first_tokens")) {
    result.first_tokens = json.at("first_tokens").get<std::vector<int32_t>>();
  }
  result.failed = json.value("failed", result.failed);
  result.error = json.value("error", result.error);
  result.computed_tokens = json.value("computed_tokens", result.computed_tokens);
  result.first_token = json.value("first_token", result.first_token);
  if (json.contains("src_pool")) {
    result.src_pool = kv_pool_descriptor_from_json(json.at("src_pool"));
  }
  if (json.contains("layers") && json.at("layers").is_array()) {
    for (const auto& layer_json : json.at("layers")) {
      RemoteKVLayerPayload layer;
      layer.layer_idx = layer_json.value("layer_idx", layer.layer_idx);
      if (layer_json.contains("src_block_ids")) {
        layer.src_block_ids =
            layer_json.at("src_block_ids").get<std::vector<int32_t>>();
      }
      if (layer_json.contains("blocks") && layer_json.at("blocks").is_array()) {
        for (const auto& block_json : layer_json.at("blocks")) {
          RemoteKVBlockPayload block;
          block.src_block_id = block_json.value("src_block_id", block.src_block_id);
          block.key = hex_json_to_binary(block_json.value("key", std::string()));
          block.value = hex_json_to_binary(block_json.value("value", std::string()));
          block.key_scale =
              hex_json_to_binary(block_json.value("key_scale", std::string()));
          block.value_scale =
              hex_json_to_binary(block_json.value("value_scale", std::string()));
          layer.blocks.push_back(std::move(block));
        }
      }
      result.layers.push_back(std::move(layer));
    }
  }
  return result;
}

std::string zmq_rpc_message_type_name(ZmqRpcMessageType type) {
  switch (type) {
    case ZmqRpcMessageType::kGenerate:
      return "generate";
    case ZmqRpcMessageType::kToken:
      return "token";
    case ZmqRpcMessageType::kFinal:
      return "final";
    case ZmqRpcMessageType::kCancel:
      return "cancel";
    case ZmqRpcMessageType::kMetrics:
      return "metrics";
    case ZmqRpcMessageType::kHealth:
      return "health";
    case ZmqRpcMessageType::kPrefill:
      return "prefill";
    case ZmqRpcMessageType::kPrefillResult:
      return "prefill_result";
    case ZmqRpcMessageType::kKvTransfer:
      return "kv_transfer";
    case ZmqRpcMessageType::kKvTransferResult:
      return "kv_transfer_result";
    case ZmqRpcMessageType::kKvRelease:
      return "kv_release";
    case ZmqRpcMessageType::kPrefillSubmit:
      return "prefill_submit";
    case ZmqRpcMessageType::kPrefillPoll:
      return "prefill_poll";
    case ZmqRpcMessageType::kUnknown:
    default:
      return "unknown";
  }
}

ZmqRpcMessageType zmq_rpc_message_type_from_string(const std::string& name) {
  if (name == "generate") return ZmqRpcMessageType::kGenerate;
  if (name == "token") return ZmqRpcMessageType::kToken;
  if (name == "final") return ZmqRpcMessageType::kFinal;
  if (name == "cancel") return ZmqRpcMessageType::kCancel;
  if (name == "metrics") return ZmqRpcMessageType::kMetrics;
  if (name == "health") return ZmqRpcMessageType::kHealth;
  if (name == "prefill") return ZmqRpcMessageType::kPrefill;
  if (name == "prefill_result") return ZmqRpcMessageType::kPrefillResult;
  if (name == "kv_transfer") return ZmqRpcMessageType::kKvTransfer;
  if (name == "kv_transfer_result") return ZmqRpcMessageType::kKvTransferResult;
  if (name == "kv_release") return ZmqRpcMessageType::kKvRelease;
  if (name == "prefill_submit") return ZmqRpcMessageType::kPrefillSubmit;
  if (name == "prefill_poll") return ZmqRpcMessageType::kPrefillPoll;
  return ZmqRpcMessageType::kUnknown;
}

} // namespace serving
