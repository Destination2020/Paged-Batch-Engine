#ifndef KUIPER_INCLUDE_SERVING_SERVING_ZMQ_RPC_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ZMQ_RPC_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "base/base.h"
#include "serving/generation_config.h"
#include "serving/pd_handoff.h"
#include "serving/serving_config.h"
#include "serving/serving_online_engine.h"

namespace serving {

inline constexpr const char* kOnlineProcessRoleInProc = "inproc";
inline constexpr const char* kOnlineProcessRoleZmqHttpApi = "zmq-http-api";
inline constexpr const char* kOnlineProcessRoleZmqEngineCore = "zmq-engine-core";
inline constexpr const char* kOnlineProcessRoleZmqPrefillEngineCore =
    "zmq-prefill-engine-core";
inline constexpr const char* kOnlineProcessRoleZmqDecodeEngineCore =
    "zmq-decode-engine-core";

enum class ZmqRpcMessageType {
  kUnknown = 0,
  kGenerate = 1,
  kToken = 2,
  kFinal = 3,
  kCancel = 4,
  kMetrics = 5,
  kHealth = 6,
  kPrefill = 7,
  kPrefillResult = 8,
  kKvTransfer = 9,
  kKvTransferResult = 10,
  kKvRelease = 11,
  kPrefillSubmit = 13,
  kPrefillPoll = 14,
};

struct ZmqRpcConfig {
  std::string endpoint = "tcp://127.0.0.1:19090";
  int32_t timeout_ms = 30000;
};

bool is_zmq_online_process_role(const std::string& role);
bool is_zmq_engine_core_role(const std::string& role);
ZmqRpcConfig make_zmq_rpc_config(const BenchConfig& config);
ZmqRpcConfig make_prefill_zmq_rpc_config(const BenchConfig& config);

nlohmann::json generation_config_to_json(const GenerationConfig& config);
GenerationConfig generation_config_from_json(const nlohmann::json& json);
nlohmann::json online_generate_request_to_json(
    const OnlineGenerateRequest& request);
OnlineGenerateRequest online_generate_request_from_json(
    const nlohmann::json& json);

nlohmann::json kv_pool_descriptor_to_json(const KVPoolDescriptor& pool);
KVPoolDescriptor kv_pool_descriptor_from_json(const nlohmann::json& json);
nlohmann::json kv_block_manifest_to_json(const KVBlockManifest& manifest);
KVBlockManifest kv_block_manifest_from_json(const nlohmann::json& json);
nlohmann::json layer_kv_transfer_request_to_json(
    const LayerKVTransferRequest& request);
LayerKVTransferRequest layer_kv_transfer_request_from_json(
    const nlohmann::json& json);
std::string binary_to_hex_json(const std::string& bytes);
std::string hex_json_to_binary(const std::string& hex);

struct RemoteKVBlockPayload {
  int32_t src_block_id = -1;
  std::string key;
  std::string value;
  std::string key_scale;
  std::string value_scale;
};

struct RemoteKVLayerPayload {
  int32_t layer_idx = -1;
  std::vector<int32_t> src_block_ids;
  std::vector<RemoteKVBlockPayload> blocks;
};

struct RemotePrefillResult {
  GlobalRequestId client_request_id;
  HandoffId handoff_id;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> output_tokens;
  std::vector<int32_t> first_tokens;
  bool failed = false;
  std::string error;
  int32_t computed_tokens = 0;
  int32_t first_token = -1;
  KVPoolDescriptor src_pool;
  std::vector<RemoteKVLayerPayload> layers;
};

nlohmann::json remote_prefill_result_to_json(const RemotePrefillResult& result);
RemotePrefillResult remote_prefill_result_from_json(const nlohmann::json& json);

std::string zmq_rpc_message_type_name(ZmqRpcMessageType type);
ZmqRpcMessageType zmq_rpc_message_type_from_string(const std::string& name);

class ZmqSocket {
 public:
  ZmqSocket() = default;
  ZmqSocket(void* socket, void* context);
  ~ZmqSocket();

  ZmqSocket(const ZmqSocket&) = delete;
  ZmqSocket& operator=(const ZmqSocket&) = delete;
  ZmqSocket(ZmqSocket&& other) noexcept;
  ZmqSocket& operator=(ZmqSocket&& other) noexcept;

  bool valid() const;
  base::Status set_linger(int32_t linger_ms);
  base::Status set_timeouts(int32_t timeout_ms);
  base::Status bind(const std::string& endpoint);
  base::Status connect(const std::string& endpoint);
  base::Status send_json(const nlohmann::json& message, int flags = 0);
  base::Status recv_json(nlohmann::json* message, int flags = 0);

 private:
  void close();

  void* socket_ = nullptr;
  void* context_ = nullptr;
};

base::Status make_zmq_req_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket);
base::Status make_zmq_rep_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket);
base::Status zmq_request_response(const ZmqRpcConfig& config,
                                  const nlohmann::json& request,
                                  nlohmann::json* response);

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ZMQ_RPC_H_
