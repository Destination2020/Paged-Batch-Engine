#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "base/base.h"

namespace base::transport {
// Each socket has one I/O owner, or externally serialized access.
struct ZmqRpcConfig {
  std::string endpoint = "tcp://127.0.0.1:19090";
  int32_t timeout_ms = 30000;
};

class ZmqSocket {
 public:
  ZmqSocket() = default;
  ZmqSocket(void* socket, std::shared_ptr<void> context);
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
  std::shared_ptr<void> context_;
};

base::Status make_zmq_req_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket);
base::Status make_zmq_rep_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket);
base::Status zmq_request_response(const ZmqRpcConfig& config,
                                  const nlohmann::json& request,
                                  nlohmann::json* response);

std::string binary_to_hex_json(const std::string& bytes);
std::string hex_json_to_binary(const std::string& hex);
} // namespace base::transport
