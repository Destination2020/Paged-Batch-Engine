#include "base/zmq_transport.h"
#include "base/nvtx_utils.h"
#include <utility>
#if defined(KUIPER_ENABLE_ZMQ)
#include <zmq.h>
#endif
namespace base::transport {
namespace {
#if defined(KUIPER_ENABLE_ZMQ)
// Shared by sockets, never by forked children: serving workers are exec-spawned.
// ZMQ contexts are thread-safe; socket ownership/serialization remains unchanged.
std::shared_ptr<void> process_context() {
  static const auto context = std::shared_ptr<void>(zmq_ctx_new(), [](void* value) {
    if (value) zmq_ctx_term(value);
  });
  return context;
}
#endif


base::Status zmq_unavailable_status() {
  return base::Status(
      base::StatusCode::kFunctionUnImplement,
      "ZMQ support is not enabled. Install libzmq development headers and "
      "reconfigure CMake with KUIPER_ENABLE_ZMQ=ON.");
}

#if defined(KUIPER_ENABLE_ZMQ)
base::Status zmq_error_status(const std::string& action) {
  return base::Status(base::StatusCode::kInternalError,
                      action + " failed: " + zmq_strerror(zmq_errno()));
}
#endif

std::string bytes_to_hex(const std::string& bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    hex.push_back(kHex[(byte >> 4) & 0x0f]);
    hex.push_back(kHex[byte & 0x0f]);
  }
  return hex;
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string hex_to_bytes(const std::string& hex) {
  if (hex.size() % 2 != 0) {
    return {};
  }
  std::string bytes;
  bytes.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    const int hi = hex_value(hex[i]);
    const int lo = hex_value(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      return {};
    }
    bytes.push_back(static_cast<char>((hi << 4) | lo));
  }
  return bytes;
}

}  // namespace

std::string binary_to_hex_json(const std::string& bytes) {
  return bytes_to_hex(bytes);
}

std::string hex_json_to_binary(const std::string& hex) {
  return hex_to_bytes(hex);
}

ZmqSocket::ZmqSocket(void* socket, std::shared_ptr<void> context)
    : socket_(socket), context_(std::move(context)) {}

ZmqSocket::~ZmqSocket() { close(); }

ZmqSocket::ZmqSocket(ZmqSocket&& other) noexcept
    : socket_(other.socket_), context_(other.context_) {
  other.socket_ = nullptr;
  other.context_ = nullptr;
}

ZmqSocket& ZmqSocket::operator=(ZmqSocket&& other) noexcept {
  if (this != &other) {
    close();
    socket_ = other.socket_;
    context_ = other.context_;
    other.socket_ = nullptr;
    other.context_ = nullptr;
  }
  return *this;
}

bool ZmqSocket::valid() const { return socket_ != nullptr && context_ != nullptr; }

base::Status ZmqSocket::set_linger(int32_t linger_ms) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  if (zmq_setsockopt(socket_, ZMQ_LINGER, &linger_ms, sizeof(linger_ms)) != 0) {
    return zmq_error_status("zmq_setsockopt(ZMQ_LINGER)");
  }
  return {};
#else
  (void)linger_ms;
  return zmq_unavailable_status();
#endif
}

base::Status ZmqSocket::set_timeouts(int32_t timeout_ms) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  if (zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0) {
    return zmq_error_status("zmq_setsockopt(ZMQ_SNDTIMEO)");
  }
  if (zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) != 0) {
    return zmq_error_status("zmq_setsockopt(ZMQ_RCVTIMEO)");
  }
  return {};
#else
  (void)timeout_ms;
  return zmq_unavailable_status();
#endif
}

base::Status ZmqSocket::bind(const std::string& endpoint) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  if (zmq_bind(socket_, endpoint.c_str()) != 0) {
    return zmq_error_status("zmq_bind(" + endpoint + ")");
  }
  return {};
#else
  (void)endpoint;
  return zmq_unavailable_status();
#endif
}

base::Status ZmqSocket::connect(const std::string& endpoint) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  if (zmq_connect(socket_, endpoint.c_str()) != 0) {
    return zmq_error_status("zmq_connect(" + endpoint + ")");
  }
  return {};
#else
  (void)endpoint;
  return zmq_unavailable_status();
#endif
}

base::Status ZmqSocket::send_json(const nlohmann::json& message, int flags) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  std::string serialized;
  try { serialized = message.dump(); }
  catch (const nlohmann::json::exception& e) {
    return base::error::InvalidArgument(std::string("Invalid JSON payload: ") + e.what());
  }
  const int rc = zmq_send(socket_, serialized.data(), serialized.size(), flags);
  if (rc < 0) {
    return zmq_error_status("zmq_send");
  }
  return {};
#else
  (void)message;
  (void)flags;
  return zmq_unavailable_status();
#endif
}

base::Status ZmqSocket::recv_json(nlohmann::json* message, int flags) {
#if defined(KUIPER_ENABLE_ZMQ)
  if (!valid()) return zmq_unavailable_status();
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  const int rc = zmq_msg_recv(&msg, socket_, flags);
  if (rc < 0) {
    zmq_msg_close(&msg);
    return zmq_error_status("zmq_msg_recv");
  }
  const char* data = static_cast<const char*>(zmq_msg_data(&msg));
  const size_t size = zmq_msg_size(&msg);
  try {
    *message = nlohmann::json::parse(std::string(data, size));
  } catch (const std::exception& e) {
    zmq_msg_close(&msg);
    return base::Status(base::StatusCode::kInvalidArgument,
                        std::string("invalid ZMQ JSON: ") + e.what());
  }
  zmq_msg_close(&msg);
  return {};
#else
  (void)message;
  (void)flags;
  return zmq_unavailable_status();
#endif
}

void ZmqSocket::close() {
#if defined(KUIPER_ENABLE_ZMQ)
  if (socket_ != nullptr) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
  if (context_ != nullptr) {
    context_.reset();
  }
#else
  socket_ = nullptr;
  context_ = nullptr;
#endif
}

base::Status make_zmq_req_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket) {
#if defined(KUIPER_ENABLE_ZMQ)
  auto context = process_context();
  if (context == nullptr) {
    return zmq_error_status("zmq_ctx_new");
  }
  void* raw_socket = zmq_socket(context.get(), ZMQ_REQ);
  if (raw_socket == nullptr) {
    return zmq_error_status("zmq_socket(ZMQ_REQ)");
  }
  auto wrapper = std::make_unique<ZmqSocket>(raw_socket, context);
  auto status = wrapper->set_linger(0);
  if (!status) return status;
  status = wrapper->set_timeouts(config.timeout_ms);
  if (!status) return status;
  status = wrapper->connect(config.endpoint);
  if (!status) return status;
  *socket = std::move(wrapper);
  return {};
#else
  (void)config;
  (void)socket;
  return zmq_unavailable_status();
#endif
}

base::Status make_zmq_rep_socket(const ZmqRpcConfig& config,
                                 std::unique_ptr<ZmqSocket>* socket) {
#if defined(KUIPER_ENABLE_ZMQ)
  auto context = process_context();
  if (context == nullptr) {
    return zmq_error_status("zmq_ctx_new");
  }
  void* raw_socket = zmq_socket(context.get(), ZMQ_REP);
  if (raw_socket == nullptr) {
    return zmq_error_status("zmq_socket(ZMQ_REP)");
  }
  auto wrapper = std::make_unique<ZmqSocket>(raw_socket, context);
  auto status = wrapper->set_linger(0);
  if (!status) return status;
  status = wrapper->bind(config.endpoint);
  if (!status) return status;
  *socket = std::move(wrapper);
  return {};
#else
  (void)config;
  (void)socket;
  return zmq_unavailable_status();
#endif
}

base::Status zmq_request_response(const ZmqRpcConfig& config,
                                  const nlohmann::json& request,
                                  nlohmann::json* response) {
#if defined(KUIPER_ENABLE_ZMQ)
  const std::string type = request.value("type", std::string("unknown"));
  base::nvtx::ScopedRange range("zmq_rpc:" + type,
                                base::nvtx::kColorProcess);
  std::unique_ptr<ZmqSocket> socket;
  auto status = make_zmq_req_socket(config, &socket);
  if (!status) {
    return status;
  }
  status = socket->send_json(request);
  if (!status) {
    return status;
  }
  return socket->recv_json(response);
#else
  (void)config;
  (void)request;
  (void)response;
  return zmq_unavailable_status();
#endif
}

} // namespace base::transport
