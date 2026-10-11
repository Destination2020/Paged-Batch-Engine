#include "serving/serving_http_protocol.h"
#include "serving/serving_stop_signal.h"
#include "base/socket_io.h"
#include <charconv>
#include <cctype>
#include <sstream>
#include <string_view>

namespace serving {
namespace {
using namespace base::socket_io;
thread_local int io_timeout_ms = 5000;
thread_local Deadline request_deadline = Deadline::max();
constexpr size_t kMaxHeaders = 64 * 1024;
constexpr size_t kMaxBody = 8 * 1024 * 1024;
std::string_view trim(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.remove_suffix(1);
  return value;
}
}
void configure_http_io(int timeout_ms) {
  io_timeout_ms = std::max(1, timeout_ms);
  request_deadline = Deadline::max();
}
void set_http_request_deadline(int timeout_ms) {
  request_deadline = timeout_ms > 0 ? Clock::now() + std::chrono::milliseconds(timeout_ms)
                                   : Deadline::max();
}
bool send_all(int fd, const std::string& data) {
  return base::socket_io::send(fd, data.data(), data.size(),
      std::min(request_deadline, Clock::now() + std::chrono::milliseconds(io_timeout_ms)));
}
bool send_http_response(int fd, int status, const std::string& type, const std::string& body) {
  const char* reason = status == 200 ? "OK" : status == 400 ? "Bad Request" :
      status == 404 ? "Not Found" : status == 405 ? "Method Not Allowed" :
      status == 408 ? "Request Timeout" : status == 413 ? "Content Too Large" :
      status == 429 ? "Too Many Requests" : status == 503 ? "Service Unavailable" : "Internal Server Error";
  std::ostringstream output;
  output << "HTTP/1.1 " << status << " " << reason << "\r\nContent-Type: " << type
         << "\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n" << body;
  return send_all(fd, output.str());
}
int read_http_request(int fd, std::string* request) {
  request->clear();
  const auto deadline = Clock::now() + std::chrono::milliseconds(io_timeout_ms);
  size_t total = 0;
  bool parsed = false;
  while (!serving_stop_requested) {
    if (Clock::now() >= deadline) return 408;
    // Periodically observe shutdown while receiving an incomplete request.
    if (!base::socket_io::wait(fd, POLLIN, std::min(deadline, Clock::now() + std::chrono::milliseconds(100)))) {
      if (errno == ETIMEDOUT) continue;
      return 400;
    }
    char buffer[4096];
    const auto n = ::recv(fd, buffer, sizeof(buffer), MSG_DONTWAIT);
    if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (n <= 0) return 400;
    request->append(buffer, static_cast<size_t>(n));
    if (!parsed) {
      const auto end = request->find("\r\n\r\n");
      if (end == std::string::npos) {
        if (request->size() > kMaxHeaders) return 413;
        continue;
      }
      if (end + 4 > kMaxHeaders) return 413;
      size_t length = 0;
      bool seen_length = false;
      auto pos = request->find("\r\n");
      while (pos < end) {
        const auto next = request->find("\r\n", pos + 2);
        const std::string_view line(request->data() + pos + 2, next - pos - 2);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos) return 400;
        std::string key(line.substr(0, colon));
        for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const auto value = trim(line.substr(colon + 1));
        if (key == "transfer-encoding") return 400; // chunked uploads are unsupported
        if (key == "content-length") {
          if (seen_length || value.empty()) return 400;
          seen_length = true;
          const auto result = std::from_chars(value.data(), value.data() + value.size(), length);
          if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return 400;
          if (length > kMaxBody) return 413;
        }
        pos = next;
      }
      total = end + 4 + length;
      parsed = true;
    }
    if (request->size() >= total) { request->resize(total); return 200; }
  }
  return 503;
}
}  // namespace serving
