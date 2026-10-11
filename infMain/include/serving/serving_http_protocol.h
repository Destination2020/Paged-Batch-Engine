#pragma once
#include <string>

namespace serving {
// Per-handler limits. A worker handles one connection at a time.
void configure_http_io(int timeout_ms);
void set_http_request_deadline(int timeout_ms);
bool send_all(int fd, const std::string& data);
bool send_http_response(int fd, int status, const std::string& content_type,
                        const std::string& body);
// Returns HTTP status on failure; bounds both headers and body independently.
int read_http_request(int fd, std::string* request);
}  // namespace serving
