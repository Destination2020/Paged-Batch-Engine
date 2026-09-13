#include "serving/serving_online_server.h"

#include <glog/logging.h>
#include <sys/socket.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "serving/generation_config.h"
#include "serving/scheduler.h"
#include "serving/serving_benchmark_app.h"
#include "serving/serving_config.h"
#include "serving/serving_http_server.h"
#include "serving/serving_online_engine_pool.h"
#include "serving/serving_zmq_rpc.h"

namespace serving {
namespace {
std::string http_reason(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

bool send_all(int fd, const std::string& data) {
  const char* ptr = data.data();
  size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t written = ::send(fd, ptr, remaining, MSG_NOSIGNAL);
    if (written <= 0) {
      return false;
    }
    ptr += written;
    remaining -= static_cast<size_t>(written);
  }
  return true;
}

bool send_http_response(int fd,
                        int status,
                        const std::string& content_type,
                        const std::string& body) {
  std::ostringstream os;
  os << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\n"
     << "Content-Type: " << content_type << "\r\n"
     << "Content-Length: " << body.size() << "\r\n"
     << "Connection: close\r\n\r\n"
     << body;
  return send_all(fd, os.str());
}

std::string json_error(const std::string& message) {
  nlohmann::json body;
  body["error"] = message;
  return body.dump();
}

bool read_http_request(int fd, std::string* request) {
  CHECK_NE(request, nullptr);
  request->clear();
  char buffer[4096];
  size_t content_length = 0;
  size_t header_end = std::string::npos;
  while (request->size() < 8 * 1024 * 1024) {
    const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
    if (n <= 0) {
      return false;
    }
    request->append(buffer, static_cast<size_t>(n));
    header_end = request->find("\r\n\r\n");
    if (header_end != std::string::npos) {
      const std::string headers = request->substr(0, header_end + 4);
      std::istringstream input(headers);
      std::string line;
      while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        const std::string key = "Content-Length:";
        if (line.size() >= key.size() &&
            std::equal(key.begin(), key.end(), line.begin(),
                       [](char a, char b) {
                         return std::tolower(a) == std::tolower(b);
                       })) {
          content_length = static_cast<size_t>(std::stoll(line.substr(key.size())));
        }
      }
      const size_t body_received = request->size() - (header_end + 4);
      if (body_received >= content_length) {
        return true;
      }
    }
  }
  return false;
}

std::string http_body(const std::string& request) {
  const size_t header_end = request.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return {};
  }
  return request.substr(header_end + 4);
}

std::string http_method(const std::string& request) {
  const size_t space = request.find(' ');
  return space == std::string::npos ? std::string{} : request.substr(0, space);
}

std::string http_path(const std::string& request) {
  const size_t first = request.find(' ');
  if (first == std::string::npos) {
    return {};
  }
  const size_t second = request.find(' ', first + 1);
  if (second == std::string::npos) {
    return {};
  }
  return request.substr(first + 1, second - first - 1);
}

int32_t read_int_with_fallback(const nlohmann::json& payload,
                               const char* primary_key,
                               const char* fallback_key,
                               int32_t default_value) {
  if (payload.contains(primary_key)) {
    return payload.value(primary_key, default_value);
  }
  if (payload.contains(fallback_key)) {
    return payload.value(fallback_key, default_value);
  }
  return default_value;
}

double read_double_with_default(const nlohmann::json& payload,
                                const char* key,
                                double default_value) {
  return payload.contains(key) ? payload.value(key, default_value) : default_value;
}

std::vector<std::string> read_stop_strings(const nlohmann::json& payload) {
  std::vector<std::string> stops;
  if (!payload.contains("stop")) {
    return stops;
  }
  const auto& value = payload.at("stop");
  if (value.is_string()) {
    stops.push_back(value.get<std::string>());
  } else if (value.is_array()) {
    for (const auto& item : value) {
      if (item.is_string()) {
        stops.push_back(item.get<std::string>());
      }
    }
  }
  return stops;
}

std::string build_chat_prompt_from_messages(const nlohmann::json& messages) {
  if (!messages.is_array()) {
    return {};
  }
  std::ostringstream prompt;
  for (const auto& message : messages) {
    const std::string role = message.value("role", "user");
    const std::string content = message.value("content", "");
    prompt << "<|im_start|>" << role << "\n" << content << "\n<|im_end|>\n";
  }
  prompt << "<|im_start|>assistant\n";
  return prompt.str();
}

OnlineGenerateRequest parse_online_generate_request(const nlohmann::json& payload) {
  OnlineGenerateRequest request;
  request.prompt = payload.contains("messages")
      ? build_chat_prompt_from_messages(payload.at("messages"))
      : payload.value("prompt", "");
  int32_t max_new_tokens = read_int_with_fallback(
      payload, "max_new_tokens_online", "max_new_tokens", 128);
  if (payload.contains("max_tokens") &&
      !payload.contains("max_new_tokens_online") && !payload.contains("max_new_tokens")) {
    max_new_tokens = payload.value("max_tokens", max_new_tokens);
  }
  request.generation_config = GenerationConfig(
      max_new_tokens,
      read_int_with_fallback(payload, "min_new_tokens_online", "min_new_tokens", 0),
      payload.value("ignore_eos", false),
      payload.value("priority", 0));
  auto& sampling = request.generation_config.sampling;
  sampling.temperature = read_double_with_default(payload, "temperature", sampling.temperature);
  sampling.seed = payload.value("seed", sampling.seed);
  sampling.top_p = read_double_with_default(payload, "top_p", sampling.top_p);
  sampling.top_k = read_int_with_fallback(payload, "top_k", "top_k", sampling.top_k);
  sampling.repetition_penalty = read_double_with_default(
      payload, "repetition_penalty", sampling.repetition_penalty);
  sampling.stop = read_stop_strings(payload);
  request.generation_config.normalize();
  request.stream = payload.value("stream", false);
  request.timeout_ms = payload.value("timeout_ms", 0);
  return request;
}

nlohmann::json make_completion_response(const std::string& text, bool chat) {
  nlohmann::json response;
  response["id"] = "cmpl-paged-batch-engine";
  response["object"] = chat ? "chat.completion" : "text_completion";
  response["model"] = "paged-batch-engine";
  response["choices"] = nlohmann::json::array();
  nlohmann::json choice;
  choice["index"] = 0;
  choice["finish_reason"] = "stop";
  if (chat) {
    choice["message"] = {{"role", "assistant"}, {"content", text}};
  } else {
    choice["text"] = text;
  }
  response["choices"].push_back(choice);
  response["usage"] = {{"prompt_tokens", nullptr}, {"completion_tokens", nullptr}, {"total_tokens", nullptr}};
  return response;
}

void handle_generate_request(int client_fd,
                             OnlineEnginePool* engine_pool,
                             const std::string& request,
                             const std::string& path) {
  nlohmann::json payload;
  try {
    payload = nlohmann::json::parse(http_body(request));
  } catch (const std::exception& e) {
    send_http_response(client_fd, 400, "application/json", json_error(e.what()));
    return;
  }

  const bool openai_completion = path == "/v1/completions" || path == "/v1/chat/completions";
  const bool chat_completion = path == "/v1/chat/completions";
  OnlineGenerateRequest generate_request = parse_online_generate_request(payload);
  if (generate_request.prompt.empty()) {
    send_http_response(client_fd, 400, "application/json", json_error("prompt is required"));
    return;
  }

  std::string error;
  auto handle = engine_pool->submit(generate_request, &error);
  if (!handle) {
    const int status = error == "queue_full" ? 429 : (error == "prompt_too_long" ? 400 : 500);
    send_http_response(client_fd, status, "application/json", json_error(error));
    return;
  }

  if (!generate_request.stream) {
    bool failed = false;
    std::string finish_error;
    std::string text;
    const int32_t timeout_ms = generate_request.timeout_ms > 0
        ? generate_request.timeout_ms
        : engine_pool->default_timeout_ms();
    if (!handle->wait_full_text(timeout_ms, &text, &failed, &finish_error)) {
      engine_pool->cancel(handle, "request_timeout");
      send_http_response(client_fd, 408, "application/json", json_error("request_timeout"));
      return;
    }
    if (failed) {
      send_http_response(client_fd, 500, "application/json", json_error(finish_error));
      return;
    }
    if (openai_completion) {
      send_http_response(client_fd, 200, "application/json",
                         make_completion_response(text, chat_completion).dump());
    } else {
      nlohmann::json response;
      response["text"] = text;
      send_http_response(client_fd, 200, "application/json", response.dump());
    }
    return;
  }

  std::ostringstream headers;
  headers << "HTTP/1.1 200 OK\r\n"
          << "Content-Type: text/event-stream\r\n"
          << "Cache-Control: no-cache\r\n"
          << "Connection: close\r\n\r\n";
  if (!send_all(client_fd, headers.str())) {
    engine_pool->cancel(handle, "client_disconnected");
    return;
  }

  while (true) {
    std::string token_text;
    bool finished = false;
    bool failed = false;
    std::string finish_error;
    const bool has_token = handle->wait_next_token(&token_text, &finished, &failed,
                                                   &finish_error);
    if (has_token) {
      nlohmann::json event;
      if (openai_completion) {
        event["id"] = "cmpl-paged-batch-engine";
        event["object"] = chat_completion ? "chat.completion.chunk" : "text_completion";
        event["model"] = "paged-batch-engine";
        event["choices"] = nlohmann::json::array();
        nlohmann::json delta;
        delta["index"] = 0;
        delta["finish_reason"] = nullptr;
        if (chat_completion) {
          delta["delta"] = {{"content", token_text}};
        } else {
          delta["text"] = token_text;
        }
        event["choices"].push_back(delta);
      } else {
        event["text"] = token_text;
      }
      if (!send_all(client_fd, "data: " + event.dump() + "\n\n")) {
        engine_pool->cancel(handle, "client_disconnected");
        return;
      }
      continue;
    }
    if (finished) {
      if (failed) {
        nlohmann::json event;
        event["error"] = finish_error;
        send_all(client_fd, "event: error\ndata: " + event.dump() + "\n\n");
      } else {
        send_all(client_fd, "data: [DONE]\n\n");
      }
      return;
    }
  }
}

void handle_http_client(int client_fd, OnlineEnginePool* engine_pool) {
  std::string request;
  if (!read_http_request(client_fd, &request)) {
    send_http_response(client_fd, 408, "application/json", json_error("read request failed"));
    return;
  }

  const std::string method = http_method(request);
  const std::string path = http_path(request);
  if (method == "GET" && path == "/health") {
    send_http_response(client_fd, 200, "application/json", "{\"status\":\"ok\"}");
    return;
  }
  if (method == "GET" && path == "/metrics") {
    send_http_response(client_fd, 200, "application/json", engine_pool->metrics_json().dump());
    return;
  }
  if (method != "POST") {
    send_http_response(client_fd, 405, "application/json", json_error("method not allowed"));
    return;
  }
  if (path != "/generate" && path != "/v1/completions" &&
      path != "/v1/chat/completions") {
    send_http_response(client_fd, 404, "application/json", json_error("not found"));
    return;
  }
  handle_generate_request(client_fd, engine_pool, request, path);
}



}  // namespace

int run_online_server(ServingBenchmarkApp* app) {
  CHECK_NE(app, nullptr);
  return run_online_server(app, app->bench_config());
}

int run_online_server(ServingBenchmarkApp* app, const BenchConfig& config) {
  std::unique_ptr<OnlineEnginePool> engine_pool;
  if (config.online_process_role == kOnlineProcessRoleZmqHttpApi) {
    engine_pool = std::make_unique<ZmqOnlineEnginePool>(config);
  } else {
    CHECK_NE(app, nullptr);
    engine_pool = std::make_unique<SingleOnlineEnginePool>(app, config);
  }
  engine_pool->start();

  HttpServerConfig http_config;
  http_config.host = config.listen_host;
  http_config.port = config.listen_port;
  http_config.listen_backlog = config.http_listen_backlog;
  http_config.worker_threads = config.http_worker_threads;

  EpollHttpServer server(
      http_config,
      [&engine_pool](int client_fd) { handle_http_client(client_fd, engine_pool.get()); },
      [&config]() {
        LOG(INFO) << "Online serving listening on http://" << config.listen_host
                  << ":" << config.listen_port
                  << " endpoints: GET /health, GET /metrics, POST /generate, "
                  << "POST /v1/completions, POST /v1/chat/completions"
                  << " http_workers=" << config.http_worker_threads
                  << " listen_backlog=" << config.http_listen_backlog
                  << " process_role=" << config.online_process_role
                  << " engine_zmq_endpoint=" << config.engine_zmq_endpoint
                  << " prefill_zmq_endpoint=" << config.prefill_zmq_endpoint;
        std::cout << "ONLINE_SERVER_READY host=" << config.listen_host
                  << " port=" << config.listen_port
                  << " role=" << config.online_process_role
                  << " engine_zmq_endpoint=" << config.engine_zmq_endpoint
                  << " prefill_zmq_endpoint=" << config.prefill_zmq_endpoint
                  << std::endl;
      });
  return server.run();
}

}  // namespace serving
