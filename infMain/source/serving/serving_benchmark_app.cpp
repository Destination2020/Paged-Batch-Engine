#include "serving/serving_benchmark_app.h"
#include <glog/logging.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "base/nvtx_utils.h"

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

class OnlineRequestHandle {
 public:
  void set_request_id(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    request_id_ = request_id;
  }

  int64_t request_id() const {
    std::lock_guard<std::mutex> lock(mu_);
    return request_id_;
  }

  void mark_cancelled() {
    std::lock_guard<std::mutex> lock(mu_);
    cancelled_ = true;
  }

  bool cancelled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cancelled_;
  }

  void push_token(const std::string& token_text) {
    std::lock_guard<std::mutex> lock(mu_);
    token_texts_.push_back(token_text);
    cv_.notify_all();
  }

  void finish(std::string full_text, bool failed, std::string error) {
    std::lock_guard<std::mutex> lock(mu_);
    full_text_ = std::move(full_text);
    failed_ = failed;
    error_ = std::move(error);
    finished_ = true;
    cv_.notify_all();
  }

  bool wait_next_token(std::string* token_text, bool* finished, bool* failed,
                       std::string* error) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return next_idx_ < token_texts_.size() || finished_; });
    if (next_idx_ < token_texts_.size()) {
      *token_text = token_texts_[next_idx_++];
      *finished = false;
      *failed = false;
      error->clear();
      return true;
    }
    *finished = true;
    *failed = failed_;
    *error = error_;
    return false;
  }

  std::string wait_full_text(bool* failed, std::string* error) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return finished_; });
    *failed = failed_;
    *error = error_;
    return full_text_;
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::string> token_texts_;
  size_t next_idx_ = 0;
  int64_t request_id_ = -1;
  std::string full_text_;
  bool finished_ = false;
  bool failed_ = false;
  bool cancelled_ = false;
  std::string error_;
};

class OnlineServingEngine {
 public:
  OnlineServingEngine(ServingBenchmarkApp* app, const BenchConfig& config)
      : app_(app), config_(config) {}

  ~OnlineServingEngine() { stop(); }

  void start() {
    SchedulerConfig sched_config;
    sched_config.max_num_seqs = app_->max_model_batch_size();
    sched_config.max_num_batched_tokens = config_.max_num_batched_tokens;
    sched_config.prefill_chunk_cap = config_.prefill_chunk_cap;
    scheduler_ = std::make_unique<Scheduler>(sched_config, app_->kv_cache_manager());
    worker_ = std::thread([this]() { run_loop(); });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::shared_ptr<OnlineRequestHandle> submit(const std::string& prompt,
                                              int32_t max_new_tokens,
                                              int32_t min_new_tokens,
                                              bool ignore_eos,
                                              std::string* error) {
    auto handle = std::make_shared<OnlineRequestHandle>();
    auto prompt_tokens = app_->encode_prompt(prompt);
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_) {
        *error = "server_stopping";
        return nullptr;
      }
      if (pending_submissions_ >= config_.max_queue_size) {
        *error = "queue_full";
        return nullptr;
      }
      PendingSubmission submission;
      submission.prompt_tokens = std::move(prompt_tokens);
      submission.max_new_tokens = std::max(1, max_new_tokens);
      submission.min_new_tokens = std::max(0, min_new_tokens);
      submission.ignore_eos = ignore_eos;
      submission.handle = handle;
      submissions_.push_back(std::move(submission));
      ++pending_submissions_;
    }
    cv_.notify_all();
    return handle;
  }

  void cancel(const std::shared_ptr<OnlineRequestHandle>& handle,
              const std::string& reason) {
    if (!handle) {
      return;
    }
    handle->mark_cancelled();
    {
      std::lock_guard<std::mutex> lock(mu_);
      cancellations_.push_back({handle->request_id(), reason});
    }
    cv_.notify_all();
  }

 private:
  struct PendingSubmission {
    std::vector<int32_t> prompt_tokens;
    int32_t max_new_tokens = 0;
    int32_t min_new_tokens = 0;
    bool ignore_eos = false;
    std::shared_ptr<OnlineRequestHandle> handle;
  };

  void run_loop() {
    void* stream = app_->model_stream();
    while (true) {
      flush_submissions();
      if (!scheduler_->has_active_requests()) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return stopping_ || !submissions_.empty(); });
        if (stopping_ && submissions_.empty()) {
          break;
        }
        continue;
      }
      run_step(stream);
    }
  }

  void flush_submissions() {
    std::deque<PendingSubmission> local;
    {
      std::lock_guard<std::mutex> lock(mu_);
      local.swap(submissions_);
      pending_submissions_ -= static_cast<int32_t>(local.size());
    }
    while (!local.empty()) {
      auto submission = std::move(local.front());
      local.pop_front();
      const int64_t request_id = scheduler_->add_request(
          std::move(submission.prompt_tokens), submission.max_new_tokens,
          submission.min_new_tokens, submission.ignore_eos);
      submission.handle->set_request_id(request_id);
      handles_[request_id] = submission.handle;
      if (submission.handle->cancelled()) {
        scheduler_->cancel_request(request_id, "client_disconnected");
      }
    }
  }

  void flush_cancellations() {
    std::vector<std::pair<int64_t, std::string>> local;
    {
      std::lock_guard<std::mutex> lock(mu_);
      local.swap(cancellations_);
    }
    for (const auto& [request_id, reason] : local) {
      if (request_id >= 0) {
        scheduler_->cancel_request(request_id, reason);
      }
    }
  }

  void run_step(void* stream) {
    flush_cancellations();
    SchedulerOutput sched_out = scheduler_->schedule_step();
    if (sched_out.total_tokens == 0) {
      publish_finished();
      return;
    }

    const bool decode_only_step =
        sched_out.num_decode_seqs > 0 && sched_out.num_prefill_seqs == 0;
    MixedBatchMetadata batch = decode_only_step
        ? scheduler_->build_decode_batch(sched_out, stream)
        : scheduler_->build_mixed_batch(sched_out, stream);
    base::Status status = decode_only_step ? app_->forward_decode_batch(batch)
                                           : app_->forward_mixed_batch(batch);
    if (!status) {
      LOG(ERROR) << "online forward failed: " << status.get_err_msg();
      return;
    }
    auto sampled_tokens = app_->batch_sample(batch);
    publish_sampled_tokens(sched_out, batch, sampled_tokens);
    scheduler_->process_outputs(sched_out, batch, sampled_tokens,
                                [&](int32_t token) { return app_->is_sentence_ending(token); });
    publish_finished();
  }

  void publish_sampled_tokens(const SchedulerOutput& output,
                              const MixedBatchMetadata& batch,
                              const SampledTokenView& sampled_tokens) {
    CHECK_EQ(sampled_tokens.size(), static_cast<int32_t>(batch.sample_row_to_request.size()));
    for (int32_t sample_idx = 0; sample_idx < sampled_tokens.size(); ++sample_idx) {
      const int32_t request_idx = batch.sample_row_to_request[sample_idx];
      CHECK_GE(request_idx, 0);
      CHECK_LT(request_idx, static_cast<int32_t>(output.scheduled_seqs.size()));
      const auto* seq = output.scheduled_seqs[request_idx];
      CHECK_NE(seq, nullptr);
      auto it = handles_.find(seq->client_request_id);
      if (it == handles_.end()) {
        continue;
      }
      const int32_t token = sampled_tokens[sample_idx];
      if (app_->is_sentence_ending(token)) {
        continue;
      }
      it->second->push_token(app_->decode_tokens({token}));
    }
  }

  void publish_finished() {
    auto finished = scheduler_->pop_finished();
    for (const auto& seq : finished) {
      auto it = handles_.find(seq.client_request_id);
      if (it == handles_.end()) {
        continue;
      }
      if (it->second->cancelled()) {
        handles_.erase(it);
        continue;
      }
      std::string full_text;
      if (!seq.failed) {
        full_text = app_->postprocess_decoded_text(app_->decode_tokens(seq.output_tokens));
      }
      it->second->finish(full_text, seq.failed, seq.finish_reason);
      handles_.erase(it);
    }
  }

  ServingBenchmarkApp* app_ = nullptr;
  BenchConfig config_;
  std::unique_ptr<Scheduler> scheduler_;
  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<PendingSubmission> submissions_;
  std::vector<std::pair<int64_t, std::string>> cancellations_;
  int32_t pending_submissions_ = 0;
  bool stopping_ = false;
  std::unordered_map<int64_t, std::shared_ptr<OnlineRequestHandle>> handles_;
};

void handle_generate_request(int client_fd,
                             OnlineServingEngine* engine,
                             const std::string& request) {
  nlohmann::json payload;
  try {
    payload = nlohmann::json::parse(http_body(request));
  } catch (const std::exception& e) {
    send_http_response(client_fd, 400, "application/json", json_error(e.what()));
    return;
  }

  const std::string prompt = payload.value("prompt", "");
  if (prompt.empty()) {
    send_http_response(client_fd, 400, "application/json", json_error("prompt is required"));
    return;
  }
  const int32_t max_new_tokens = payload.value("max_new_tokens", 128);
  const int32_t min_new_tokens = payload.value("min_new_tokens", 0);
  const bool ignore_eos = payload.value("ignore_eos", false);
  const bool stream = payload.value("stream", false);

  std::string error;
  auto handle = engine->submit(prompt, max_new_tokens, min_new_tokens, ignore_eos, &error);
  if (!handle) {
    const int status = error == "queue_full" ? 429 : 500;
    send_http_response(client_fd, status, "application/json", json_error(error));
    return;
  }

  if (!stream) {
    bool failed = false;
    std::string finish_error;
    const std::string text = handle->wait_full_text(&failed, &finish_error);
    if (failed) {
      send_http_response(client_fd, 500, "application/json", json_error(finish_error));
      return;
    }
    nlohmann::json response;
    response["text"] = text;
    send_http_response(client_fd, 200, "application/json", response.dump());
    return;
  }

  std::ostringstream headers;
  headers << "HTTP/1.1 200 OK\r\n"
          << "Content-Type: text/event-stream\r\n"
          << "Cache-Control: no-cache\r\n"
          << "Connection: close\r\n\r\n";
  if (!send_all(client_fd, headers.str())) {
    engine->cancel(handle, "client_disconnected");
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
      event["text"] = token_text;
      if (!send_all(client_fd, "data: " + event.dump() + "\n\n")) {
        engine->cancel(handle, "client_disconnected");
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

void handle_http_client(int client_fd, OnlineServingEngine* engine) {
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
  if (method != "POST") {
    send_http_response(client_fd, 405, "application/json", json_error("method not allowed"));
    return;
  }
  if (path != "/generate" && path != "/v1/completions") {
    send_http_response(client_fd, 404, "application/json", json_error("not found"));
    return;
  }
  handle_generate_request(client_fd, engine, request);
}

int create_listen_socket(const std::string& host, int32_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    PLOG(ERROR) << "socket failed";
    return -1;
  }
  int reuse = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    LOG(ERROR) << "Invalid --listen-host=" << host;
    ::close(fd);
    return -1;
  }
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    PLOG(ERROR) << "bind failed";
    ::close(fd);
    return -1;
  }
  if (::listen(fd, 128) != 0) {
    PLOG(ERROR) << "listen failed";
    ::close(fd);
    return -1;
  }
  return fd;
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
        read_int("--max-queue-size=", config.max_queue_size)) {
      continue;
    }
    const std::string_view listen_host_prefix = "--listen-host=";
    if (starts_with(arg, listen_host_prefix)) {
      config.listen_host = std::string(arg.substr(listen_host_prefix.size()));
      continue;
    }
  }
  config.max_new_tokens = std::max(1, config.max_new_tokens);
  config.max_num_batched_tokens = std::max(1, config.max_num_batched_tokens);
  config.prefill_chunk_cap = std::max(1, config.prefill_chunk_cap);
  config.warmup_rounds = std::max(0, config.warmup_rounds);
  config.kv_cache_memory_utilization =
      std::min(1.0, std::max(0.01, config.kv_cache_memory_utilization));
  config.listen_port = std::max(1, std::min(65535, config.listen_port));
  config.max_queue_size = std::max(1, config.max_queue_size);
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

void print_step_profile(const StepProfile& profile) {
  std::cout << "STEP_PROFILE"
            << " step=" << profile.step
            << " requests=" << profile.num_requests
            << " tokens=" << profile.num_tokens
            << " decode_tokens=" << profile.num_decode_tokens
            << " prefill_tokens=" << profile.num_prefill_tokens
            << " sampled_tokens=" << profile.num_sampled_tokens
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
  if (!initialize_model(model_path_, tokenizer_path_, bench_config_)) {
    return -1;
  }

  prepare_benchmark_config();
  run_warmup();
  kv_cache_manager()->reset_radix_cache_stats();
  if (bench_config_.online_server) {
    return run_online_server();
  }
  create_scheduler();
  submit_all_requests();
  run_serving_loop();
  return 0;
}

int ServingBenchmarkApp::run_online_server() {
  OnlineServingEngine engine(this, bench_config_);
  engine.start();

  const int listen_fd = create_listen_socket(bench_config_.listen_host,
                                             bench_config_.listen_port);
  if (listen_fd < 0) {
    return -1;
  }

  LOG(INFO) << "Online serving listening on http://" << bench_config_.listen_host
            << ":" << bench_config_.listen_port
            << " endpoints: GET /health, POST /generate, POST /v1/completions";
  std::cout << "ONLINE_SERVER_READY host=" << bench_config_.listen_host
            << " port=" << bench_config_.listen_port << std::endl;

  while (true) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    const int client_fd = ::accept(listen_fd,
                                   reinterpret_cast<sockaddr*>(&client_addr),
                                   &client_len);
    if (client_fd < 0) {
      PLOG(ERROR) << "accept failed";
      continue;
    }
    std::thread([client_fd, &engine]() {
      handle_http_client(client_fd, &engine);
      ::close(client_fd);
    }).detach();
  }

  ::close(listen_fd);
  return 0;
}

std::vector<std::string> ServingBenchmarkApp::default_prompts() const {
  return {"What is AI?", "Write a haiku about coding.",
          "Explain quantum computing briefly."};
}

std::string ServingBenchmarkApp::postprocess_decoded_text(std::string text) const {
  return text;
}

bool ServingBenchmarkApp::parse_args(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: " << usage_name()
              << " <model.bin> <tokenizer.json> [prompt1] [prompt2] ..."
              << " [--max-new-tokens=N] [--max-batched-tokens=N]"
              << " [--prefill-chunk-cap=N] [--kv-cache-memory-utilization=0.8] [--quiet=0|1]"
              << " [--warmup-rounds=N]"
              << " [--step-profile=0|1] [--step-trace=0|1] [--final-summary=0|1]"
              << " [--online-server=0|1] [--listen-host=127.0.0.1]"
              << " [--listen-port=8080] [--max-queue-size=N]";
    return false;
  }

  model_path_ = argv[1];
  tokenizer_path_ = argv[2];
  bench_config_ = parse_bench_config(argc, argv, 3);
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

      auto sampled_tokens = batch_sample(batch);
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
    scheduler.add_request(std::move(tokens), max_new_tokens);
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
    summary_.total_steps++;
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
    return batch_sample(batch);
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
      batch, sampled_tokens.size(), schedule_start, schedule_end, build_start, build_end,
      forward_start, forward_end, sample_start, sample_end, process_start, process_end,
      step_start, step_end);
  record_step_profile(profile, decode_only_step, static_cast<int32_t>(finished.size()));
  maybe_print_step_profile(profile, batch, sampled_tokens.size());

  ++step_;
}

StepProfile ServingBenchmarkApp::build_step_profile(
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
