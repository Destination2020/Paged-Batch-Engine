// CPU-only IPC benchmark: control-style RPC versus a persistent token channel.
#include "base/zmq_transport.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace rpc = base::transport;
using Clock = std::chrono::steady_clock;

void require(base::Status status) {
  if (!status) throw std::runtime_error(status.get_err_msg());
}

int main(int argc, char** argv) {
  const int samples = argc > 1 ? std::stoi(argv[1]) : 1000;
  if (samples < 1) return 2;
  const std::string path = "/tmp/pbe-rpc-bench-" + std::to_string(getpid());
  rpc::ZmqRpcConfig config{"ipc://" + path, 2000};
  std::promise<void> ready;
  auto future = ready.get_future();
  auto server = std::async(std::launch::async, [&] {
    try {
      std::unique_ptr<rpc::ZmqSocket> socket;
      require(rpc::make_zmq_rep_socket(config, &socket));
      require(socket->set_timeouts(5000));
      ready.set_value();
      for (int i = 0; i < 2 * (samples + 20); ++i) {
        nlohmann::json message;
        require(socket->recv_json(&message));
        require(socket->send_json(message));
      }
    } catch (...) {
      try { ready.set_exception(std::current_exception()); } catch (...) {}
      throw;
    }
  });
  try {
    future.get();
    nlohmann::json report;
    for (bool persistent : {false, true}) {
      std::unique_ptr<rpc::ZmqSocket> socket;
      if (persistent) require(rpc::make_zmq_req_socket(config, &socket));
      std::vector<double> timings;
      for (int i = -20; i < samples; ++i) {
        nlohmann::json reply, request{{"type", "health"}, {"sequence", i}};
        auto start = Clock::now();
        if (persistent) {
          require(socket->send_json(request));
          require(socket->recv_json(&reply));
        } else {
          require(rpc::zmq_request_response(config, request, &reply));
        }
        if (reply != request) throw std::runtime_error("RPC response mismatch");
        if (i >= 0) timings.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
      }
      std::sort(timings.begin(), timings.end());
      auto percentile = [&](double fraction) {
        return timings[static_cast<size_t>(std::ceil(fraction * timings.size())) - 1];
      };
      report[persistent ? "persistent_us" : "control_us"] = {
          {"p50", percentile(.5)}, {"p95", percentile(.95)},
          {"p99", percentile(.99)}, {"samples", samples}};
    }
    server.get();
    unlink(path.c_str());
    std::cout << report.dump(2) << '\n';
  } catch (const std::exception& error) {
    unlink(path.c_str());
    std::cerr << error.what() << '\n';
    return 1;
  }
}
