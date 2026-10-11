#include "base/shared_kv_arena.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "base/zmq_transport.h"

#include "shared_kv_protocol.h"
namespace base {
using namespace shared_kv_detail;
struct SharedKVArenaClient::Impl {
  transport::ZmqRpcConfig rpc;
  std::string model;
  uint64_t bytes = 0, epoch = 0, hard_limit = 0;
  void* data = nullptr;
  int device = 0;
  std::mutex mu;
  std::unique_ptr<transport::ZmqSocket> socket;
  bool failed = false;
  bool measure = false; // Set before publishing the client to other threads.
  SharedKVRpcTiming timing;
  Json request(Json body) {
    using Clock = std::chrono::steady_clock;
    const auto waiting = measure ? Clock::now() : Clock::time_point{};
    std::lock_guard<std::mutex> lock(mu);
    if (failed) throw std::runtime_error("KV owner connection failed; stop GPU domain");
    const auto exchanging = measure ? Clock::now() : Clock::time_point{};
    body["epoch"] = epoch;
    body["model"] = model;
    Json response;
    auto status = socket ? error::Success() : transport::make_zmq_req_socket(rpc, &socket);
    if (status) status = socket->send_json(body);
    if (status) status = socket->recv_json(&response);
    if (measure) {
      ++timing.calls;
      timing.mutex_wait_us += std::chrono::duration<double, std::micro>(exchanging - waiting).count();
      timing.exchange_us += std::chrono::duration<double, std::micro>(Clock::now() - exchanging).count();
    }
    if (!status) {
      // A lost reply may follow a committed acquire/release. Never replay it.
      failed = true;
      socket.reset();
      throw std::runtime_error("KV owner unreachable: " + status.get_err_msg());
    }
    if (!response.value("ok", false)) {
      const int code = response.value("allocation_error", 0);
      if (code >= 1 && code <= 3) throw ChunkAllocationError(static_cast<ChunkAllocationFailure>(code));
      throw std::runtime_error(response.value("error", "KV owner error"));
    }
    return response;
  }
};
SharedKVArenaClient::SharedKVArenaClient(std::string endpoint, std::string model,
    uint64_t hard_limit_bytes, int device) : impl_(new Impl) {
  impl_->rpc = {std::move(endpoint), 5000};
  impl_->model = std::move(model);
  impl_->device = device;
  cuda_check(cudaSetDevice(device));
  auto response = impl_->request({{"type", "health"}});
  if (response.at("gpu_uuid") != uuid_for_device(device))
    throw std::runtime_error("shared KV owner and worker GPU UUID differ");
  impl_->bytes = response.at("bytes");
  impl_->epoch = response.at("epoch");
  impl_->hard_limit = hard_limit_bytes;
  auto handle_bytes = transport::hex_json_to_binary(response.at("memory_handle"));
  if (handle_bytes.size() != sizeof(cudaIpcMemHandle_t))
    throw std::runtime_error("invalid CUDA IPC memory handle");
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, handle_bytes.data(), sizeof(handle));
  cuda_check(cudaIpcOpenMemHandle(&impl_->data, handle, cudaIpcMemLazyEnablePeerAccess));
  try {
    impl_->request({{"type", "register"}, {"hard_limit", hard_limit_bytes}});
  } catch (...) {
    cudaIpcCloseMemHandle(impl_->data);
    impl_->data = nullptr;
    throw;
  }
}
SharedKVArenaClient::~SharedKVArenaClient() {
  if (impl_->data) { cudaSetDevice(impl_->device); cudaIpcCloseMemHandle(impl_->data); }
}
void* SharedKVArenaClient::data() const { return impl_->data; }
uint64_t SharedKVArenaClient::bytes() const { return impl_->bytes; }
uint64_t SharedKVArenaClient::epoch() const { return impl_->epoch; }
uint64_t SharedKVArenaClient::hard_limit() const { return impl_->hard_limit; }
void SharedKVArenaClient::enable_rpc_timing() { impl_->measure = true; }
SharedKVRpcTiming SharedKVArenaClient::rpc_timing() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->timing;
}
ChunkLease SharedKVArenaClient::acquire(const std::string& layout,
    uint64_t bytes, uint64_t alignment) {
  auto lease = parse_lease(impl_->request({{"type", "acquire"}, {"layout", layout},
                                    {"bytes", bytes}, {"alignment", alignment}}).at("lease"));
  if (lease.epoch != impl_->epoch || !lease.id || !lease.generation ||
      lease.model != impl_->model || lease.layout != layout || lease.bytes != bytes ||
      lease.offset % alignment || lease.offset > impl_->bytes || lease.bytes > impl_->bytes - lease.offset)
    throw std::runtime_error("invalid owner chunk descriptor");
  return lease;
}
void SharedKVArenaClient::release(const ChunkLease& lease) {
  impl_->request({{"type", "release"}, {"lease", lease_json(lease)}, {"gpu_complete", true}});
}

} // namespace base
