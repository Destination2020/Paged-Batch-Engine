#include "base/shared_kv_arena.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "serving/serving_zmq_rpc.h"

namespace base {
namespace {
using Json = nlohmann::json;
void cuda_check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
std::string uuid_for_device(int device) {
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device));
  static const char digits[] = "0123456789abcdef";
  std::string uuid;
  for (unsigned char byte : prop.uuid.bytes) {
    uuid += digits[byte >> 4]; uuid += digits[byte & 15];
  }
  return uuid;
}
Json lease_json(const ChunkLease& lease) {
  return {{"epoch", lease.epoch}, {"id", lease.id}, {"generation", lease.generation},
          {"offset", lease.offset}, {"bytes", lease.bytes},
          {"model", lease.model}, {"layout", lease.layout}};
}
ChunkLease parse_lease(const Json& j) {
  return {j.at("epoch"), j.at("id"), j.at("generation"), j.at("offset"),
          j.at("bytes"), j.at("model"), j.at("layout")};
}
uint64_t round_up(uint64_t bytes, uint64_t alignment) {
  if (bytes > std::numeric_limits<uint64_t>::max() - alignment)
    throw std::invalid_argument("KV size overflow");
  return (bytes + alignment - 1) / alignment * alignment;
}
}  // namespace

struct SharedKVArenaClient::Impl {
  serving::ZmqRpcConfig rpc;
  std::string model;
  uint64_t bytes = 0, epoch = 0, hard_limit = 0;
  void* data = nullptr;
  int device = 0;
  std::mutex mu;
  Json request(Json body) {
    std::lock_guard<std::mutex> lock(mu);
    body["epoch"] = epoch;
    body["model"] = model;
    Json response;
    auto status = serving::zmq_request_response(rpc, body, &response);
    if (!status) throw std::runtime_error("KV owner unreachable: " + status.get_err_msg());
    if (!response.value("ok", false)) throw std::runtime_error(response.value("error", "KV owner error"));
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
  auto handle_bytes = serving::hex_json_to_binary(response.at("memory_handle"));
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
uint64_t SharedKVArenaClient::hard_limit() const { return impl_->hard_limit; }
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

struct SharedKVPageGroup::Impl {
  struct Page {
    uint64_t offset;
    std::vector<bool> active, reserved;
    std::vector<cudaEvent_t> retired;
    explicit Page(uint64_t offset_value, int layers)
        : offset(offset_value), active(layers, false), reserved(layers, false), retired(layers, nullptr) {}
    bool empty() const {
      return std::none_of(active.begin(), active.end(), [](bool v) { return v; }) &&
             std::all_of(retired.begin(), retired.end(), [](cudaEvent_t e) { return !e; });
    }
  };
  struct Chunk { ChunkLease lease; std::vector<Page> pages; };
  std::shared_ptr<SharedKVArenaClient> arena;
  int layers;
  uint64_t half_bytes, page_bytes, alignment, chunk_bytes;
  cudaStream_t stream;
  std::string layout;
  std::vector<Chunk> chunks;
  std::vector<std::deque<int>> pending;
  mutable std::recursive_mutex mu;
  uint64_t leased() const {
    uint64_t result = 0;
    for (const auto& chunk : chunks) result += chunk.lease.bytes;
    return result;
  }
};
SharedKVPageGroup::SharedKVPageGroup(std::shared_ptr<SharedKVArenaClient> arena,
    int layers, uint64_t half_bytes, uint64_t target, cudaStream_t stream) : impl_(new Impl) {
  if (!arena || layers <= 0 || !half_bytes || half_bytes % 256 || !target ||
      half_bytes > std::numeric_limits<uint64_t>::max() / (2ull * layers))
    throw std::invalid_argument("unsupported shared KV geometry/alignment");
  impl_->arena = std::move(arena);
  impl_->layers = layers;
  impl_->pending.resize(layers);
  impl_->half_bytes = half_bytes;
  impl_->page_bytes = half_bytes * 2 * layers;
  impl_->alignment = impl_->page_bytes; // all-layer pages, each plane 256-byte aligned
  impl_->chunk_bytes = round_up(std::max(target, impl_->page_bytes), impl_->alignment);
  impl_->stream = stream;
  impl_->layout = "page-major-gqa-v1:" + std::to_string(layers) + ":" + std::to_string(half_bytes);
  if (capacity() <= 0 || impl_->arena->bytes() / half_bytes > INT32_MAX)
    throw std::invalid_argument("shared KV capacity exceeds block ID range or is zero");
}
SharedKVPageGroup::~SharedKVPageGroup() {
  // Failed CUDA synchronization must quarantine leases; never return their bytes.
  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) return;
  for (auto& chunk : impl_->chunks) {
    for (auto& page : chunk.pages)
      for (auto event : page.retired) if (event) cudaEventDestroy(event);
    try { impl_->arena->release(chunk.lease); } catch (...) { /* domain shutdown */ }
  }
}
void SharedKVPageGroup::collect() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mu);
  for (auto it = impl_->chunks.begin(); it != impl_->chunks.end();) {
    for (auto& page : it->pages) {
      for (auto& event : page.retired) {
        if (!event) continue;
        const auto status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) continue;
        cuda_check(status);
        cuda_check(cudaEventDestroy(event));
        event = nullptr;
      }
    }
    if (std::all_of(it->pages.begin(), it->pages.end(), [](const Impl::Page& p) { return p.empty(); })) {
      impl_->arena->release(it->lease);
      it = impl_->chunks.erase(it);
    } else ++it;
  }
}
int SharedKVPageGroup::allocate(int layer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mu);
  if (layer < 0 || layer >= impl_->layers) throw std::invalid_argument("invalid layer");
  collect();
  if (layer > 0) {
    auto& pending = impl_->pending[layer];
    if (pending.empty()) throw std::logic_error("Allocate shared pages in layer order");
    const int id = pending.front(); pending.pop_front();
    for (auto& chunk : impl_->chunks) for (auto& page : chunk.pages)
      if (id == static_cast<int>(page.offset / impl_->half_bytes)) page.reserved[layer] = false;
    return id;
  }
  auto reserve = [&](Impl::Page& page) {
    const int id = static_cast<int>(page.offset / impl_->half_bytes);
    std::fill(page.active.begin(), page.active.end(), true);
    for (int l = 1; l < impl_->layers; ++l) {
      page.reserved[l] = true;
      impl_->pending[l].push_back(id);
    }
    return id;
  };
  for (auto& chunk : impl_->chunks)
    for (auto& page : chunk.pages) if (page.empty()) return reserve(page);
  ChunkLease lease;
  // Quota/fragmentation failures may retry a single all-layer page. Other errors
  // (owner death, epoch mismatch, protocol failure) fail the worker/domain.
  auto acquire = [&](uint64_t bytes) {
    return impl_->arena->acquire(impl_->layout, bytes, impl_->alignment);
  };
  try { lease = acquire(impl_->chunk_bytes); }
  catch (const std::runtime_error& e) {
    const std::string error = e.what();
    if (error != "model_kv_quota_exhausted" && error != "arena_fragmented" && error != "arena_exhausted") throw;
    try { lease = acquire(impl_->page_bytes); }
    catch (const std::runtime_error& retry) {
      const std::string message = retry.what();
      if (message == "model_kv_quota_exhausted" || message == "arena_fragmented" || message == "arena_exhausted") return -1;
      throw;
    }
  }
  Impl::Chunk chunk;
  chunk.lease = lease;
  for (uint64_t offset = lease.offset; offset < lease.offset + lease.bytes; offset += impl_->page_bytes)
    chunk.pages.emplace_back(offset, impl_->layers);
  const int id = reserve(chunk.pages.front());
  impl_->chunks.emplace_back(std::move(chunk));
  return id;
}
void SharedKVPageGroup::retire(int layer, int id) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mu);
  for (auto& chunk : impl_->chunks) {
    for (auto& page : chunk.pages) {
      if (id != static_cast<int>(page.offset / impl_->half_bytes)) continue;
      if (!page.active.at(layer) || page.retired.at(layer) || page.reserved.at(layer))
        throw std::logic_error("double/stale page retire");
      if (layer == 0) {
        // Roll back unconsumed layer reservations if layer-0 allocation failed.
        for (int l = 1; l < impl_->layers; ++l) if (page.reserved[l]) {
          auto& pending = impl_->pending[l];
          pending.erase(std::remove(pending.begin(), pending.end(), id), pending.end());
          page.reserved[l] = false;
          page.active[l] = false;
        }
      }
      cudaEvent_t event{};
      cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      const auto status = cudaEventRecord(event, impl_->stream);
      if (status != cudaSuccess) { cudaEventDestroy(event); cuda_check(status); }
      page.retired[layer] = event;
      page.active[layer] = false;
      // Fast reclamation after sampled-token synchronization; otherwise deferred.
      collect();
      return;
    }
  }
  throw std::logic_error("page does not belong to current chunk generation");
}
int SharedKVPageGroup::free_blocks(int layer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mu);
  uint64_t count = (impl_->arena->hard_limit() - impl_->leased()) / impl_->page_bytes;
  for (const auto& chunk : impl_->chunks)
    for (const auto& page : chunk.pages)
      if (page.empty()) ++count;
  count += impl_->pending.at(layer).size();
  return static_cast<int>(std::min<uint64_t>(INT32_MAX, count));
}
uint64_t SharedKVPageGroup::leased_bytes() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mu);
  return impl_->leased();
}
uint64_t SharedKVPageGroup::arena_bytes() const { return impl_->arena->bytes(); }
int SharedKVPageGroup::capacity() const {
  return static_cast<int>(std::min<uint64_t>(INT32_MAX, impl_->arena->hard_limit() / impl_->page_bytes));
}
int SharedKVPageGroup::addressable_blocks() const {
  // Each layer has a different base; keep the last V plane within the mapping.
  return static_cast<int>(impl_->arena->bytes() / impl_->half_bytes - 2 * impl_->layers + 1);
}
void* SharedKVPageGroup::key_base(int layer) const {
  return static_cast<char*>(impl_->arena->data()) + 2ull * layer * impl_->half_bytes;
}
void* SharedKVPageGroup::value_base(int layer) const {
  return static_cast<char*>(impl_->arena->data()) + (2ull * layer + 1) * impl_->half_bytes;
}

int run_shared_kv_owner(const std::string& endpoint, uint64_t bytes, int device) {
  cuda_check(cudaSetDevice(device));
  void* data = nullptr;
  cuda_check(cudaMalloc(&data, bytes)); // dedicated allocation, never allocator-cached
  try {
    cudaIpcMemHandle_t handle{};
    cuda_check(cudaIpcGetMemHandle(&handle, data));
    const auto epoch = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    GlobalChunkAllocator allocator(bytes, epoch);
    std::unique_ptr<serving::ZmqSocket> socket;
    auto status = serving::make_zmq_rep_socket({endpoint, 1000}, &socket);
    if (!status) throw std::runtime_error(status.get_err_msg());
    const auto uuid = uuid_for_device(device);
    for (;;) {
      Json request;
      status = socket->recv_json(&request);
      if (!status) continue;
      Json response;
      try {
        const std::string type = request.at("type");
        if (type == "health" || type == "metrics") {
          size_t free_bytes = 0, total_bytes = 0;
          cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
          response = {{"gpu_free_bytes", free_bytes}, {"gpu_total_bytes", total_bytes},
              {"ok", true}, {"bytes", bytes}, {"epoch", epoch}, {"gpu_uuid", uuid},
              {"free_bytes", allocator.free_bytes()}, {"lease_count", allocator.lease_count()},
              {"memory_handle", serving::binary_to_hex_json(std::string(reinterpret_cast<char*>(&handle), sizeof(handle)))}};
        } else {
          if (request.at("epoch").get<uint64_t>() != epoch) throw std::invalid_argument("stale_arena_epoch");
          const std::string model = request.at("model");
          if (type == "register") allocator.register_model(model, request.at("hard_limit"));
          else if (type == "acquire") {
            auto lease = allocator.acquire(model, request.at("layout"), request.at("bytes"), request.at("alignment"));
            response["lease"] = lease_json(lease);
          } else if (type == "release") {
            auto lease = parse_lease(request.at("lease"));
            if (lease.model != model || !request.value("gpu_complete", false))
              throw std::invalid_argument("release_without_owner_identity_or_gpu_fence");
            allocator.release(lease);
          } else throw std::invalid_argument("unknown_owner_operation");
          response["ok"] = true;
        }
      } catch (const std::exception& e) { response = {{"ok", false}, {"error", e.what()}}; }
      status = socket->send_json(response);
      if (!status) throw std::runtime_error(status.get_err_msg());
    }
  } catch (...) { cudaFree(data); throw; }
}
}  // namespace base
