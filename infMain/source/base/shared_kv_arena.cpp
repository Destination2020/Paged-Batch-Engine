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
struct SharedKVPageGroup::Impl {
  struct Chunk;
  struct Page {
    uint64_t offset;
    std::vector<bool> active, reserved;
    int active_layers = 0;
    cudaEvent_t retired = nullptr;
    Chunk* chunk = nullptr;
    explicit Page(uint64_t offset_value, int layers)
        : offset(offset_value), active(layers, false), reserved(layers, false) {}
  };
  struct Chunk { ChunkLease lease; std::vector<Page> pages; size_t free_pages = 0; };
  std::shared_ptr<SharedKVArenaClient> arena;
  int layers;
  uint64_t half_bytes, page_bytes, alignment, chunk_bytes;
  cudaStream_t stream;
  std::string layout;
  // Heap-owned chunks and fixed-size page vectors keep indexed Page* stable.
  std::unordered_map<uint64_t, std::unique_ptr<Chunk>> chunks;
  std::unordered_map<int, Page*> pages;
  std::set<int> free_pages;
  std::deque<Page*> retiring;
  std::vector<cudaEvent_t> events;
  uint64_t leased_bytes = 0;
  bool failed = false; // An unacknowledged release can never become locally reusable.
  std::vector<std::deque<int>> pending;
  mutable std::mutex mu;
  uint64_t leased() const {
    return leased_bytes;
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
  for (auto& entry : impl_->chunks) {
    auto& chunk = *entry.second;
    for (auto& page : chunk.pages)
      if (page.retired) cudaEventDestroy(page.retired);
    try { impl_->arena->release(chunk.lease); } catch (...) { /* domain shutdown */ }
  }
  for (auto event : impl_->events) cudaEventDestroy(event);
}
void SharedKVPageGroup::collect() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  collect_locked();
}
void SharedKVPageGroup::collect_locked() {
  if (impl_->failed) throw std::runtime_error("shared KV page group failed; stop GPU domain");
  while (!impl_->retiring.empty()) {
    auto* page = impl_->retiring.front();
    const auto status = cudaEventQuery(page->retired);
    // All retirement events use the same stream and are recorded in queue order.
    if (status == cudaErrorNotReady) break;
    cuda_check(status);
    impl_->events.push_back(page->retired);
    page->retired = nullptr;
    impl_->retiring.pop_front();
    impl_->free_pages.insert(static_cast<int>(page->offset / impl_->half_bytes));
    auto* chunk = page->chunk;
    if (++chunk->free_pages == chunk->pages.size()) {
      try {
        impl_->arena->release(chunk->lease);
      } catch (...) {
        impl_->failed = true;
        throw;
      }
      for (const auto& item : chunk->pages) {
        const int id = static_cast<int>(item.offset / impl_->half_bytes);
        impl_->free_pages.erase(id);
        impl_->pages.erase(id);
      }
      impl_->leased_bytes -= chunk->lease.bytes;
      impl_->chunks.erase(chunk->lease.id);
    }
  }
}
int SharedKVPageGroup::allocate(int layer) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->failed) throw std::runtime_error("shared KV page group failed; stop GPU domain");
  if (layer < 0 || layer >= impl_->layers) throw std::invalid_argument("invalid layer");
  if (layer > 0) {
    auto& pending = impl_->pending[layer];
    if (pending.empty()) throw std::logic_error("Allocate shared pages in layer order");
    const int id = pending.front(); pending.pop_front();
    impl_->pages.at(id)->reserved[layer] = false;
    return id;
  }
  collect_locked();
  auto reserve = [&](Impl::Page& page) {
    const int id = static_cast<int>(page.offset / impl_->half_bytes);
    std::fill(page.active.begin(), page.active.end(), true);
    page.active_layers = impl_->layers;
    --page.chunk->free_pages;
    impl_->free_pages.erase(id);
    for (int l = 1; l < impl_->layers; ++l) {
      page.reserved[l] = true;
      impl_->pending[l].push_back(id);
    }
    return id;
  };
  if (!impl_->free_pages.empty()) return reserve(*impl_->pages.at(*impl_->free_pages.begin()));
  const uint64_t remaining = impl_->arena->hard_limit() - impl_->leased_bytes;
  if (remaining < impl_->page_bytes) return -1;
  const uint64_t request_bytes = remaining >= impl_->chunk_bytes ? impl_->chunk_bytes : impl_->page_bytes;
  ChunkLease lease;
  // Quota/fragmentation failures may retry a single all-layer page. Other errors
  // (owner death, epoch mismatch, protocol failure) fail the worker/domain.
  auto acquire = [&](uint64_t bytes) {
    return impl_->arena->acquire(impl_->layout, bytes, impl_->alignment);
  };
  try { lease = acquire(request_bytes); }
  catch (const ChunkAllocationError&) {
    if (request_bytes == impl_->page_bytes) return -1;
    try { lease = acquire(impl_->page_bytes); }
    catch (const ChunkAllocationError&) { return -1; }
  }
  auto chunk = std::make_unique<Impl::Chunk>();
  chunk->lease = lease;
  chunk->pages.reserve(lease.bytes / impl_->page_bytes);
  for (uint64_t offset = lease.offset; offset < lease.offset + lease.bytes; offset += impl_->page_bytes)
    chunk->pages.emplace_back(offset, impl_->layers);
  chunk->free_pages = chunk->pages.size();
  for (auto& page : chunk->pages) {
    page.chunk = chunk.get();
    const int id = static_cast<int>(page.offset / impl_->half_bytes);
    impl_->pages.emplace(id, &page);
    impl_->free_pages.insert(id);
  }
  const int id = reserve(chunk->pages.front());
  impl_->leased_bytes += lease.bytes;
  impl_->chunks.emplace(lease.id, std::move(chunk));
  return id;
}
void SharedKVPageGroup::retire(int layer, int id) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->failed) throw std::runtime_error("shared KV page group failed; stop GPU domain");
  auto found = impl_->pages.find(id);
  if (found == impl_->pages.end()) throw std::logic_error("page does not belong to current chunk generation");
  auto& page = *found->second;
  if (!page.active.at(layer) || page.retired || page.reserved.at(layer))
    throw std::logic_error("double/stale page retire");
  if (layer == 0) {
    // Roll back unconsumed layer reservations if layer-0 allocation failed.
    for (int l = 1; l < impl_->layers; ++l) if (page.reserved[l]) {
      auto& pending = impl_->pending[l];
      pending.erase(std::remove(pending.begin(), pending.end(), id), pending.end());
      page.reserved[l] = false;
      page.active[l] = false;
      --page.active_layers;
    }
  }
  page.active[layer] = false;
  if (--page.active_layers == 0) {
    // A whole page is indivisible across layers. Its final reference retires
    // after every layer's use on the group's stream, so one fence suffices.
    cudaEvent_t event{};
    if (impl_->events.empty()) cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    else { event = impl_->events.back(); impl_->events.pop_back(); }
    const auto status = cudaEventRecord(event, impl_->stream);
    if (status != cudaSuccess) { cudaEventDestroy(event); cuda_check(status); }
    page.retired = event;
    impl_->retiring.push_back(&page);
    collect_locked();
  }
}
int SharedKVPageGroup::free_blocks(int layer) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  uint64_t count = (impl_->arena->hard_limit() - impl_->leased()) / impl_->page_bytes;
  count += impl_->free_pages.size();
  count += impl_->pending.at(layer).size();
  return static_cast<int>(std::min<uint64_t>(INT32_MAX, count));
}
uint64_t SharedKVPageGroup::leased_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->leased();
}
uint64_t SharedKVPageGroup::arena_bytes() const { return impl_->arena->bytes(); }
uint64_t SharedKVPageGroup::arena_epoch() const { return impl_->arena->epoch(); }
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

} // namespace base
