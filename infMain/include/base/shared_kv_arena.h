#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include "base/global_chunk_allocator.h"

namespace base {
struct SharedKVRpcTiming {
  uint64_t calls = 0;
  double mutex_wait_us = 0;
  double exchange_us = 0;
};
// One process maps the owner allocation once. All layer views share this lifetime.
class SharedKVArenaClient {
 public:
  SharedKVArenaClient(std::string endpoint, std::string model,
                     uint64_t hard_limit_bytes, int device);
  ~SharedKVArenaClient();
  SharedKVArenaClient(const SharedKVArenaClient&) = delete;
  SharedKVArenaClient& operator=(const SharedKVArenaClient&) = delete;
  ChunkLease acquire(const std::string& layout, uint64_t bytes, uint64_t alignment);
  void release(const ChunkLease& lease); // caller has completed every GPU access
  void* data() const;
  uint64_t bytes() const;
  uint64_t epoch() const;
  uint64_t hard_limit() const;
  // Opt-in diagnostics; disabled in serving to avoid clock reads on the hot path.
  void enable_rpc_timing();
  SharedKVRpcTiming rpc_timing() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// page = [layer0 K][layer0 V] ... [layerN K][layerN V].
// Kernel block IDs are half-page byte offsets, not chunk IDs.
class SharedKVPageGroup {
 public:
  SharedKVPageGroup(std::shared_ptr<SharedKVArenaClient> arena, int layers,
                    uint64_t half_layer_bytes, uint64_t chunk_target_bytes,
                    cudaStream_t stream);
  ~SharedKVPageGroup();
  int allocate(int layer);
  void retire(int layer, int block_id);
  int free_blocks(int layer) const;
  int capacity() const;
  uint64_t leased_bytes() const;
  uint64_t arena_bytes() const;
  uint64_t arena_epoch() const;
  int addressable_blocks() const;
  void* key_base(int layer) const;
  void* value_base(int layer) const;
  void collect();
 private:
  void collect_locked(); // caller holds the page-state mutex
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Long-lived allocation owner; never auto-reclaims a dead client's chunks.
int run_shared_kv_owner(const std::string& endpoint, uint64_t bytes, int device);
}  // namespace base
