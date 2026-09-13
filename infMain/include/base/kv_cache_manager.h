// KVCacheManager: request-level KV cache management for multi-sequence batching
#ifndef KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
#define KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_

#include <cstdint>
#include <map>
#include <limits>
#include <set>
#include <vector>
#include "base/block_allocator.h"
#include "base/compressed_radix_cache_tree.h"
#include "base/sequence_kv_manager.h"
#include "cache/page_directory.h"
#include "cache/page_migration.h"
#include "cache/recovery_dependency.h"
#include "cache/transfer_scheduler.h"

namespace base {

using RequestId = int64_t;

struct KVAllocationIntent {
  RequestId request_id = -1;
  int32_t expected_allocated_tokens = 0;
  int32_t append_tokens = 0;
};

struct RadixCacheStats {
  int64_t lookup_requests = 0;
  int64_t cache_hits = 0;
  int64_t cache_misses = 0;
  int64_t tokens_reused = 0;
  int64_t publish_requests = 0;
  int64_t published_blocks = 0;
  int64_t evictions = 0;
  int64_t evicted_blocks = 0;
  int64_t host_demoted_blocks = 0;
  int64_t host_restore_requests = 0;
  int64_t host_restored_blocks = 0;
  int64_t host_restore_failures = 0;
  int64_t recovery_graph_demote_decisions = 0;
  int64_t recovery_graph_keep_decisions = 0;
  int64_t recovery_graph_committed_demotions = 0;
  int64_t recovery_graph_restores = 0;
  int64_t recovery_graph_publish_rejections = 0;
};

struct RadixPrefixProbe {
  int32_t prompt_tokens = 0;
  int32_t matched_tokens = 0;
  int32_t gpu_pages = 0;
  int32_t host_pages = 0;
  int32_t missing_pages = 0;
  bool fully_serviceable = false;
};

struct HostCacheConfig {
  bool enabled = false;
  size_t byte_capacity = 0;
  size_t page_capacity = 0;
  size_t max_inflight_pages = 0;
  bool direction_lanes_enabled = true;
};

struct KVBlockSnapshot {
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scale;
  std::vector<uint8_t> value_scale;
};

struct KVLayerSnapshot {
  BlockStorageMode storage_mode = BlockStorageMode::kPlain;
  DataType storage_dtype = DataType::kDataTypeUnknown;
  DataType scale_dtype = DataType::kDataTypeUnknown;
  std::vector<KVBlockSnapshot> blocks;
};

// Immutable, process-local checkpoint payload. valid_tokens makes the last
// private page explicit; unused bytes are retained only for a bitwise restore.
struct KVRequestSnapshot {
  uint64_t schema_version = 1;
  int32_t block_size = 0;
  int32_t num_layers = 0;
  int32_t valid_tokens = 0;
  int32_t committed_tokens = 0;
  std::vector<KVLayerSnapshot> layers;
};

struct ExternalKVRequestState {
  uint64_t schema_version = 1;
  int32_t block_size = 0;
  int32_t num_layers = 0;
  int32_t valid_tokens = 0;
  int32_t committed_tokens = 0;
  std::vector<std::vector<int32_t>> block_ids_per_layer;
};

enum class RequestRestoreState { kNone, kPending, kReady, kFailed };

struct BranchTokenBoundaries {
  int32_t computed_tokens = 0;
  int32_t sampled_tokens = 0;
  int32_t pending_tokens = 0;
};

struct BranchSnapshot {
  RequestId source_request_id = -1;
  int32_t valid_tokens = 0;
  int32_t committed_tokens = 0;
  BranchTokenBoundaries boundaries;
  int32_t first_token = -1;
  bool inherits_first_token = false;
  std::vector<std::vector<int32_t>> block_ids_per_layer;
};

class KVCacheManager {
 public:
  KVCacheManager(int32_t block_size, int32_t num_layers,
                 std::vector<std::unique_ptr<BlockAllocator>> layer_allocators,
                 HostCacheConfig host_cache = {});

  // Register a new request, returns its ID
  RequestId register_request();
  RequestId register_request_with_radix_cache(const std::vector<int32_t>& prompt_tokens);

  // Free all blocks held by a request
  void free_request(RequestId id);
  void publish_radix_cache(RequestId id, const std::vector<int32_t>& prompt_tokens);
  // Caller must observe successful producer completion before committing.
  bool commit_kv(RequestId id, int32_t tokens);
  int32_t kv_committed_tokens(RequestId id) const;
  int32_t published_full_tokens(RequestId id) const;
  base::Status acquire_compute_leases(RequestId id, std::vector<BlockLease>* leases);
  bool snapshot_request(RequestId id, KVRequestSnapshot* snapshot) const;
  // Computes the memory owned by a snapshot before allocating or copying it.
  bool snapshot_request_bytes(RequestId id, size_t* bytes) const;
  bool restore_request_snapshot(const KVRequestSnapshot& snapshot,
                                RequestId* restored_id);
  bool export_external_request(RequestId id, ExternalKVRequestState* state) const;
  bool restore_external_request(const ExternalKVRequestState& state,
                                RequestId* restored_id);
  // Persistent decode workers borrow immutable Prefill pages repeatedly. The
  // binding/provider retains its baseline reference across request teardown.
  bool restore_external_shared_request(const ExternalKVRequestState& state,
                                       RequestId* restored_id);

  // Same-pool fork. All pages are initially shared; appending to a partial
  // tail triggers device-local COW before the writable slot is returned.
  bool fork_request(RequestId source_id, const BranchTokenBoundaries& boundaries,
                    bool inherits_first_token, int32_t first_token,
                    RequestId* branch_id, BranchSnapshot* snapshot = nullptr);

  void clear_radix_cache();

  // Append one token slot for a request (allocates across all layers)
  bool append_slot(RequestId id);
  bool plan_allocation(RequestId id, int32_t tokens, KVAllocationIntent* intent) const;
  bool admit_allocation(const KVAllocationIntent& intent);

  // Append multiple token slots for a request at once
  bool append_slots(RequestId id, int32_t num_tokens);

  // Get slot (block_id, offset) for a specific token position in a layer
  std::pair<int32_t, int32_t> get_slot(RequestId id, int32_t layer_idx,
                                       int32_t token_pos) const;

  // Get block IDs for a request at a given layer
  const std::vector<int32_t>& get_block_ids(RequestId id, int32_t layer_idx) const;

  // Get context length (number of tokens cached) for a request
  int32_t get_context_len(RequestId id) const;

  // Get tokens in last block for a request
  int32_t num_tokens_in_last_block(RequestId id) const;

  // Get current write slot (block_id, offset_in_block) for a layer
  std::pair<int32_t, int32_t> current_slot(RequestId id, int32_t layer_idx) const;

  // Build slot_mapping for a batch of requests (one token per request)
  // slot = block_id * block_size + offset_in_block
  // Returns CPU vector of slot indices, one per request per layer
  std::vector<int32_t> build_slot_mapping(const std::vector<RequestId>& request_ids,
                                          int32_t layer_idx) const;

  // Build block_tables [batch_size, max_blocks_per_seq] for a batch (padded with -1)
  // Also returns max_blocks_per_seq
  std::pair<std::vector<int32_t>, int32_t> build_block_tables(
      const std::vector<RequestId>& request_ids, int32_t layer_idx) const;

  // Build seq_lens [batch_size] for a batch
  std::vector<int32_t> build_seq_lens(const std::vector<RequestId>& request_ids) const;

  // Access allocator for a layer (to pass pool to kernels)
  const BlockAllocator& allocator(int32_t layer_idx) const;
  BlockAllocator& allocator_mut(int32_t layer_idx);

  int32_t block_size() const { return block_size_; }
  int32_t num_layers() const { return num_layers_; }
  int32_t num_free_blocks(int32_t layer_idx) const;
  int32_t num_total_blocks(int32_t layer_idx) const;
  uint64_t total_block_copy_bytes() const;
  const KVAppendStats& append_stats(RequestId id) const;
  BranchTokenBoundaries branch_boundaries(RequestId id) const;
  uint64_t branch_generation(RequestId id) const;
  const RadixCacheStats& radix_cache_stats() const { return radix_cache_stats_; }
  void reset_radix_cache_stats() { radix_cache_stats_ = RadixCacheStats{}; }
  void set_radix_cache_enabled(bool enabled);
  int32_t radix_cache_node_count() const { return radix_cache_.node_count(); }
  int32_t radix_cache_split_count() const { return radix_cache_.split_count(); }
  int32_t radix_cache_evictable_blocks() const;
  bool is_valid_request(RequestId id) const;
  bool radix_cache_enabled() const { return radix_cache_enabled_; }
  RadixPrefixProbe probe_radix_cache(
      const std::vector<int32_t>& prompt_tokens) const;
  bool host_cache_enabled() const { return migration_engine_ != nullptr; }
  RequestRestoreState request_restore_state(RequestId id) const;
  // Progresses completions even if no request can run. Returns completed jobs.
  int32_t service_cache_transfers();
  bool drain_cache_transfers();
  bool cache_transfers_pending() const {
    return transfer_scheduler_ && transfer_scheduler_->flight_count() != 0;
  }
  const cache::TransferSchedulerStats* transfer_scheduler_stats() const {
    return transfer_scheduler_ ? &transfer_scheduler_->stats() : nullptr;
  }
  size_t recovery_dependency_object_count() const {
    return recovery_graph_.size();
  }
  int32_t schedulable_free_blocks(int32_t layer_idx) const;
  size_t restore_reserved_target_pages() const {
    return restore_target_reservations_.size();
  }
  // Test/maintenance policy hook: demotes currently evictable radix pages.
  int32_t demote_radix_cache_to_host(
      int32_t max_pages = std::numeric_limits<int32_t>::max());
  int32_t request_slot_capacity() const {
    return static_cast<int32_t>(request_slots_.size());
  }
  int32_t num_active_requests() const { return num_active_requests_; }

 private:
  friend struct KVCacheManagerTestPeer;

  struct RequestSlot {
    RequestSlot(int32_t block_size, int32_t num_layers)
        : manager(block_size, num_layers) {}

    SequenceKVManager manager;
    uint64_t generation = 0;
    int32_t kv_committed_tokens = 0;
    int32_t published_full_tokens = 0;
    bool active = false;
    LogicalRadixCacheTree::Node* radix_cache_leaf = nullptr;
    int32_t radix_cache_shared_tokens = 0;
    RequestRestoreState restore_state = RequestRestoreState::kNone;
    std::vector<cache::LogicalPageId> restore_pages;
    std::vector<cache::LogicalPageId> recovery_pages;
    std::vector<int32_t> restore_prefix_tokens;
    size_t restore_next_page = 0;
    cache::WaiterTicket restore_waiter;
    int32_t restore_tokens = 0;
    BranchTokenBoundaries branch_boundaries;
    uint64_t branch_generation = 0;
    int32_t branch_first_token = -1;
    bool inherits_first_token = false;
  };

  static constexpr uint32_t kRequestIdBits = 63;
  static constexpr uint32_t kRequestSlotBits = 20;
  static constexpr uint32_t kRequestGenerationBits =
      kRequestIdBits - kRequestSlotBits;
  static constexpr uint32_t kRequestSlotMask = (1u << kRequestSlotBits) - 1u;
  static constexpr uint32_t kMaxRequestSlots = 1u << kRequestSlotBits;
  static constexpr uint64_t kMaxRequestGeneration =
      (uint64_t{1} << kRequestGenerationBits) - 1u;

  static RequestId encode_request_id(uint32_t slot_idx, uint64_t generation);
  static uint32_t decode_slot_index(RequestId id);
  static uint64_t decode_generation(RequestId id);

  bool try_decode_request_id(RequestId id, uint32_t* slot_idx,
                             uint64_t* generation) const;
  int32_t full_blocks_for_tokens(int32_t num_tokens) const;
  int32_t additional_blocks_needed(int32_t current_tokens, int32_t appended_tokens) const;
  int32_t min_free_blocks_across_layers() const;
  std::vector<std::vector<int32_t>> full_block_ids_for_request(
      const RequestSlot& slot, int32_t full_blocks) const;
  void maybe_attach_radix_cache(RequestSlot* slot,
                                const std::vector<int32_t>& prompt_tokens);
  bool start_next_restore(RequestSlot* slot);
  bool finish_restore(RequestSlot* slot);
  bool reserve_restore_targets(const std::vector<cache::LogicalPageId>& pages);
  void reconcile_restore_target_reservations();
  void maybe_unpin_radix_cache_path(RequestSlot* slot);
  bool ensure_recovery_object(cache::LogicalPageId id);
  bool retire_recovery_object(cache::LogicalPageId id);
  static std::string recovery_object_id(cache::LogicalPageId id);
  bool ensure_free_blocks_available(int32_t required_blocks_per_layer);
  int32_t evict_radix_cache_blocks(int32_t min_blocks_to_evict);
  RequestSlot& lookup_request_slot(RequestId id, uint32_t* slot_idx = nullptr);
  const RequestSlot& lookup_request_slot(RequestId id,
                                         uint32_t* slot_idx = nullptr) const;
  SequenceKVManager& lookup_request(RequestId id);
  const SequenceKVManager& lookup_request(RequestId id) const;

  int32_t block_size_;
  int32_t num_layers_;
  std::vector<std::unique_ptr<BlockAllocator>> layer_allocators_;
  std::unique_ptr<cache::HostStore> host_store_;
  std::unique_ptr<cache::PageDirectory> page_directory_;
  std::unique_ptr<cache::PageMigrationEngine> migration_engine_;
  std::unique_ptr<cache::TransferScheduler> transfer_scheduler_;
  std::map<cache::LogicalPageId, cache::WaiterTicket> demotions_;
  std::vector<cache::WaiterId> orphaned_restores_;
  // One physical target per unique host-only logical page. Shared restore
  // waiters consume one reservation, not one reservation per request.
  std::set<cache::LogicalPageId> restore_target_reservations_;
  cache::PageSchema page_schema_;
  cache::RecoveryDependencyGraph recovery_graph_;
  size_t host_cache_capacity_bytes_ = 0;
  size_t recovery_page_bytes_ = 0;
  LogicalRadixCacheTree radix_cache_;
  RadixCacheStats radix_cache_stats_;
  std::vector<RequestSlot> request_slots_;
  std::vector<uint32_t> free_slot_indices_;
  int32_t num_active_requests_ = 0;
  bool radix_cache_enabled_ = true;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
