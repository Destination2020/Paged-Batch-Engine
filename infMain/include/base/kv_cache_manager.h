// KVCacheManager: request-level KV cache management for multi-sequence batching
#ifndef KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
#define KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_

#include <cstdint>
#include <vector>
#include "base/block_allocator.h"
#include "base/compressed_radix_cache_tree.h"
#include "base/sequence_kv_manager.h"

namespace base {

using RequestId = int32_t;

struct RadixCacheStats {
  int64_t lookup_requests = 0;
  int64_t cache_hits = 0;
  int64_t cache_misses = 0;
  int64_t tokens_reused = 0;
  int64_t publish_requests = 0;
  int64_t published_blocks = 0;
  int64_t evictions = 0;
  int64_t evicted_blocks = 0;
};

class KVCacheManager {
 public:
  KVCacheManager(int32_t block_size, int32_t num_layers,
                 std::vector<std::unique_ptr<BlockAllocator>> layer_allocators);

  // Register a new request, returns its ID
  RequestId register_request();
  RequestId register_request_with_radix_cache(const std::vector<int32_t>& prompt_tokens);

  // Free all blocks held by a request
  void free_request(RequestId id);
  void publish_radix_cache(RequestId id, const std::vector<int32_t>& prompt_tokens);
  void clear_radix_cache();

  // Append one token slot for a request (allocates across all layers)
  bool append_slot(RequestId id);

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
  const KVAppendStats& append_stats(RequestId id) const;
  const RadixCacheStats& radix_cache_stats() const { return radix_cache_stats_; }
  void reset_radix_cache_stats() { radix_cache_stats_ = RadixCacheStats{}; }
  void set_radix_cache_enabled(bool enabled);
  int32_t radix_cache_node_count() const { return radix_cache_.node_count(); }
  int32_t radix_cache_split_count() const { return radix_cache_.split_count(); }
  int32_t radix_cache_evictable_blocks() const;
  bool is_valid_request(RequestId id) const;
  bool radix_cache_enabled() const { return radix_cache_enabled_; }
  int32_t request_slot_capacity() const {
    return static_cast<int32_t>(request_slots_.size());
  }
  int32_t num_active_requests() const { return num_active_requests_; }

 private:
  struct RequestSlot {
    RequestSlot(int32_t block_size, int32_t num_layers)
        : manager(block_size, num_layers) {}

    SequenceKVManager manager;
    uint32_t generation = 0;
    bool active = false;
    CompressedRadixCacheTree::Node* radix_cache_leaf = nullptr;
    int32_t radix_cache_shared_tokens = 0;
  };

  static constexpr uint32_t kRequestIdBits = 31;
  static constexpr uint32_t kRequestSlotBits = 20;
  static constexpr uint32_t kRequestGenerationBits =
      kRequestIdBits - kRequestSlotBits;
  static constexpr uint32_t kRequestSlotMask = (1u << kRequestSlotBits) - 1u;
  static constexpr uint32_t kMaxRequestSlots = 1u << kRequestSlotBits;
  static constexpr uint32_t kMaxRequestGeneration =
      (1u << kRequestGenerationBits) - 1u;

  static RequestId encode_request_id(uint32_t slot_idx, uint32_t generation);
  static uint32_t decode_slot_index(RequestId id);
  static uint32_t decode_generation(RequestId id);

  bool try_decode_request_id(RequestId id, uint32_t* slot_idx,
                             uint32_t* generation) const;
  int32_t full_blocks_for_tokens(int32_t num_tokens) const;
  int32_t additional_blocks_needed(int32_t current_tokens, int32_t appended_tokens) const;
  int32_t min_free_blocks_across_layers() const;
  std::vector<std::vector<int32_t>> full_block_ids_for_request(
      const RequestSlot& slot, int32_t full_blocks) const;
  void release_block_ids(const std::vector<std::vector<int32_t>>& block_ids_per_layer);
  void release_block_ids(const std::vector<std::vector<int32_t>>& block_ids_per_layer,
                         int32_t start_block,
                         int32_t block_count);
  void retain_block_ids(const std::vector<std::vector<int32_t>>& block_ids_per_layer,
                        int32_t start_block,
                        int32_t block_count);
  void maybe_attach_radix_cache(RequestSlot* slot,
                                const std::vector<int32_t>& prompt_tokens);
  void maybe_unpin_radix_cache_path(RequestSlot* slot);
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
  CompressedRadixCacheTree radix_cache_;
  RadixCacheStats radix_cache_stats_;
  std::vector<RequestSlot> request_slots_;
  std::vector<uint32_t> free_slot_indices_;
  int32_t num_active_requests_ = 0;
  bool radix_cache_enabled_ = true;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
