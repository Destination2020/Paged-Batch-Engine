// KVCacheManager: request-level KV cache management for multi-sequence batching
#ifndef KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
#define KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "base/block_allocator.h"
#include "base/sequence_kv_manager.h"

namespace base {

using RequestId = int32_t;

class KVCacheManager {
 public:
  KVCacheManager(int32_t block_size, int32_t num_layers,
                 std::vector<std::unique_ptr<BlockAllocator>> layer_allocators);

  // Register a new request, returns its ID
  RequestId register_request();

  // Free all blocks held by a request
  void free_request(RequestId id);

  // Append one token slot for a request (allocates across all layers)
  bool append_slot(RequestId id);

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

 private:
  int32_t block_size_;
  int32_t num_layers_;
  RequestId next_id_ = 0;
  std::vector<std::unique_ptr<BlockAllocator>> layer_allocators_;
  std::unordered_map<RequestId, SequenceKVManager> requests_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
