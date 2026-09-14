// KVCacheManager implementation
#include "base/kv_cache_manager.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <glog/logging.h>

namespace base {

namespace {

void copy_payload_to_host(const BlockAllocator& allocator, const void* source,
                          void* destination, size_t bytes) {
  if (bytes == 0) return;
  if (allocator.device_type() == DeviceType::kDeviceCPU) {
    std::memcpy(destination, source, bytes);
    return;
  }
#ifndef KUIPER_CPU_ONLY
  CUDADeviceAllocatorFactory::get_instance()->memcpy(
      source, destination, bytes, MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);
#else
  LOG(FATAL) << "CUDA checkpoint snapshot in CPU-only build";
#endif
}

void copy_payload_from_host(const BlockAllocator& allocator, const void* source,
                            void* destination, size_t bytes) {
  if (bytes == 0) return;
  if (allocator.device_type() == DeviceType::kDeviceCPU) {
    std::memcpy(destination, source, bytes);
    return;
  }
#ifndef KUIPER_CPU_ONLY
  CUDADeviceAllocatorFactory::get_instance()->memcpy(
      source, destination, bytes, MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
#else
  LOG(FATAL) << "CUDA checkpoint restore in CPU-only build";
#endif
}

}  // namespace

RequestId KVCacheManager::encode_request_id(uint32_t slot_idx,
                                            uint64_t generation) {
  CHECK_LT(slot_idx, kMaxRequestSlots) << "slot index overflow";
  CHECK_LE(generation, kMaxRequestGeneration) << "request generation overflow";
  const uint64_t raw = (generation << kRequestSlotBits) | slot_idx;
  CHECK_LE(raw, static_cast<uint64_t>(INT64_MAX));
  return static_cast<RequestId>(raw);
}
 
uint32_t KVCacheManager::decode_slot_index(RequestId id) {
  return static_cast<uint32_t>(id) & kRequestSlotMask;
}

uint64_t KVCacheManager::decode_generation(RequestId id) {
  return static_cast<uint64_t>(id) >> kRequestSlotBits;
}

KVCacheManager::KVCacheManager(int32_t block_size, int32_t num_layers,
                               std::vector<std::unique_ptr<BlockAllocator>> layer_allocators,
                               HostCacheConfig host_cache)
    : block_size_(block_size),
      num_layers_(num_layers),
      layer_allocators_(std::move(layer_allocators)),
      radix_cache_(block_size, 1) {
  CHECK_GT(block_size_, 0);
  CHECK_GT(num_layers_, 0);
  CHECK_EQ(static_cast<int32_t>(layer_allocators_.size()), num_layers_);
  std::vector<BlockAllocator*> pools;
  size_t capacity = static_cast<size_t>(layer_allocators_.front()->num_total_blocks());
  for (auto& pool : layer_allocators_) {
    pools.push_back(pool.get());
    capacity = std::min(capacity, static_cast<size_t>(pool->num_total_blocks()));
  }
  page_directory_ = std::make_unique<cache::PageDirectory>(pools, capacity);
  const auto& pool = *pools.front();
  STATUS_CHECK(cache::MakeKVPageSchema(1, num_layers_, block_size_, pool.num_kv_heads(),
                                     pool.head_size(), pool.storage_spec(), &page_schema_));
  cache::RecoveryObject schema_dependency;
  schema_dependency.object_id = "kv-schema";
  schema_dependency.version = page_schema_.fingerprint();
  schema_dependency.kind = cache::RecoveryKind::kPreserveUntilReleased;
  schema_dependency.gpu_serviceable = true;
  CHECK(recovery_graph_.publish(std::move(schema_dependency)));
  if (host_cache.enabled) {
    CHECK_GT(host_cache.byte_capacity, 0u);
    CHECK_GT(host_cache.page_capacity, 0u);
    CHECK_GT(host_cache.max_inflight_pages, 0u);
    cache::HostStore::Allocate allocate;
    if (pool.device_type() == DeviceType::kDeviceCUDA) {
#ifndef KUIPER_CPU_ONLY
      auto pinned = PinnedCPUDeviceAllocatorFactory::get_instance();
      allocate = [pinned](size_t bytes) {
        return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(pinned->allocate(bytes)),
            [pinned](uint8_t* pointer) { pinned->release(pointer); });
      };
#else
      LOG(FATAL) << "CUDA host cache is unavailable in a CPU-only build";
#endif
    }
    host_store_ = std::make_unique<cache::HostStore>(
        host_cache.byte_capacity, host_cache.page_capacity, std::move(allocate));
    host_cache_capacity_bytes_ = host_cache.byte_capacity;
    for (const auto& component : page_schema_.components)
      recovery_page_bytes_ += component.byte_size();
    std::unique_ptr<cache::TransferBackend> backend;
    if (pool.device_type() == DeviceType::kDeviceCUDA) {
#ifndef KUIPER_CPU_ONLY
      backend = cache::MakeCudaTransferBackend(host_cache.max_inflight_pages, -1);
#endif
    } else {
      backend = cache::MakeCpuTransferBackend();
    }
    migration_engine_ = std::make_unique<cache::PageMigrationEngine>(
        pools, *page_directory_, *host_store_, page_schema_,
        host_cache.max_inflight_pages, host_cache.byte_capacity, std::move(backend));
    const size_t max_flights = std::max(host_cache.page_capacity,
                                        host_cache.max_inflight_pages);
    const size_t max_waiters = max_flights > SIZE_MAX / 64 ? SIZE_MAX : max_flights * 64;
    const size_t total_active = std::min(max_flights, host_cache.max_inflight_pages);
    // With lanes enabled, reserve half of the physical concurrency for each
    // direction. Disabling lanes is the only switch in the scheduler A/B.
    const size_t direction_active = std::max<size_t>(1, total_active / 2);
    cache::TransferSchedulerConfig scheduler_config{
        max_flights, max_waiters, total_active,
        direction_active, direction_active, direction_active, 64,
        max_flights > 1 ? 1u : 0u, max_waiters > 1 ? std::min<size_t>(32, max_waiters - 1) : 0u,
        host_cache.direction_lanes_enabled};
    transfer_scheduler_ = std::make_unique<cache::TransferScheduler>(
        scheduler_config, cache::MakePageMigrationFlightExecutor(*migration_engine_));
  }
}

RadixPrefixProbe KVCacheManager::probe_radix_cache(
    const std::vector<int32_t>& prompt_tokens) const {
  RadixPrefixProbe output;
  output.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  if (!radix_cache_enabled_) return output;
  const int32_t reusable_blocks =
      std::max(0, output.prompt_tokens - 1) / block_size_;
  if (reusable_blocks <= 0) return output;
  output.missing_pages = reusable_blocks;
  std::vector<int32_t> reusable(prompt_tokens.begin(),
      prompt_tokens.begin() + reusable_blocks * block_size_);
  const auto match = radix_cache_.probe_prefix(reusable);
  if (match.matched_blocks <= 0 || match.block_ids_per_layer.empty()) return output;
  const auto& pages = match.block_ids_per_layer[0];
  for (size_t index = 0; index < pages.size(); ++index) {
    const auto id = pages[index];
    if (page_directory_->has_gpu(id)) {
      ++output.gpu_pages;
      output.matched_tokens += block_size_;
    } else if (migration_engine_ && page_directory_->has_host(id)) {
      ++output.host_pages;
      output.matched_tokens += block_size_;
    } else {
      break;
    }
  }
  output.missing_pages = reusable_blocks - output.matched_tokens / block_size_;
  output.fully_serviceable = output.missing_pages == 0;
  return output;
}

RequestId KVCacheManager::register_request() {
  uint32_t slot_idx = 0;
  RequestSlot* slot = nullptr;
  if (!free_slot_indices_.empty()) {
    slot_idx = free_slot_indices_.back();
    free_slot_indices_.pop_back();
    slot = &request_slots_[slot_idx];
    CHECK(!slot->active);
    CHECK_LT(slot->generation, kMaxRequestGeneration)
        << "request slot generation overflow at slot " << slot_idx;
    slot->generation++;
    slot->manager.clear();
    slot->kv_committed_tokens = 0;
    slot->published_full_tokens = 0;
    slot->radix_cache_leaf = nullptr;
    slot->radix_cache_shared_tokens = 0;
    slot->restore_state = RequestRestoreState::kNone;
    slot->restore_pages.clear();
    slot->recovery_pages.clear();
    slot->restore_prefix_tokens.clear();
    slot->restore_next_page = 0;
    slot->restore_waiter = {};
    slot->restore_tokens = 0;
    slot->branch_boundaries = {};
    slot->branch_generation = 0;
    slot->branch_first_token = -1;
    slot->inherits_first_token = false;
  } else {
    CHECK_LT(request_slots_.size(), static_cast<size_t>(kMaxRequestSlots))
        << "request slot capacity exhausted";
    slot_idx = static_cast<uint32_t>(request_slots_.size());
    request_slots_.emplace_back(block_size_, num_layers_);
    slot = &request_slots_.back();
  }

  slot->active = true;
  num_active_requests_++;
  return encode_request_id(slot_idx, slot->generation);
}

RequestId KVCacheManager::register_request_with_radix_cache(
    const std::vector<int32_t>& prompt_tokens) {
  const RequestId id = register_request();
  RequestSlot& slot = lookup_request_slot(id);
  maybe_attach_radix_cache(&slot, prompt_tokens);
  return id;
}

void KVCacheManager::free_request(RequestId id) {
  uint32_t slot_idx = 0;
  auto& slot = lookup_request_slot(id, &slot_idx);
  maybe_unpin_radix_cache_path(&slot);
  if (slot.restore_waiter.waiter && transfer_scheduler_) {
    transfer_scheduler_->cancel(slot.restore_waiter.waiter);
    orphaned_restores_.push_back(slot.restore_waiter.waiter);
    slot.restore_waiter = {};
  }
  slot.restore_pages.clear();
  slot.restore_prefix_tokens.clear();
  slot.restore_state = RequestRestoreState::kNone;
  slot.manager.release_all(layer_allocators_);
  slot.active = false;
  reconcile_restore_target_reservations();
  // Retire exhausted slots permanently: an old handle must never alias a new request.
  if (slot.generation < kMaxRequestGeneration) {
    free_slot_indices_.push_back(slot_idx);
  }
  CHECK_GT(num_active_requests_, 0);
  num_active_requests_--;
}

bool KVCacheManager::commit_kv(RequestId id, int32_t tokens) {
  if (!is_valid_request(id)) return false;
  auto& slot = lookup_request_slot(id);
  if (tokens < slot.kv_committed_tokens || tokens > slot.manager.num_tokens()) return false;
  slot.kv_committed_tokens = tokens;
  return true;
}

bool KVCacheManager::snapshot_request(RequestId id,
                                      KVRequestSnapshot* snapshot) const {
  if (!snapshot || !is_valid_request(id)) return false;
  const auto& slot = lookup_request_slot(id);
  KVRequestSnapshot pending;
  pending.block_size = block_size_;
  pending.num_layers = num_layers_;
  pending.valid_tokens = slot.manager.num_tokens();
  pending.committed_tokens = slot.kv_committed_tokens;
  pending.layers.resize(num_layers_);
  const size_t expected_blocks = pending.valid_tokens == 0
      ? 0 : static_cast<size_t>((pending.valid_tokens + block_size_ - 1) / block_size_);
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    const auto& allocator = *layer_allocators_[layer];
    const auto& block_ids = get_block_ids(id, layer);
    if (block_ids.size() != expected_blocks) return false;
    auto& layer_snapshot = pending.layers[layer];
    layer_snapshot.storage_mode = allocator.storage_mode();
    layer_snapshot.storage_dtype = allocator.storage_dtype();
    layer_snapshot.scale_dtype = allocator.scale_dtype();
    layer_snapshot.blocks.reserve(block_ids.size());
    for (int32_t block_id : block_ids) {
      const auto payload = allocator.get_block_payload_ptrs(block_id);
      KVBlockSnapshot block;
      block.key.resize(payload.key_value_bytes);
      block.value.resize(payload.key_value_bytes);
      block.key_scale.resize(payload.scale_bytes);
      block.value_scale.resize(payload.scale_bytes);
      copy_payload_to_host(allocator, payload.key, block.key.data(), block.key.size());
      copy_payload_to_host(allocator, payload.value, block.value.data(), block.value.size());
      copy_payload_to_host(allocator, payload.key_scale, block.key_scale.data(),
                           block.key_scale.size());
      copy_payload_to_host(allocator, payload.value_scale, block.value_scale.data(),
                           block.value_scale.size());
      layer_snapshot.blocks.push_back(std::move(block));
    }
  }
  *snapshot = std::move(pending);
  return true;
}

bool KVCacheManager::snapshot_request_bytes(RequestId id, size_t* bytes) const {
  if (!bytes || !is_valid_request(id)) return false;
  const auto& slot = lookup_request_slot(id);
  const int32_t valid_tokens = slot.manager.num_tokens();
  const size_t block_count = valid_tokens == 0
      ? 0 : static_cast<size_t>((valid_tokens + block_size_ - 1) / block_size_);
  size_t total = sizeof(KVRequestSnapshot);
  const auto add = [&total](size_t value) {
    if (value > std::numeric_limits<size_t>::max() - total) return false;
    total += value;
    return true;
  };
  const auto multiply = [](size_t lhs, size_t rhs, size_t* value) {
    if (!value || (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs))
      return false;
    *value = lhs * rhs;
    return true;
  };
  size_t layer_metadata = 0;
  if (!multiply(static_cast<size_t>(num_layers_), sizeof(KVLayerSnapshot),
                &layer_metadata) || !add(layer_metadata)) return false;
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    const auto& allocator = *layer_allocators_[layer];
    const auto& block_ids = get_block_ids(id, layer);
    if (block_ids.size() != block_count) return false;
    size_t block_metadata = 0;
    if (!multiply(block_count, sizeof(KVBlockSnapshot), &block_metadata) ||
        !add(block_metadata)) return false;
    const size_t component_bytes = allocator.key_value_bytes_per_block();
    const size_t scale_bytes = allocator.scale_bytes_per_block();
    if (component_bytes > std::numeric_limits<size_t>::max() - scale_bytes)
      return false;
    const size_t one_pair = component_bytes + scale_bytes;
    if (one_pair > std::numeric_limits<size_t>::max() / 2) return false;
    size_t payload = 0;
    if (!multiply(block_count, one_pair * 2, &payload) || !add(payload))
      return false;
  }
  *bytes = total;
  return true;
}

bool KVCacheManager::restore_request_snapshot(const KVRequestSnapshot& snapshot,
                                              RequestId* restored_id) {
  if (!restored_id || snapshot.schema_version != 1 ||
      snapshot.block_size != block_size_ || snapshot.num_layers != num_layers_ ||
      snapshot.valid_tokens < 0 || snapshot.committed_tokens < 0 ||
      snapshot.committed_tokens > snapshot.valid_tokens ||
      snapshot.layers.size() != static_cast<size_t>(num_layers_)) return false;
  const size_t expected_blocks = snapshot.valid_tokens == 0
      ? 0 : static_cast<size_t>((snapshot.valid_tokens + block_size_ - 1) / block_size_);
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    const auto& allocator = *layer_allocators_[layer];
    const auto& source = snapshot.layers[layer];
    if (source.storage_mode != allocator.storage_mode() ||
        source.storage_dtype != allocator.storage_dtype() ||
        source.scale_dtype != allocator.scale_dtype() ||
        source.blocks.size() != expected_blocks) return false;
    for (const auto& block : source.blocks) {
      if (block.key.size() != allocator.key_value_bytes_per_block() ||
          block.value.size() != allocator.key_value_bytes_per_block() ||
          block.key_scale.size() != allocator.scale_bytes_per_block() ||
          block.value_scale.size() != allocator.scale_bytes_per_block()) return false;
    }
  }

  const RequestId id = register_request();
  if (snapshot.valid_tokens > 0 && !append_slots(id, snapshot.valid_tokens)) {
    free_request(id);
    return false;
  }
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    const auto& allocator = *layer_allocators_[layer];
    const auto& block_ids = get_block_ids(id, layer);
    for (size_t index = 0; index < block_ids.size(); ++index) {
      const auto payload = allocator.get_block_payload_ptrs(block_ids[index]);
      const auto& block = snapshot.layers[layer].blocks[index];
      copy_payload_from_host(allocator, block.key.data(), payload.key, block.key.size());
      copy_payload_from_host(allocator, block.value.data(), payload.value, block.value.size());
      copy_payload_from_host(allocator, block.key_scale.data(), payload.key_scale,
                             block.key_scale.size());
      copy_payload_from_host(allocator, block.value_scale.data(), payload.value_scale,
                             block.value_scale.size());
    }
  }
  auto& slot = lookup_request_slot(id);
  slot.kv_committed_tokens = snapshot.committed_tokens;
  slot.published_full_tokens = 0;
  *restored_id = id;
  return true;
}

bool KVCacheManager::export_external_request(RequestId id,
                                             ExternalKVRequestState* state) const {
  if(!state||!is_valid_request(id))return false;
  const auto& slot=lookup_request_slot(id);
  state->schema_version=1;state->block_size=block_size_;state->num_layers=num_layers_;
  state->valid_tokens=slot.manager.num_tokens();
  state->committed_tokens=slot.kv_committed_tokens;
  state->block_ids_per_layer.resize(num_layers_);
  for(int32_t layer=0;layer<num_layers_;++layer)
    state->block_ids_per_layer[layer]=slot.manager.page_table(layer).block_ids();
  return true;
}

bool KVCacheManager::restore_external_request(const ExternalKVRequestState& state,
                                              RequestId* restored_id) {
  if(!restored_id||state.schema_version!=1||state.block_size!=block_size_||
     state.num_layers!=num_layers_||state.valid_tokens<0||
     state.committed_tokens<0||state.committed_tokens>state.valid_tokens||
     state.block_ids_per_layer.size()!=static_cast<size_t>(num_layers_))return false;
  const size_t expected=state.valid_tokens==0?0:
      static_cast<size_t>((state.valid_tokens+block_size_-1)/block_size_);
  for(int32_t layer=0;layer<num_layers_;++layer){
    if(state.block_ids_per_layer[layer].size()!=expected)return false;
    for(int32_t id:state.block_ids_per_layer[layer])
      if(id<0||id>=layer_allocators_[layer]->num_total_blocks()||
         layer_allocators_[layer]->ref_count(id)<=0)return false;
  }
  const RequestId id=register_request();
  auto& slot=lookup_request_slot(id);
  slot.manager.adopt_owned_pages(layer_allocators_,state.block_ids_per_layer,
                                 state.valid_tokens);
  slot.kv_committed_tokens=state.committed_tokens;
  slot.published_full_tokens=0;*restored_id=id;return true;
}

bool KVCacheManager::restore_external_shared_request(
    const ExternalKVRequestState& state, RequestId* restored_id) {
  if(!restored_id||state.schema_version!=1||state.block_size!=block_size_||
     state.num_layers!=num_layers_||state.valid_tokens<0||
     state.committed_tokens<0||state.committed_tokens>state.valid_tokens||
     state.block_ids_per_layer.size()!=static_cast<size_t>(num_layers_))return false;
  const size_t expected=state.valid_tokens==0?0:
      static_cast<size_t>((state.valid_tokens+block_size_-1)/block_size_);
  for(int32_t layer=0;layer<num_layers_;++layer){
    if(state.block_ids_per_layer[layer].size()!=expected)return false;
    for(int32_t id:state.block_ids_per_layer[layer])
      if(id<0||id>=layer_allocators_[layer]->num_total_blocks()||
         layer_allocators_[layer]->ref_count(id)<=0)return false;
  }
  const RequestId id=register_request();
  auto& slot=lookup_request_slot(id);
  slot.manager.adopt_branch_snapshot(layer_allocators_,state.block_ids_per_layer,
                                     state.valid_tokens);
  slot.kv_committed_tokens=state.committed_tokens;
  slot.published_full_tokens=0;*restored_id=id;return true;
}

bool KVCacheManager::attach_external_shared_pages(
    const ExternalKVRequestState& state) {
  if (state.schema_version != 1 || state.block_size != block_size_ ||
      state.num_layers != num_layers_ || state.valid_tokens <= 0 ||
      state.block_ids_per_layer.size() != static_cast<size_t>(num_layers_))
    return false;
  const size_t expected = static_cast<size_t>(
      (state.valid_tokens + block_size_ - 1) / block_size_);
  size_t attached_layers = 0;
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    if (state.block_ids_per_layer[layer].size() != expected) break;
    size_t attached_pages = 0;
    for (int32_t id : state.block_ids_per_layer[layer]) {
      if (!layer_allocators_[layer]->attach_external(id)) break;
      ++attached_pages;
    }
    if (attached_pages != expected) {
      while (attached_pages)
        layer_allocators_[layer]->detach_external(
            state.block_ids_per_layer[layer][--attached_pages]);
      break;
    }
    ++attached_layers;
  }
  if (attached_layers == static_cast<size_t>(num_layers_)) return true;
  while (attached_layers) {
    const size_t layer = --attached_layers;
    for (int32_t id : state.block_ids_per_layer[layer])
      layer_allocators_[layer]->detach_external(id);
  }
  return false;
}

bool KVCacheManager::detach_external_shared_pages(
    const ExternalKVRequestState& state) {
  if (state.block_ids_per_layer.size() != static_cast<size_t>(num_layers_))
    return false;
  bool ok = true;
  for (int32_t layer = 0; layer < num_layers_; ++layer)
    for (int32_t id : state.block_ids_per_layer[layer])
      ok = layer_allocators_[layer]->detach_external(id) && ok;
  return ok;
}

uint64_t KVCacheManager::total_block_copy_bytes() const {
  uint64_t total=0;for(const auto& allocator:layer_allocators_)
    total+=allocator->payload_copy_bytes();return total;
}

bool KVCacheManager::fork_request(RequestId source_id,
                                  const BranchTokenBoundaries& boundaries,
                                  bool inherits_first_token, int32_t first_token,
                                  RequestId* branch_id, BranchSnapshot* snapshot) {
  if (!branch_id || !is_valid_request(source_id)) return false;
  const auto& source = lookup_request_slot(source_id);
  if (boundaries.computed_tokens < 0 || boundaries.sampled_tokens < 0 ||
      boundaries.pending_tokens < 0 ||
      boundaries.computed_tokens != source.manager.num_tokens() ||
      boundaries.sampled_tokens > boundaries.computed_tokens ||
      (inherits_first_token && first_token < 0) ||
      (!inherits_first_token && first_token != -1)) return false;

  BranchSnapshot captured;
  captured.source_request_id = source_id;
  captured.valid_tokens = source.manager.num_tokens();
  captured.committed_tokens = source.kv_committed_tokens;
  captured.boundaries = boundaries;
  captured.first_token = first_token;
  captured.inherits_first_token = inherits_first_token;
  captured.block_ids_per_layer.resize(num_layers_);
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    captured.block_ids_per_layer[layer] = source.manager.page_table(layer).block_ids();
  }

  const RequestId id = register_request();
  auto& branch = lookup_request_slot(id);
  branch.manager.adopt_branch_snapshot(layer_allocators_, captured.block_ids_per_layer,
                                       captured.valid_tokens);
  branch.kv_committed_tokens = captured.committed_tokens;
  branch.branch_boundaries = captured.boundaries;
  // The full stale-safe request handle is the branch generation identity;
  // two fresh slots may both have allocator generation zero.
  branch.branch_generation = static_cast<uint64_t>(id) + 1;
  branch.branch_first_token = first_token;
  branch.inherits_first_token = inherits_first_token;
  *branch_id = id;
  if (snapshot) *snapshot = std::move(captured);
  return true;
}

int32_t KVCacheManager::kv_committed_tokens(RequestId id) const {
  return lookup_request_slot(id).kv_committed_tokens;
}
int32_t KVCacheManager::published_full_tokens(RequestId id) const {
  return lookup_request_slot(id).published_full_tokens;
}
base::Status KVCacheManager::acquire_compute_leases(RequestId id, std::vector<BlockLease>* leases) {
  if (!is_valid_request(id) || !leases || !leases->empty()) {
    return base::error::InvalidArgument("invalid request compute lease acquisition");
  }
  std::vector<BlockLease> pending;
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    auto& pool = *layer_allocators_[layer];
    for (auto block : get_block_ids(id, layer)) {
      BlockLease pin;
      if (!pool.acquire_lease(pool.handle(block), BlockLeaseKind::kCompute, &pin)) {
        return base::error::InvalidArgument("stale compute block");
      }
      pending.push_back(std::move(pin));
    }
  }
  *leases = std::move(pending);
  return base::error::Success();
}

void KVCacheManager::publish_radix_cache(
    RequestId id, const std::vector<int32_t>& prompt_tokens) {
  auto& slot = lookup_request_slot(id);
  if (!radix_cache_enabled_) {
    return;
  }

  radix_cache_stats_.publish_requests++;
  const int32_t full_blocks = full_blocks_for_tokens(
      std::min(static_cast<int32_t>(prompt_tokens.size()), slot.kv_committed_tokens));
  if (full_blocks == 0) {
    return;
  }

  const std::vector<std::vector<int32_t>> block_ids_per_layer =
      full_block_ids_for_request(slot, full_blocks);
  const std::vector<int32_t> full_tokens(prompt_tokens.begin(),
                                        prompt_tokens.begin() + full_blocks * block_size_);
  const auto match = radix_cache_.match_prefix(full_tokens);
  std::vector<std::vector<uint64_t>> logical_ids = match.block_ids_per_layer;
  std::vector<cache::LogicalPageId> created;
  // Pool-local identity includes the entire prefix, encoded explicitly LE32.
  // A directory belongs to one model's allocator set; it is never shared across models.
  std::string content = "kv-prefix-v1:";
  for (int32_t page = 0; page < full_blocks; ++page) {
    for (int32_t j = 0; j < block_size_; ++j) {
      const uint32_t token = static_cast<uint32_t>(full_tokens[page * block_size_ + j]);
      for (int shift = 0; shift < 32; shift += 8) content.push_back(static_cast<char>(token >> shift));
    }
    if (page < match.matched_blocks) continue;
    std::vector<BlockHandle> handles;
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      handles.push_back(layer_allocators_[layer]->handle(block_ids_per_layer[layer][page]));
    }
    cache::LogicalPageId logical_id = 0;
    if (!page_directory_->publish(content, page_schema_, handles, &logical_id)) {
      for (auto id : created) page_directory_->erase(id);
      return;
    }
    created.push_back(logical_id);
    logical_ids[0].push_back(logical_id);
  }
  const auto result = radix_cache_.insert(full_tokens, logical_ids);
  radix_cache_stats_.published_blocks += result.inserted_blocks;
  slot.published_full_tokens = std::max(slot.published_full_tokens, full_blocks * block_size_);
}

void KVCacheManager::clear_radix_cache() {
  if (migration_engine_) {
    for (auto& slot : request_slots_) {
      if (slot.restore_waiter.waiter) {
        transfer_scheduler_->cancel(slot.restore_waiter.waiter);
        orphaned_restores_.push_back(slot.restore_waiter.waiter);
        slot.restore_waiter = {};
        slot.restore_state = RequestRestoreState::kFailed;
      }
    }
    for (const auto& item : demotions_) transfer_scheduler_->cancel(item.second.waiter);
    CHECK(transfer_scheduler_->drain()) << "cache transfer backend did not reach quiescence";
    service_cache_transfers();
  }
  for (auto& slot : request_slots_) {
    if (slot.active) {
      maybe_unpin_radix_cache_path(&slot);
    }
  }
  radix_cache_.clear_and_collect_block_ids();
  for (auto id : page_directory_->page_ids()) {
    CHECK(retire_recovery_object(id))
        << "live recovery dependency while clearing page " << id;
    CHECK(page_directory_->erase(id));
  }
}

bool KVCacheManager::plan_allocation(RequestId id, int32_t tokens,
                                     KVAllocationIntent* intent) const {
  if (!intent || tokens < 0 || !is_valid_request(id)) return false;
  const auto current = get_context_len(id);
  if (tokens > INT32_MAX - current - (block_size_ - 1)) return false;
  *intent = {id, current, tokens};
  return true;
}
bool KVCacheManager::admit_allocation(const KVAllocationIntent& intent) {
  KVAllocationIntent checked;
  if (!plan_allocation(intent.request_id, intent.append_tokens, &checked) ||
      checked.expected_allocated_tokens != intent.expected_allocated_tokens) return false;
  return append_slots(intent.request_id, intent.append_tokens);
}
bool KVCacheManager::append_slot(RequestId id) {
  KVAllocationIntent intent;
  return plan_allocation(id, 1, &intent) && admit_allocation(intent);
}

bool KVCacheManager::append_slots(RequestId id, int32_t num_tokens) {
  SequenceKVManager& request = lookup_request(id);
  if (num_tokens <= 0 || num_tokens > INT32_MAX - request.num_tokens()) return false;
  const int32_t current_tokens = request.num_tokens();
  const int32_t blocks_before = (current_tokens + block_size_ - 1) / block_size_;
  const int32_t blocks_after = (current_tokens + num_tokens + block_size_ - 1) / block_size_;
  const int32_t required_blocks_per_layer = blocks_after - blocks_before +
      (request.requires_tail_cow() ? 1 : 0);
  if (required_blocks_per_layer <= 0) {
    return request.append_tokens(layer_allocators_, num_tokens);
  }
  if (!ensure_free_blocks_available(required_blocks_per_layer)) {
    return false;
  }
  return request.append_tokens(layer_allocators_, num_tokens);
}

std::pair<int32_t, int32_t> KVCacheManager::get_slot(
    RequestId id, int32_t layer_idx, int32_t token_pos) const {
  return lookup_request(id).get_slot(layer_idx, token_pos);
}

const std::vector<int32_t>& KVCacheManager::get_block_ids(RequestId id,
                                                           int32_t layer_idx) const {
  return lookup_request(id).page_table(layer_idx).block_ids();
}

int32_t KVCacheManager::get_context_len(RequestId id) const {
  return lookup_request(id).num_tokens();
}

int32_t KVCacheManager::num_tokens_in_last_block(RequestId id) const {
  return lookup_request(id).num_tokens_in_last_block();
}

std::pair<int32_t, int32_t> KVCacheManager::current_slot(RequestId id,
                                                          int32_t layer_idx) const {
  return lookup_request(id).current_slot(layer_idx);
}

std::vector<int32_t> KVCacheManager::build_slot_mapping(
    const std::vector<RequestId>& request_ids, int32_t layer_idx) const {
  std::vector<int32_t> mapping;
  mapping.reserve(request_ids.size());
  for (auto rid : request_ids) {
    auto [block_id, offset] = current_slot(rid, layer_idx);
    mapping.push_back(block_id * block_size_ + offset);
  }
  return mapping;
}

std::pair<std::vector<int32_t>, int32_t> KVCacheManager::build_block_tables(
    const std::vector<RequestId>& request_ids, int32_t layer_idx) const {
  int32_t max_blocks = 0;
  for (auto rid : request_ids) {
    auto& ids = get_block_ids(rid, layer_idx);
    max_blocks = std::max(max_blocks, static_cast<int32_t>(ids.size()));
  }

  int32_t batch_size = static_cast<int32_t>(request_ids.size());
  std::vector<int32_t> table(batch_size * max_blocks, -1);  // pad with -1
  for (int32_t b = 0; b < batch_size; ++b) {
    auto& ids = get_block_ids(request_ids[b], layer_idx);
    for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
      table[b * max_blocks + i] = ids[i];
    }
  }
  return {table, max_blocks};
}

std::vector<int32_t> KVCacheManager::build_seq_lens(
    const std::vector<RequestId>& request_ids) const {
  std::vector<int32_t> lens;
  lens.reserve(request_ids.size());
  for (auto rid : request_ids) {
    lens.push_back(get_context_len(rid));
  }
  return lens;
}

const BlockAllocator& KVCacheManager::allocator(int32_t layer_idx) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);
  return *layer_allocators_[layer_idx];
}

BlockAllocator& KVCacheManager::allocator_mut(int32_t layer_idx) {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, num_layers_);
  return *layer_allocators_[layer_idx];
}

int32_t KVCacheManager::num_free_blocks(int32_t layer_idx) const {
  return layer_allocators_[layer_idx]->num_free_blocks();
}

int32_t KVCacheManager::num_total_blocks(int32_t layer_idx) const {
  return layer_allocators_[layer_idx]->num_total_blocks();
}

const KVAppendStats& KVCacheManager::append_stats(RequestId id) const {
  return lookup_request(id).stats();
}

BranchTokenBoundaries KVCacheManager::branch_boundaries(RequestId id) const {
  return lookup_request_slot(id).branch_boundaries;
}

uint64_t KVCacheManager::branch_generation(RequestId id) const {
  return lookup_request_slot(id).branch_generation;
}

bool KVCacheManager::is_valid_request(RequestId id) const {
  uint32_t slot_idx = 0;
  uint64_t generation = 0;
  if (!try_decode_request_id(id, &slot_idx, &generation)) {
    return false;
  }
  const auto& slot = request_slots_[slot_idx];
  return slot.active && slot.generation == generation;
}

bool KVCacheManager::try_decode_request_id(RequestId id, uint32_t* slot_idx,
                                           uint64_t* generation) const {
  if (id < 0) {
    return false;
  }

  const uint32_t decoded_slot_idx = decode_slot_index(id);
  if (decoded_slot_idx >= request_slots_.size()) {
    return false;
  }

  if (slot_idx != nullptr) {
    *slot_idx = decoded_slot_idx;
  }
  if (generation != nullptr) {
    *generation = decode_generation(id);
  }
  return true;
}

int32_t KVCacheManager::full_blocks_for_tokens(int32_t num_tokens) const {
  CHECK_GE(num_tokens, 0);
  return num_tokens / block_size_;
}

int32_t KVCacheManager::additional_blocks_needed(int32_t current_tokens,
                                                 int32_t appended_tokens) const {
  CHECK_GE(current_tokens, 0);
  CHECK_GE(appended_tokens, 0);
  const int32_t blocks_before =
      (current_tokens + block_size_ - 1) / block_size_;
  const int32_t blocks_after =
      (current_tokens + appended_tokens + block_size_ - 1) / block_size_;
  return blocks_after - blocks_before;
}

int32_t KVCacheManager::min_free_blocks_across_layers() const {
  int32_t min_free_blocks = num_free_blocks(0);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    min_free_blocks = std::min(min_free_blocks, num_free_blocks(layer_idx));
  }
  return min_free_blocks;
}

std::vector<std::vector<int32_t>> KVCacheManager::full_block_ids_for_request(
    const RequestSlot& slot, int32_t full_blocks) const {
  CHECK_GE(full_blocks, 0);
  std::vector<std::vector<int32_t>> block_ids_per_layer(num_layers_);
  for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& block_ids = slot.manager.page_table(layer_idx).block_ids();
    CHECK_GE(static_cast<int32_t>(block_ids.size()), full_blocks);
    block_ids_per_layer[layer_idx].assign(block_ids.begin(),
                                          block_ids.begin() + full_blocks);
  }
  return block_ids_per_layer;
}

bool KVCacheManager::ensure_free_blocks_available(int32_t required_blocks_per_layer) {
  CHECK_GE(required_blocks_per_layer, 0);
  if (required_blocks_per_layer == 0) {
    return true;
  }

  auto has_capacity = [&]() {
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      const int32_t available = migration_engine_
          ? schedulable_free_blocks(layer_idx)
          : num_free_blocks(layer_idx);
      if (available < required_blocks_per_layer) {
        return false;
      }
    }
    return true;
  };

  while (!has_capacity()) {
    if (migration_engine_) {
      service_cache_transfers();
      if (has_capacity()) return true;
      if (demote_radix_cache_to_host() == 0) return false;
      service_cache_transfers();
      return has_capacity();
    }
    if (evict_radix_cache_blocks(required_blocks_per_layer) == 0) {
      return false;
    }
  }
  return true;
}

void KVCacheManager::maybe_attach_radix_cache(
    RequestSlot* slot, const std::vector<int32_t>& prompt_tokens) {
  CHECK_NE(slot, nullptr);
  if (!radix_cache_enabled_) {
    return;
  }
  radix_cache_stats_.lookup_requests++;

  const int32_t max_reusable_blocks =
      std::max(0, static_cast<int32_t>(prompt_tokens.size()) - 1) / block_size_;
  if (max_reusable_blocks <= 0) {
    radix_cache_stats_.cache_misses++;
    return;
  }
  const int32_t max_reusable_tokens = max_reusable_blocks * block_size_;
  std::vector<int32_t> reusable_tokens(prompt_tokens.begin(),
                                       prompt_tokens.begin() + max_reusable_tokens);
  auto match = radix_cache_.probe_prefix(reusable_tokens);
  if (match.matched_blocks <= 0) {
    radix_cache_stats_.cache_misses++;
    return;
  }

  std::vector<cache::LogicalPageId> usable_pages;
  bool needs_restore = false;
  for (auto logical_id : match.block_ids_per_layer[0]) {
    if (page_directory_->has_gpu(logical_id)) {
      usable_pages.push_back(logical_id);
    } else if (migration_engine_ && page_directory_->has_host(logical_id)) {
      usable_pages.push_back(logical_id);
      needs_restore = true;
    } else {
      break;
    }
  }
  if (usable_pages.empty()) {
    radix_cache_stats_.cache_misses++;
    return;
  }
  reusable_tokens.resize(usable_pages.size() * block_size_);
  if (needs_restore) {
    if (!reserve_restore_targets(usable_pages)) {
      slot->restore_state = RequestRestoreState::kFailed;
      radix_cache_stats_.host_restore_failures++;
      radix_cache_stats_.cache_misses++;
      return;
    }
    slot->restore_state = RequestRestoreState::kPending;
    slot->restore_pages = std::move(usable_pages);
    slot->restore_prefix_tokens = std::move(reusable_tokens);
    slot->restore_tokens = static_cast<int32_t>(slot->restore_prefix_tokens.size());
    slot->restore_next_page = 0;
    radix_cache_stats_.host_restore_requests++;
    if (!start_next_restore(slot)) {
      slot->restore_state = RequestRestoreState::kFailed;
      radix_cache_stats_.host_restore_failures++;
    }
    return;
  }
  slot->restore_pages = std::move(usable_pages);
  slot->restore_prefix_tokens = std::move(reusable_tokens);
  slot->restore_tokens = static_cast<int32_t>(slot->restore_prefix_tokens.size());
  if (!finish_restore(slot)) {
    radix_cache_stats_.cache_misses++;
  }
}

bool KVCacheManager::start_next_restore(RequestSlot* slot) {
  CHECK_NE(slot, nullptr);
  CHECK(migration_engine_ != nullptr);
  while (slot->restore_next_page < slot->restore_pages.size() &&
         page_directory_->has_gpu(slot->restore_pages[slot->restore_next_page])) {
    ++slot->restore_next_page;
  }
  if (slot->restore_next_page == slot->restore_pages.size()) return finish_restore(slot);
  cache::WaiterTicket ticket;
  if (!transfer_scheduler_->submit(
          {slot->restore_pages[slot->restore_next_page], cache::TransferTarget::kGpu, -1},
          cache::TransferPriority::kPrefill, {}, &ticket)) return false;
  if (ensure_recovery_object(slot->restore_pages[slot->restore_next_page]))
    recovery_graph_.set_in_flight(
        recovery_object_id(slot->restore_pages[slot->restore_next_page]), true);
  slot->restore_waiter = ticket;
  return true;
}

bool KVCacheManager::finish_restore(RequestSlot* slot) {
  CHECK_NE(slot, nullptr);
  std::vector<std::vector<int32_t>> physical(num_layers_);
  std::vector<std::vector<BlockLease>> pins;
  for (auto logical_id : slot->restore_pages) {
    std::vector<BlockLease> page_pins;
    if (!page_directory_->acquire(logical_id, BlockLeaseKind::kCompute, &page_pins)) return false;
    for (int32_t layer = 0; layer < num_layers_; ++layer)
      physical[layer].push_back(page_pins[layer].handle().block_id);
    pins.push_back(std::move(page_pins));
  }
  auto match = radix_cache_.match_prefix(slot->restore_prefix_tokens);
  if (match.matched_tokens != slot->restore_tokens) return false;
  slot->manager.adopt_shared_prefix(layer_allocators_, physical, match.matched_tokens);
  slot->kv_committed_tokens = match.matched_tokens;
  slot->published_full_tokens = match.matched_tokens;
  radix_cache_.pin_path(match.matched_leaf);
  slot->radix_cache_leaf = match.matched_leaf;
  slot->radix_cache_shared_tokens = match.matched_tokens;
  slot->recovery_pages = slot->restore_pages;
  for (auto id : slot->recovery_pages) {
    CHECK(ensure_recovery_object(id));
    CHECK(recovery_graph_.acquire(recovery_object_id(id), true, false));
  }
  slot->restore_state = RequestRestoreState::kReady;
  slot->restore_pages.clear();
  slot->restore_prefix_tokens.clear();
  slot->restore_next_page = 0;
  radix_cache_stats_.cache_hits++;
  radix_cache_stats_.tokens_reused += match.matched_tokens;
  reconcile_restore_target_reservations();
  return true;
}

bool KVCacheManager::reserve_restore_targets(
    const std::vector<cache::LogicalPageId>& pages) {
  std::set<cache::LogicalPageId> additions;
  for (auto id : pages) {
    if (!page_directory_->has_gpu(id) && page_directory_->has_host(id) &&
        restore_target_reservations_.count(id) == 0) {
      additions.insert(id);
    }
  }
  for (int32_t layer = 0; layer < num_layers_; ++layer) {
    if (schedulable_free_blocks(layer) < static_cast<int32_t>(additions.size())) {
      return false;
    }
  }
  restore_target_reservations_.insert(additions.begin(), additions.end());
  return true;
}

void KVCacheManager::reconcile_restore_target_reservations() {
  if (!migration_engine_) return;
  for (auto it = restore_target_reservations_.begin();
       it != restore_target_reservations_.end();) {
    const auto id = *it;
    if (page_directory_->has_gpu(id)) {
      it = restore_target_reservations_.erase(it);
      continue;
    }
    bool referenced = false;
    for (const auto& slot : request_slots_) {
      if (!slot.active || slot.restore_state != RequestRestoreState::kPending) continue;
      if (std::find(slot.restore_pages.begin() +
                        std::min(slot.restore_next_page, slot.restore_pages.size()),
                    slot.restore_pages.end(), id) != slot.restore_pages.end()) {
        referenced = true;
        break;
      }
    }
    // A cancelled waiter's physical flight can still install this page. Keep
    // its target protected until the scheduler reaches quiescence.
    if (!referenced && !cache_transfers_pending()) {
      it = restore_target_reservations_.erase(it);
    } else {
      ++it;
    }
  }
}

void KVCacheManager::maybe_unpin_radix_cache_path(RequestSlot* slot) {
  CHECK_NE(slot, nullptr);
  if (slot->radix_cache_leaf != nullptr) {
    radix_cache_.unpin_path(slot->radix_cache_leaf);
    slot->radix_cache_leaf = nullptr;
    slot->radix_cache_shared_tokens = 0;
  }
  for (auto id : slot->recovery_pages)
    CHECK(recovery_graph_.release(recovery_object_id(id), true, false));
  slot->recovery_pages.clear();
}

std::string KVCacheManager::recovery_object_id(cache::LogicalPageId id) {
  return "kv-page:" + std::to_string(id);
}

bool KVCacheManager::ensure_recovery_object(cache::LogicalPageId id) {
  const auto object_id = recovery_object_id(id);
  if (recovery_graph_.get(object_id)) return true;
  cache::RecoveryObject object;
  object.object_id = object_id;
  object.version = page_schema_.fingerprint() + ":" + std::to_string(id);
  object.kind = cache::RecoveryKind::kPreserveUntilReleased;
  object.continuation = cache::ContinuationRequirement::kExact;
  object.dependencies = {{"kv-schema", page_schema_.fingerprint()}};
  object.gpu_serviceable = page_directory_->has_gpu(id);
  object.host_serviceable = page_directory_->has_host(id);
  object.bytes = recovery_page_bytes_;
  std::string error;
  return recovery_graph_.publish(std::move(object), &error);
}

bool KVCacheManager::retire_recovery_object(cache::LogicalPageId id) {
  const auto object_id = recovery_object_id(id);
  if (!recovery_graph_.get(object_id)) return true;
  if (!recovery_graph_.set_residency(
          object_id, cache::RecoveryTier::kGpu, false) ||
      !recovery_graph_.set_residency(
          object_id, cache::RecoveryTier::kHost, false)) return false;
  return recovery_graph_.erase(object_id);
}

int32_t KVCacheManager::evict_radix_cache_blocks(int32_t min_blocks_to_evict) {
  CHECK_GE(min_blocks_to_evict, 0);
  int32_t evicted_blocks = 0;
  while (evicted_blocks < min_blocks_to_evict) {
    const auto leaves = radix_cache_.collect_evictable_leaves();
    if (leaves.empty()) {
      break;
    }
    const auto removed_block_ids = radix_cache_.erase_leaf(leaves.front());
    const int32_t removed_blocks =
        removed_block_ids.empty() ? 0 : static_cast<int32_t>(removed_block_ids[0].size());
    for (auto logical_id : removed_block_ids[0]) {
      CHECK(retire_recovery_object(logical_id))
          << "live recovery dependency while evicting page " << logical_id;
      CHECK(page_directory_->erase(logical_id));
    }
    evicted_blocks += removed_blocks;
    radix_cache_stats_.evictions++;
    radix_cache_stats_.evicted_blocks += removed_blocks;
  }
  return evicted_blocks;
}

int32_t KVCacheManager::radix_cache_evictable_blocks() const {
  return radix_cache_.evictable_blocks();
}

int32_t KVCacheManager::schedulable_free_blocks(int32_t layer_idx) const {
  const int32_t free = num_free_blocks(layer_idx);
  if (!migration_engine_) return free + radix_cache_evictable_blocks();
  return std::max(0, free - static_cast<int32_t>(restore_target_reservations_.size()));
}

RequestRestoreState KVCacheManager::request_restore_state(RequestId id) const {
  if (!is_valid_request(id)) return RequestRestoreState::kFailed;
  return lookup_request_slot(id).restore_state;
}

int32_t KVCacheManager::service_cache_transfers() {
  if (!migration_engine_) return 0;
  transfer_scheduler_->poll();
  int32_t completed = 0;
  for (auto it = demotions_.begin(); it != demotions_.end();) {
    cache::WaiterState result;
    if (!transfer_scheduler_->state(it->second.waiter, &result) ||
        result == cache::WaiterState::kPending) {
      ++it;
      continue;
    }
    if (result == cache::WaiterState::kSucceeded &&
        ensure_recovery_object(it->first)) {
      const auto object_id = recovery_object_id(it->first);
      recovery_graph_.set_residency(object_id, cache::RecoveryTier::kHost, true);
      recovery_graph_.set_in_flight(object_id, false);
      std::string reason;
      if (recovery_graph_.can_remove_replica(
              object_id, cache::RecoveryTier::kGpu, &reason) &&
          page_directory_->demote_gpu(it->first)) {
        recovery_graph_.set_residency(object_id, cache::RecoveryTier::kGpu, false);
        ++radix_cache_stats_.host_demoted_blocks;
        ++radix_cache_stats_.recovery_graph_committed_demotions;
      } else {
        ++radix_cache_stats_.recovery_graph_keep_decisions;
      }
    } else if (ensure_recovery_object(it->first)) {
      recovery_graph_.set_in_flight(recovery_object_id(it->first), false);
    }
    transfer_scheduler_->consume(it->second.waiter);
    it = demotions_.erase(it);
    ++completed;
  }
  for (auto it = orphaned_restores_.begin(); it != orphaned_restores_.end();) {
    cache::WaiterState result;
    if (transfer_scheduler_->state(*it, &result) &&
        result != cache::WaiterState::kPending) {
      transfer_scheduler_->consume(*it);
      it = orphaned_restores_.erase(it);
      ++completed;
    } else {
      ++it;
    }
  }
  // CPU completions can immediately submit the next page, so iterate until
  // every request either waits on a physical flight or reaches a terminal state.
  bool advanced = true;
  while (advanced) {
    advanced = false;
    for (auto& slot : request_slots_) {
      if (!slot.active || slot.restore_state != RequestRestoreState::kPending ||
          !slot.restore_waiter.waiter) continue;
      cache::WaiterState result;
      if (!transfer_scheduler_->state(slot.restore_waiter.waiter, &result) ||
          result == cache::WaiterState::kPending) continue;
      transfer_scheduler_->consume(slot.restore_waiter.waiter);
      slot.restore_waiter = {};
      ++completed;
      if (result == cache::WaiterState::kSucceeded) {
        if (ensure_recovery_object(slot.restore_pages[slot.restore_next_page])) {
          const auto object_id = recovery_object_id(
              slot.restore_pages[slot.restore_next_page]);
          recovery_graph_.set_residency(object_id, cache::RecoveryTier::kGpu, true);
          recovery_graph_.set_in_flight(object_id, false);
          ++radix_cache_stats_.recovery_graph_restores;
        }
        ++slot.restore_next_page;
        ++radix_cache_stats_.host_restored_blocks;
        if (!start_next_restore(&slot)) {
          slot.restore_state = RequestRestoreState::kFailed;
          ++radix_cache_stats_.host_restore_failures;
        }
      } else {
        if (slot.restore_next_page < slot.restore_pages.size() &&
            ensure_recovery_object(slot.restore_pages[slot.restore_next_page]))
          recovery_graph_.set_in_flight(
              recovery_object_id(slot.restore_pages[slot.restore_next_page]), false);
        slot.restore_state = RequestRestoreState::kFailed;
        ++radix_cache_stats_.host_restore_failures;
      }
      if (slot.restore_state == RequestRestoreState::kFailed) {
        slot.restore_pages.clear();
        slot.restore_prefix_tokens.clear();
        slot.restore_next_page = 0;
        slot.restore_tokens = 0;
        ++radix_cache_stats_.cache_misses;
      }
      advanced = true;
    }
    if (advanced) transfer_scheduler_->poll();
  }
  reconcile_restore_target_reservations();
  return completed;
}

bool KVCacheManager::drain_cache_transfers() {
  if (!migration_engine_) return true;
  if (!transfer_scheduler_->drain()) return false;
  service_cache_transfers();
  return true;
}

int32_t KVCacheManager::demote_radix_cache_to_host(int32_t max_pages) {
  if (!migration_engine_) return 0;
  CHECK_GT(max_pages, 0);
  int32_t admitted = 0;
  for (auto id : page_directory_->page_ids()) {
    if (!page_directory_->has_gpu(id) || page_directory_->has_host(id) || demotions_.count(id))
      continue;
    if (!ensure_recovery_object(id)) {
      ++radix_cache_stats_.recovery_graph_publish_rejections;
      continue;
    }
    const auto object_id = recovery_object_id(id);
    const size_t host_free = host_cache_capacity_bytes_ > host_store_->bytes_used()
        ? host_cache_capacity_bytes_ - host_store_->bytes_used() : 0;
    const auto action = recovery_graph_.choose_pressure_action(object_id, host_free);
    if (action != cache::PressureAction::kDemoteToHost) {
      ++radix_cache_stats_.recovery_graph_keep_decisions;
      continue;
    }
    ++radix_cache_stats_.recovery_graph_demote_decisions;
    cache::WaiterTicket waiter;
    if (!transfer_scheduler_->submit(
            {id, cache::TransferTarget::kHost, -1},
            cache::TransferPriority::kBackground, {}, &waiter)) break;
    recovery_graph_.set_in_flight(object_id, true);
    demotions_.emplace(id, waiter);
    ++admitted;
    service_cache_transfers();
    if (admitted >= max_pages) break;
  }
  service_cache_transfers();
  return admitted;
}

void KVCacheManager::set_radix_cache_enabled(bool enabled) {
  radix_cache_enabled_ = enabled;
}

KVCacheManager::RequestSlot& KVCacheManager::lookup_request_slot(
    RequestId id, uint32_t* slot_idx) {
  return const_cast<RequestSlot&>(
      static_cast<const KVCacheManager*>(this)->lookup_request_slot(id, slot_idx));
}

const KVCacheManager::RequestSlot& KVCacheManager::lookup_request_slot(
    RequestId id, uint32_t* slot_idx) const {
  uint32_t decoded_slot_idx = 0;
  uint64_t generation = 0;
  CHECK(try_decode_request_id(id, &decoded_slot_idx, &generation))
      << "Request " << id << " not found";

  const auto& slot = request_slots_[decoded_slot_idx];
  CHECK(slot.active) << "Request " << id << " is inactive";
  CHECK_EQ(slot.generation, generation)
      << "Request " << id << " generation mismatch";

  if (slot_idx != nullptr) {
    *slot_idx = decoded_slot_idx;
  }
  return slot;
}

SequenceKVManager& KVCacheManager::lookup_request(RequestId id) {
  return lookup_request_slot(id).manager;
}

const SequenceKVManager& KVCacheManager::lookup_request(RequestId id) const {
  return lookup_request_slot(id).manager;
}

}  // namespace base
