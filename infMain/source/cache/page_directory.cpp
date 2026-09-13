#include "cache/page_directory.h"
#include <limits>
#include <utility>

namespace cache {
PageDirectory::PageDirectory(std::vector<base::BlockAllocator*> pools, size_t capacity)
    : pools_(std::move(pools)), capacity_(capacity) {}
PageDirectory::~PageDirectory() { clear(); }
std::string PageDirectory::key(const std::string& content, const PageSchema& schema) {
  return std::to_string(content.size()) + ":" + content + schema.fingerprint();
}
LogicalPageId PageDirectory::find(const std::string& content, const PageSchema& schema) const {
  const auto it = index_.find(key(content, schema));
  return it == index_.end() ? 0 : it->second;
}
base::Status PageDirectory::publish(const std::string& content, const PageSchema& schema,
                                  const std::vector<base::BlockHandle>& blocks,
                                  LogicalPageId* id) {
  if (id == nullptr || content.empty() || !schema.validate() ||
      blocks.size() != pools_.size() || schema.layer_count != static_cast<int32_t>(pools_.size())) {
    return base::error::InvalidArgument("incomplete page publication");
  }
  if (!valid_blocks(schema, blocks)) {
    return base::error::InvalidArgument("stale block or incompatible page pool");
  }
  const auto identity = key(content, schema);
  const auto existing = index_.find(identity);
  if (existing != index_.end()) { *id = existing->second; return base::error::Success(); }
  if (pages_.size() >= capacity_ || next_id_ == std::numeric_limits<LogicalPageId>::max()) {
    return base::error::InvalidArgument("page directory capacity exhausted");
  }
  const auto new_id = next_id_++;
  Page page;
  page.key = identity;
  page.blocks = blocks;
  page.schema = schema;
  pages_.emplace(new_id, std::move(page));
  try {
    index_.emplace(identity, new_id);
  } catch (...) {
    pages_.erase(new_id);
    throw;
  }
  for (size_t i = 0; i < pools_.size(); ++i) pools_[i]->incref(blocks[i].block_id);
  *id = new_id;
  return base::error::Success();
}
base::Status PageDirectory::acquire(LogicalPageId id, base::BlockLeaseKind kind,
                                  std::vector<base::BlockLease>* leases) const {
  const auto it = pages_.find(id);
  if (!leases || !leases->empty() || it == pages_.end() ||
      it->second.blocks.size() != pools_.size()) {
    return base::error::InvalidArgument("invalid page lease output or unknown page");
  }
  std::vector<base::BlockLease> pending(pools_.size());
  for (size_t i = 0; i < pools_.size(); ++i) {
    if (!pools_[i]->acquire_lease(it->second.blocks[i], kind, &pending[i])) {
      return base::error::InvalidArgument("page residency is stale");
    }
  }
  *leases = std::move(pending);
  return base::error::Success();
}
bool PageDirectory::valid_blocks(const PageSchema& schema,
                                 const std::vector<base::BlockHandle>& blocks) const {
  if (blocks.size() != pools_.size() || schema.layer_count != static_cast<int32_t>(pools_.size()))
    return false;
  // Validate the entire page before retaining the first physical block.
  for (size_t i = 0; i < pools_.size(); ++i) {
    const auto* pool = pools_[i];
    if (!pool || !pool->is_current(blocks[i]) || pool->block_size() != schema.page_size ||
        pool->num_kv_heads() != schema.kv_head_count || pool->head_size() != schema.head_size ||
        pool->logical_dtype() != schema.storage.logical_dtype ||
        pool->storage_dtype() != schema.storage.storage_dtype ||
        pool->scale_dtype() != schema.storage.scale_dtype ||
        pool->storage_mode() != schema.storage.storage_mode) {
      return false;
    }
  }
  return true;
}
base::Status PageDirectory::attach_host(LogicalPageId id, HostStore* store, HostTicket ticket) {
  const auto it = pages_.find(id);
  if (it == pages_.end() || !store || it->second.host_store ||
      !store->matches_schema(ticket, it->second.schema))
    return base::error::InvalidArgument("unknown page, existing host or incompatible schema");
  for (const auto& entry : pages_) {
    if (entry.second.host_store == store && entry.second.host_ticket == ticket)
      return base::error::InvalidArgument("host ticket already belongs to a page");
  }
  HostReadLease pin;
  if (!store->acquire(ticket, &pin))
    return base::error::InvalidArgument("host page is not ready");
  it->second.host_pin = std::move(pin);
  it->second.host_store = store;
  it->second.host_ticket = ticket;
  return base::error::Success();
}
bool PageDirectory::acquire_host(LogicalPageId id, HostReadLease* lease) const {
  const auto it = pages_.find(id);
  return it != pages_.end() && it->second.host_store &&
         it->second.host_store->acquire(it->second.host_ticket, lease);
}
bool PageDirectory::demote_gpu(LogicalPageId id) {
  const auto it = pages_.find(id);
  if (it == pages_.end() || it->second.blocks.empty()) return false;
  HostReadLease proof;
  if (!acquire_host(id, &proof)) return false;
  for (size_t i = 0; i < pools_.size(); ++i) {
    const auto block = it->second.blocks[i].block_id;
    if (pools_[i]->compute_pins(block) || pools_[i]->io_pins(block)) return false;
  }
  for (size_t i = 0; i < pools_.size(); ++i) pools_[i]->free(it->second.blocks[i].block_id);
  it->second.blocks.clear();
  return true;
}
base::Status PageDirectory::install_gpu(LogicalPageId id, const HostStore* expected_store,
                                       HostTicket expected_host,
                                       const std::vector<base::BlockHandle>& blocks) {
  const auto it = pages_.find(id);
  HostReadLease proof;
  if (it == pages_.end() || !it->second.blocks.empty() ||
      it->second.host_store != expected_store ||
      it->second.host_ticket != expected_host || !acquire_host(id, &proof) ||
      !valid_blocks(it->second.schema, blocks))
    return base::error::InvalidArgument("stale restore source or invalid GPU destination");
  auto retained = blocks;  // Allocation can fail before any reference is changed.
  for (size_t i = 0; i < pools_.size(); ++i) pools_[i]->incref(blocks[i].block_id);
  it->second.blocks = std::move(retained);
  return base::error::Success();
}
void PageDirectory::release_host(Page& page) {
  if (!page.host_store) return;
  page.host_store->retire(page.host_ticket);
  page.host_pin.reset();
  page.host_store = nullptr;
  page.host_ticket = 0;
}
bool PageDirectory::drop_host(LogicalPageId id) {
  const auto it = pages_.find(id);
  if (it == pages_.end() || !it->second.host_store) return false;
  if (it->second.blocks.empty()) return erase(id);
  release_host(it->second);
  return true;
}
bool PageDirectory::has_gpu(LogicalPageId id) const {
  const auto it = pages_.find(id);
  return it != pages_.end() && it->second.blocks.size() == pools_.size();
}
bool PageDirectory::has_host(LogicalPageId id) const {
  const auto it = pages_.find(id);
  if (it == pages_.end() || !it->second.host_store) return false;
  HostReadLease proof;
  return it->second.host_store->acquire(it->second.host_ticket, &proof);
}
std::vector<LogicalPageId> PageDirectory::page_ids() const {
  std::vector<LogicalPageId> result;
  result.reserve(pages_.size());
  for (const auto& entry : pages_) result.push_back(entry.first);
  return result;
}
bool PageDirectory::erase(LogicalPageId id) {
  const auto it = pages_.find(id);
  if (it == pages_.end()) return false;
  for (size_t i = 0; i < it->second.blocks.size(); ++i) pools_[i]->free(it->second.blocks[i].block_id);
  release_host(it->second);
  index_.erase(it->second.key);
  pages_.erase(it);
  return true;
}
void PageDirectory::clear() { while (!pages_.empty()) erase(pages_.begin()->first); }
}  // namespace cache
