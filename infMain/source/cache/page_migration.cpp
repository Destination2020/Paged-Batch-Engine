#include "cache/page_migration.h"
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cache {
namespace {
class CpuTransferBackend final : public TransferBackend {
 public:
  bool supports(base::DeviceType device) const override { return device == base::DeviceType::kDeviceCPU; }
  BackendResult submit(MigrationId, TransferDirection, const BoundTransfer& transfer) noexcept override {
    return ExecuteCpuTransfer(transfer) ? BackendResult::kSucceeded : BackendResult::kFailedSafe;
  }
  BackendResult poll(MigrationId) noexcept override { return BackendResult::kSucceeded; }
  BackendResult drain(MigrationId) noexcept override { return BackendResult::kSucceeded; }
  void release(MigrationId) noexcept override {}
};
void* ComponentAddress(const base::KVBlockPayloadPtrs& payload, ComponentKind kind) {
  switch (kind) {
    case ComponentKind::kKey: return payload.key;
    case ComponentKind::kValue: return payload.value;
    case ComponentKind::kKeyScale: return payload.key_scale;
    case ComponentKind::kValueScale: return payload.value_scale;
  }
  return nullptr;
}
bool Terminal(MigrationState state) {
  return state == MigrationState::kSucceeded || state == MigrationState::kFailed ||
         state == MigrationState::kCancelled;
}
}
std::unique_ptr<TransferBackend> MakeCpuTransferBackend() {
  return std::make_unique<CpuTransferBackend>();
}
PageMigrationEngine::PageMigrationEngine(std::vector<base::BlockAllocator*> pools,
    PageDirectory& directory, HostStore& host, PageSchema schema, size_t max_jobs,
    size_t max_bytes, std::unique_ptr<TransferBackend> backend)
    : pools_(std::move(pools)), directory_(directory), host_(host), schema_(std::move(schema)),
      max_jobs_(max_jobs), max_bytes_(max_bytes), backend_(std::move(backend)) {
  std::vector<size_t> order(schema_.components.size());
  std::iota(order.begin(), order.end(), 0);
  if (!backend_ || !schema_.validate() || pools_.size() != static_cast<size_t>(schema_.layer_count) ||
      !MakePackedLayout(schema_, "migration-packed", 1, order, {}, &packed_layout_) ||
      !planner_.compile(schema_, schema_, TransferPlacement::kExactDirect,
                        &plan_))
    throw std::invalid_argument("invalid page migration configuration");
  for (auto* pool : pools_) {
    if (!pool || !backend_->supports(pool->device_type()) || pool->block_size() != schema_.page_size ||
        pool->num_kv_heads() != schema_.kv_head_count || pool->head_size() != schema_.head_size ||
        pool->storage_spec().storage_mode != schema_.storage.storage_mode ||
        pool->logical_dtype() != schema_.storage.logical_dtype ||
        pool->storage_dtype() != schema_.storage.storage_dtype ||
        pool->scale_dtype() != schema_.storage.scale_dtype)
      throw std::invalid_argument("incompatible migration pool");
  }
}
PageMigrationEngine::~PageMigrationEngine() {
  for (auto& item : jobs_) if (!Terminal(item.second.result.state)) item.second.cancelled = true;
  if (!drain()) std::terminate();
}
base::Status PageMigrationEngine::submit(LogicalPageId page, TransferDirection direction,
                                         MigrationId* id) {
  if (!id || jobs_.size() >= max_jobs_ || packed_layout_.total_bytes > max_bytes_ - used_bytes_ ||
      next_ == std::numeric_limits<MigrationId>::max())
    return base::error::InvalidArgument("migration admission exhausted");
  // No source addresses or source pins are acquired before admission.
  const auto ticket = next_++;
  auto inserted = jobs_.try_emplace(ticket);
  auto& job = inserted.first->second;
  job.result = {ticket, page, direction, MigrationState::kPending};
  used_bytes_ += packed_layout_.total_bytes;
  base::Status status;
  try { status = prepare(job); }
  catch (...) {
    rollback(job);
    jobs_.erase(ticket);
    used_bytes_ -= packed_layout_.total_bytes;
    throw;
  }
  if (!status) {
    rollback(job);
    jobs_.erase(ticket);
    used_bytes_ -= packed_layout_.total_bytes;
    return status;
  }
  // From here rollback requires a backend completion proof.
  const auto result = backend_->submit(ticket, direction, job.transfer);
  finish(job, result);
  *id = ticket;
  return base::error::Success();
}
base::Status PageMigrationEngine::prepare(Job& job) {
  const bool to_host = job.result.direction == TransferDirection::kToHost;
  if (to_host) {
    auto layout = packed_layout_;
    layout.epoch = job.result.id;
    auto status = host_.reserve(schema_, layout, &job.host_ticket);
    if (!status) return status;
    status = directory_.acquire(job.result.page, base::BlockLeaseKind::kIO, &job.gpu_pins);
    if (!status) return status;
  } else {
    if (!directory_.acquire_host(job.result.page, &job.host_source) ||
        job.host_source.store() != &host_)
      return base::error::InvalidArgument("host restore source unavailable");
    job.host_ticket = job.host_source.ticket();
    job.gpu_pins.resize(pools_.size());
    for (size_t i = 0; i < pools_.size(); ++i) {
      auto* pool = pools_[i];
      const auto block = pool->allocate();
      if (block < 0) return base::error::InvalidArgument("GPU restore capacity exhausted");
      const bool pinned = pool->acquire_lease(pool->handle(block), base::BlockLeaseKind::kIO,
                                             &job.gpu_pins[i]);
      pool->free(block); // The journal's I/O lease now owns the fresh allocation.
      if (!pinned) return base::error::InvalidArgument("GPU restore lease failed");
    }
  }
  if (job.gpu_pins.size() != pools_.size())
    return base::error::InvalidArgument("migration layer count mismatch");
  const auto* host_layout = host_.layout(job.host_ticket);
  if (!host_layout || !host_.matches_schema(job.host_ticket, schema_))
    return base::error::InvalidArgument("incompatible migration host schema");
  job.gpu_layout = packed_layout_;
  job.gpu_layout.epoch = job.result.id;
  ConstComponentEndpoint source;
  MutableComponentEndpoint target;
  uint8_t* host_target = nullptr;
  if (to_host) {
    host_target = host_.begin_copy(job.host_ticket);
    if (!host_target) return base::error::InvalidArgument("host copy reservation unavailable");
    source.layout = &job.gpu_layout;
    target.layout = host_layout;
  } else {
    source.layout = host_layout;
    target.layout = &job.gpu_layout;
  }
  for (auto& entry : job.gpu_layout.components) {
    const auto layer = static_cast<size_t>(entry.component.layer);
    const auto handle = job.gpu_pins[layer].handle();
    if (!pools_[layer]->is_current(handle))
      return base::error::InvalidArgument("migration pool or generation mismatch");
    entry.virtual_block_id = handle.block_id;
    auto* gpu = static_cast<uint8_t*>(ComponentAddress(
        pools_[layer]->get_block_payload_ptrs(handle.block_id), entry.component.kind));
    if (to_host) source.components.push_back({entry.component, gpu, entry.size_bytes});
    else target.components.push_back({entry.component, gpu, entry.size_bytes});
  }
  for (const auto& entry : host_layout->components) {
    if (to_host) target.components.push_back(
        {entry.component, host_target + entry.offset_bytes, entry.size_bytes});
    else source.components.push_back(
        {entry.component, job.host_source.data() + entry.offset_bytes, entry.size_bytes});
  }
  return BoundTransfer::BindComponents(*plan_, schema_, source, target,
                                       &job.transfer);
}
void PageMigrationEngine::rollback(Job& job) {
  if (job.result.direction == TransferDirection::kToHost && job.host_ticket) {
    HostPageState state;
    if (host_.state(job.host_ticket, &state)) {
      if (state == HostPageState::kReserved) host_.cancel(job.host_ticket);
      else if (state == HostPageState::kCopying) host_.complete(job.host_ticket, false);
      else if (state == HostPageState::kQuarantined) host_.reclaim_after_quiescence(job.host_ticket);
      else host_.retire(job.host_ticket);
    }
  }
  job.gpu_pins.clear();
  job.host_source.reset();
  job.transfer = BoundTransfer{};
}
void PageMigrationEngine::finish(Job& job, BackendResult result) {
  if (Terminal(job.result.state) || result == BackendResult::kPending) return;
  if (result == BackendResult::kUnknown) {
    job.result.state = MigrationState::kQuarantined;
    if (job.result.direction == TransferDirection::kToHost) host_.quarantine(job.host_ticket);
    return;
  }
  // Even if a later drain succeeds, a quarantined copy is discarded, not published.
  const bool valid = result == BackendResult::kSucceeded && !job.cancelled &&
                     job.result.state != MigrationState::kQuarantined;
  bool committed = false;
  if (valid && job.result.direction == TransferDirection::kToHost) {
    if (host_.complete(job.host_ticket, true))
      committed = static_cast<bool>(directory_.attach_host(job.result.page, &host_, job.host_ticket));
  } else if (valid) {
    std::vector<base::BlockHandle> blocks;
    for (const auto& pin : job.gpu_pins) blocks.push_back(pin.handle());
    committed = static_cast<bool>(directory_.install_gpu(
        job.result.page, &host_, job.host_ticket, blocks));
  }
  backend_->release(job.result.id);
  if (!committed) rollback(job);
  else {
    job.gpu_pins.clear();
    job.host_source.reset();
    job.transfer = BoundTransfer{};
  }
  job.result.state = job.cancelled ? MigrationState::kCancelled :
                     committed ? MigrationState::kSucceeded : MigrationState::kFailed;
}
bool PageMigrationEngine::cancel(MigrationId id) {
  auto it = jobs_.find(id);
  if (it == jobs_.end() || Terminal(it->second.result.state)) return false;
  it->second.cancelled = true;
  return true;
}
void PageMigrationEngine::poll() {
  for (auto& item : jobs_) {
    auto& job = item.second;
    if (job.result.state == MigrationState::kPending) finish(job, backend_->poll(item.first));
  }
}
bool PageMigrationEngine::drain() {
  bool safe = true;
  for (auto& item : jobs_) {
    auto& job = item.second;
    if (!Terminal(job.result.state)) {
      finish(job, backend_->drain(item.first));
      safe = Terminal(job.result.state) && safe;
    }
  }
  return safe;
}
bool PageMigrationEngine::completion(MigrationId id, MigrationCompletion* output) const {
  const auto it = jobs_.find(id);
  if (!output || it == jobs_.end()) return false;
  *output = it->second.result;
  return true;
}
bool PageMigrationEngine::consume(MigrationId id) {
  const auto it = jobs_.find(id);
  if (it == jobs_.end() || !Terminal(it->second.result.state)) return false;
  jobs_.erase(it);
  used_bytes_ -= packed_layout_.total_bytes;
  return true;
}
}  // namespace cache
