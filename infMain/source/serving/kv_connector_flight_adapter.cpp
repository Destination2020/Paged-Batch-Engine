#include "serving/pd_handoff.h"

#include <map>
#include <utility>

#include "cache/transfer_scheduler.h"

namespace serving {
namespace {
class KVConnectorFlightExecutor final : public cache::FlightExecutor {
 public:
  KVConnectorFlightExecutor(KVTransferConnector* connector, KVManifestResolver resolver)
      : connector_(connector), resolver_(std::move(resolver)) {}

  cache::FlightExecutionResult submit(cache::FlightId id,
      const cache::FlightKey& key) noexcept override {
    if (!connector_ || !resolver_ || key.target != cache::TransferTarget::kPeerGpu ||
        handles_.count(id)) return cache::FlightExecutionResult::kFailedSafe;
    KVBlockManifest manifest;
    auto status = resolver_(key.page, &manifest);
    if (!status) return cache::FlightExecutionResult::kFailedSafe;
    HandoffId handle;
    status = connector_->submit(manifest, &handle);
    if (!status) return cache::FlightExecutionResult::kFailedSafe;
    handles_.emplace(id, handle);
    return translate(connector_->poll(handle));
  }
  cache::FlightExecutionResult poll(cache::FlightId id) noexcept override {
    const auto it = handles_.find(id);
    return it == handles_.end() ? cache::FlightExecutionResult::kFailedSafe
                                : translate(connector_->poll(it->second));
  }
  bool cancel(cache::FlightId id) noexcept override {
    const auto it = handles_.find(id);
    if (it == handles_.end()) return false;
    connector_->cancel(it->second, "all shared transfer waiters cancelled");
    return true;
  }
  cache::FlightExecutionResult drain(cache::FlightId id) noexcept override {
    const auto it = handles_.find(id);
    return it == handles_.end() ? cache::FlightExecutionResult::kFailedSafe
                                : translate(connector_->drain(it->second));
  }
  void release(cache::FlightId id) noexcept override {
    const auto it = handles_.find(id);
    if (it == handles_.end()) return;
    connector_->release(it->second);
    handles_.erase(it);
  }

 private:
  static cache::FlightExecutionResult translate(const KVTransferStatus& status) {
    switch (status.state) {
      case KVTransferState::kPending: return cache::FlightExecutionResult::kPending;
      case KVTransferState::kCompleted: return cache::FlightExecutionResult::kSucceeded;
      case KVTransferState::kFailed:
      case KVTransferState::kCancelled: return cache::FlightExecutionResult::kFailedSafe;
    }
    return cache::FlightExecutionResult::kUnknown;
  }
  KVTransferConnector* connector_;
  KVManifestResolver resolver_;
  std::map<cache::FlightId, HandoffId> handles_;
};
}  // namespace

std::unique_ptr<cache::FlightExecutor> MakeKVConnectorFlightExecutor(
    KVTransferConnector* connector, KVManifestResolver resolver) {
  return std::make_unique<KVConnectorFlightExecutor>(connector, std::move(resolver));
}
}  // namespace serving
