#include "cache/transfer_scheduler.h"

#include <map>

#include "cache/page_migration.h"

namespace cache {
namespace {
class PageMigrationFlightExecutor final : public FlightExecutor {
 public:
  explicit PageMigrationFlightExecutor(PageMigrationEngine& engine) : engine_(engine) {}

  FlightExecutionResult submit(FlightId id, const FlightKey& key) noexcept override {
    if (key.target == TransferTarget::kPeerGpu || migrations_.count(id))
      return FlightExecutionResult::kFailedSafe;
    MigrationId migration = 0;
    const auto direction = key.target == TransferTarget::kHost
                               ? TransferDirection::kToHost : TransferDirection::kToGpu;
    if (!engine_.submit(key.page, direction, &migration))
      return FlightExecutionResult::kFailedSafe;
    migrations_.emplace(id, migration);
    return result(id, false);
  }
  FlightExecutionResult poll(FlightId id) noexcept override {
    engine_.poll();
    return result(id, false);
  }
  bool cancel(FlightId id) noexcept override {
    const auto it = migrations_.find(id);
    return it != migrations_.end() && engine_.cancel(it->second);
  }
  FlightExecutionResult drain(FlightId id) noexcept override {
    if (!engine_.drain()) return FlightExecutionResult::kUnknown;
    return result(id, true);
  }
  void release(FlightId id) noexcept override {
    const auto it = migrations_.find(id);
    if (it == migrations_.end()) return;
    if (!engine_.consume(it->second)) std::terminate();
    migrations_.erase(it);
  }

 private:
  FlightExecutionResult result(FlightId id, bool drained) const noexcept {
    const auto it = migrations_.find(id);
    if (it == migrations_.end()) return FlightExecutionResult::kFailedSafe;
    MigrationCompletion completion;
    if (!engine_.completion(it->second, &completion)) return FlightExecutionResult::kUnknown;
    switch (completion.state) {
      case MigrationState::kPending: return FlightExecutionResult::kPending;
      case MigrationState::kSucceeded: return FlightExecutionResult::kSucceeded;
      case MigrationState::kCancelled:
      case MigrationState::kFailed: return FlightExecutionResult::kFailedSafe;
      case MigrationState::kQuarantined:
        return drained ? FlightExecutionResult::kFailedSafe : FlightExecutionResult::kUnknown;
    }
    return FlightExecutionResult::kUnknown;
  }
  PageMigrationEngine& engine_;
  std::map<FlightId, MigrationId> migrations_;
};
}  // namespace

std::unique_ptr<FlightExecutor> MakePageMigrationFlightExecutor(PageMigrationEngine& engine) {
  return std::make_unique<PageMigrationFlightExecutor>(engine);
}
}  // namespace cache
