#ifndef KUIPER_INCLUDE_CACHE_TRANSFER_SCHEDULER_H_
#define KUIPER_INCLUDE_CACHE_TRANSFER_SCHEDULER_H_

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <map>
#include <memory>
#include <vector>

namespace cache {

using FlightId = uint64_t;
using WaiterId = uint64_t;

enum class TransferTarget { kHost, kGpu, kPeerGpu };
enum class TransferPriority : int32_t { kBackground = 0, kPrefill = 1, kDecode = 2 };
enum class FlightExecutionResult { kPending, kSucceeded, kFailedSafe, kUnknown };
enum class WaiterState { kPending, kSucceeded, kFailed, kCancelled, kTimedOut };
enum class SubmitOutcome {
  kAdmitted,
  kJoined,
  kRetryCapacity,
  kStale,
  kUnsafe,
};

struct TransferSchedulerConfig {
  size_t max_flights = 0;
  size_t max_waiters = 0;
  size_t max_total_active = 0;
  size_t max_d2h_active = 0;
  size_t max_h2d_active = 0;
  size_t max_p2p_active = 0;
  uint64_t aging_ticks = 1;
  size_t reserved_demand_flights = 0;
  size_t reserved_demand_waiters = 0;
  bool direction_lanes_enabled = true;
};

struct FlightKey {
  uint64_t page = 0;
  TransferTarget target = TransferTarget::kHost;
  int32_t destination = -1;
  bool operator<(const FlightKey& other) const;
  bool operator==(const FlightKey& other) const;
};

struct WaiterTicket {
  WaiterId waiter = 0;
  FlightId flight = 0;
};

struct TransferSchedulerStats {
  uint64_t submitted_waiters = 0;
  uint64_t merged_waiters = 0;
  uint64_t physical_submissions = 0;
  uint64_t priority_donations = 0;
  uint64_t cancelled_waiters = 0;
  uint64_t max_flights = 0;
  uint64_t max_waiters = 0;
  uint64_t oldest_queue_ticks = 0;
  uint64_t background_completed = 0;
  uint64_t demand_completed = 0;
  uint64_t timed_out_waiters = 0;
  uint64_t max_d2h_active = 0;
  uint64_t max_h2d_active = 0;
  uint64_t max_p2p_active = 0;
};

class FlightExecutor {
 public:
  virtual ~FlightExecutor() = default;
  virtual FlightExecutionResult submit(FlightId id, const FlightKey& key) noexcept = 0;
  virtual FlightExecutionResult poll(FlightId id) noexcept = 0;
  virtual bool cancel(FlightId id) noexcept = 0;
  virtual FlightExecutionResult drain(FlightId id) noexcept = 0;
  virtual void release(FlightId id) noexcept = 0;
};

// Single owner thread. A physical flight is shared by all waiters for its key;
// waiter cancellation never asserts that the physical operation has stopped.
class TransferScheduler {
 public:
  TransferScheduler(size_t max_flights, size_t max_waiters, size_t max_active,
                    uint64_t aging_ticks, std::unique_ptr<FlightExecutor> executor);
  TransferScheduler(TransferSchedulerConfig config,
                    std::unique_ptr<FlightExecutor> executor);
  ~TransferScheduler();
  TransferScheduler(const TransferScheduler&) = delete;
  TransferScheduler& operator=(const TransferScheduler&) = delete;

  bool submit(const FlightKey& key, TransferPriority priority,
              const std::vector<FlightId>& dependencies, WaiterTicket* ticket);
  SubmitOutcome submit_typed(
      const FlightKey& key, TransferPriority priority,
      const std::vector<FlightId>& dependencies,
      std::chrono::steady_clock::time_point absolute_deadline,
      WaiterTicket* ticket);
  bool cancel(WaiterId waiter);
  void poll();
  bool drain();
  bool state(WaiterId waiter, WaiterState* state) const;
  bool consume(WaiterId waiter);
  size_t flight_count() const { return flights_.size(); }
  size_t waiter_count() const { return waiters_.size(); }
  size_t active_count() const { return active_; }
  const TransferSchedulerStats& stats() const { return stats_; }

 private:
  enum class FlightState { kQueued, kActive, kSucceeded, kFailed, kDraining, kQuarantined };
  struct Flight {
    FlightId id = 0;
    FlightKey key;
    FlightState state = FlightState::kQueued;
    TransferPriority priority = TransferPriority::kBackground;
    TransferPriority initial_priority = TransferPriority::kBackground;
    uint64_t enqueue_tick = 0;
    std::vector<FlightId> dependencies;
    std::vector<WaiterId> waiters;
    size_t live_waiters = 0;
  };
  struct Waiter {
    FlightId flight = 0;
    WaiterState state = WaiterState::kPending;
    TransferPriority original_priority = TransferPriority::kBackground;
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max();
  };

  bool dependencies_ready(const Flight& flight, bool* failed) const;
  void dispatch();
  void finish(Flight& flight, FlightExecutionResult result);
  void mark_waiters(Flight& flight, WaiterState state);
  void cleanup_flight(FlightId id);
  void recompute_priority(Flight* flight);
  void expire_waiters(std::chrono::steady_clock::time_point now);
  size_t lane_active(const FlightKey& key) const;
  size_t lane_limit(const FlightKey& key) const;
  void change_lane_active(const FlightKey& key, int delta);
  bool terminal(FlightState state) const;

  size_t max_flights_, max_waiters_, max_active_;
  size_t max_d2h_active_, max_h2d_active_, max_p2p_active_;
  size_t reserved_demand_flights_ = 0, reserved_demand_waiters_ = 0;
  bool direction_lanes_enabled_ = true;
  size_t d2h_active_ = 0, h2d_active_ = 0, p2p_active_ = 0;
  uint64_t aging_ticks_, tick_ = 0;
  FlightId next_flight_ = 1;
  WaiterId next_waiter_ = 1;
  size_t active_ = 0;
  std::unique_ptr<FlightExecutor> executor_;
  std::map<FlightId, Flight> flights_;
  std::map<FlightKey, FlightId> by_key_;
  std::map<WaiterId, Waiter> waiters_;
  TransferSchedulerStats stats_;
};

class PageMigrationEngine;
std::unique_ptr<FlightExecutor> MakePageMigrationFlightExecutor(PageMigrationEngine& engine);

}  // namespace cache
#endif
