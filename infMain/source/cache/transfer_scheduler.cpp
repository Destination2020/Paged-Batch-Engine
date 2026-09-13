#include "cache/transfer_scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cache {

bool FlightKey::operator<(const FlightKey& other) const {
  if (page != other.page) return page < other.page;
  if (target != other.target) return target < other.target;
  return destination < other.destination;
}
bool FlightKey::operator==(const FlightKey& other) const {
  return page == other.page && target == other.target && destination == other.destination;
}

TransferScheduler::TransferScheduler(size_t max_flights, size_t max_waiters,
    size_t max_active, uint64_t aging_ticks, std::unique_ptr<FlightExecutor> executor)
    : TransferScheduler(TransferSchedulerConfig{max_flights, max_waiters, max_active,
          max_active, max_active, max_active, aging_ticks, 0, 0}, std::move(executor)) {}

TransferScheduler::TransferScheduler(TransferSchedulerConfig config,
    std::unique_ptr<FlightExecutor> executor)
    : max_flights_(config.max_flights), max_waiters_(config.max_waiters),
      max_active_(config.max_total_active), max_d2h_active_(config.max_d2h_active),
      max_h2d_active_(config.max_h2d_active), max_p2p_active_(config.max_p2p_active),
      reserved_demand_flights_(config.reserved_demand_flights),
      reserved_demand_waiters_(config.reserved_demand_waiters),
      direction_lanes_enabled_(config.direction_lanes_enabled),
      aging_ticks_(std::max<uint64_t>(1, config.aging_ticks)), executor_(std::move(executor)) {
  if (!max_flights_ || !max_waiters_ || !max_active_ || max_active_ > max_flights_ || !executor_)
    throw std::invalid_argument("invalid transfer scheduler capacity");
  if (!max_d2h_active_ || !max_h2d_active_ || !max_p2p_active_)
    throw std::invalid_argument("invalid transfer scheduler lane capacity");
  if (reserved_demand_flights_ > max_flights_ || reserved_demand_waiters_ > max_waiters_)
    throw std::invalid_argument("invalid transfer scheduler demand reserve");
}

TransferScheduler::~TransferScheduler() {
  for (auto& item : flights_) {
    auto& flight = item.second;
    if (!terminal(flight.state)) {
      mark_waiters(flight, WaiterState::kCancelled);
      if (flight.state == FlightState::kQueued) {
        flight.state = FlightState::kFailed;
      } else if (flight.state == FlightState::kActive || flight.state == FlightState::kDraining ||
          flight.state == FlightState::kQuarantined) executor_->cancel(flight.id);
    }
  }
  if (!drain()) std::terminate();
}

bool TransferScheduler::submit(const FlightKey& key, TransferPriority priority,
    const std::vector<FlightId>& dependencies, WaiterTicket* ticket) {
  const auto outcome = submit_typed(key, priority, dependencies,
      std::chrono::steady_clock::time_point::max(), ticket);
  return outcome == SubmitOutcome::kAdmitted || outcome == SubmitOutcome::kJoined;
}

SubmitOutcome TransferScheduler::submit_typed(
    const FlightKey& key, TransferPriority priority,
    const std::vector<FlightId>& dependencies,
    std::chrono::steady_clock::time_point absolute_deadline,
    WaiterTicket* ticket) {
  const size_t waiter_limit = priority == TransferPriority::kBackground
      ? max_waiters_ - reserved_demand_waiters_ : max_waiters_;
  if (!ticket || !key.page || waiters_.size() >= waiter_limit ||
      next_waiter_ == std::numeric_limits<WaiterId>::max()) return SubmitOutcome::kRetryCapacity;
  if (absolute_deadline <= std::chrono::steady_clock::now()) return SubmitOutcome::kStale;
  for (auto dependency : dependencies)
    if (!flights_.count(dependency)) return SubmitOutcome::kStale;
  Flight* flight = nullptr;
  bool joined = false;
  auto existing = by_key_.find(key);
  if (existing != by_key_.end()) {
    flight = &flights_.at(existing->second);
    joined = true;
    ++stats_.merged_waiters;
    if (priority > flight->priority && !terminal(flight->state)) {
      flight->priority = priority;
      ++stats_.priority_donations;
    }
  } else {
    const size_t flight_limit = priority == TransferPriority::kBackground
        ? max_flights_ - reserved_demand_flights_ : max_flights_;
    if (flights_.size() >= flight_limit ||
        next_flight_ == std::numeric_limits<FlightId>::max()) return SubmitOutcome::kRetryCapacity;
    const auto id = next_flight_++;
    Flight candidate;
    candidate.id = id;
    candidate.key = key;
    candidate.priority = priority;
    candidate.initial_priority = priority;
    candidate.enqueue_tick = tick_;
    candidate.dependencies = dependencies;
    flight = &flights_.emplace(id, std::move(candidate)).first->second;
    by_key_.emplace(key, id);
  }
  const auto waiter = next_waiter_++;
  WaiterState initial = WaiterState::kPending;
  if (flight->state == FlightState::kSucceeded) initial = WaiterState::kSucceeded;
  if (flight->state == FlightState::kFailed) initial = WaiterState::kFailed;
  waiters_.emplace(waiter, Waiter{flight->id, initial, priority, absolute_deadline});
  flight->waiters.push_back(waiter);
  if (initial == WaiterState::kPending) ++flight->live_waiters;
  *ticket = {waiter, flight->id};
  ++stats_.submitted_waiters;
  stats_.max_flights = std::max<uint64_t>(stats_.max_flights, flights_.size());
  stats_.max_waiters = std::max<uint64_t>(stats_.max_waiters, waiters_.size());
  dispatch();
  return joined ? SubmitOutcome::kJoined : SubmitOutcome::kAdmitted;
}

bool TransferScheduler::cancel(WaiterId waiter) {
  auto it = waiters_.find(waiter);
  if (it == waiters_.end() || it->second.state != WaiterState::kPending) return false;
  it->second.state = WaiterState::kCancelled;
  ++stats_.cancelled_waiters;
  auto& flight = flights_.at(it->second.flight);
  if (!flight.live_waiters) std::terminate();
  --flight.live_waiters;
  recompute_priority(&flight);
  if (!flight.live_waiters) {
    if (flight.state == FlightState::kQueued) flight.state = FlightState::kFailed;
    else if (flight.state == FlightState::kActive) {
      executor_->cancel(flight.id);
      flight.state = FlightState::kDraining;
    }
  }
  return true;
}

bool TransferScheduler::dependencies_ready(const Flight& flight, bool* failed) const {
  *failed = false;
  for (auto id : flight.dependencies) {
    const auto it = flights_.find(id);
    if (it == flights_.end() || it->second.state == FlightState::kFailed) {
      *failed = true;
      return false;
    }
    if (it->second.state != FlightState::kSucceeded) return false;
  }
  return true;
}

void TransferScheduler::dispatch() {
  while (active_ < max_active_) {
    Flight* selected = nullptr;
    int64_t selected_score = std::numeric_limits<int64_t>::min();
    for (auto& item : flights_) {
      auto& flight = item.second;
      if (flight.state != FlightState::kQueued) continue;
      if (direction_lanes_enabled_ &&
          lane_active(flight.key) >= lane_limit(flight.key)) continue;
      bool failed = false;
      if (!dependencies_ready(flight, &failed)) {
        if (failed) {
          flight.state = FlightState::kFailed;
          mark_waiters(flight, WaiterState::kFailed);
        }
        continue;
      }
      const auto age = tick_ - flight.enqueue_tick;
      stats_.oldest_queue_ticks = std::max(stats_.oldest_queue_ticks, age);
      // One priority lane costs four aging quanta.  Consequently even a
      // background flight outranks newly arriving decode work after eight
      // quanta, while fresh demand still wins the common short-queue case.
      constexpr int64_t kPriorityLaneQuanta = 4;
      const int64_t score = static_cast<int64_t>(flight.priority) * kPriorityLaneQuanta +
                            static_cast<int64_t>(age / aging_ticks_);
      if (!selected || score > selected_score ||
          (score == selected_score && flight.id < selected->id)) {
        selected = &flight;
        selected_score = score;
      }
    }
    if (!selected) return;
    const auto satisfied_dependencies = std::move(selected->dependencies);
    selected->dependencies.clear();
    selected->state = FlightState::kActive;
    ++active_;
    change_lane_active(selected->key, 1);
    ++stats_.physical_submissions;
    finish(*selected, executor_->submit(selected->id, selected->key));
    for (auto dependency : satisfied_dependencies) cleanup_flight(dependency);
  }
}

void TransferScheduler::finish(Flight& flight, FlightExecutionResult result) {
  if (result == FlightExecutionResult::kPending) return;
  if (result == FlightExecutionResult::kUnknown) {
    flight.state = FlightState::kQuarantined;
    return;
  }
  if (flight.state == FlightState::kActive || flight.state == FlightState::kDraining ||
      flight.state == FlightState::kQuarantined) {
    if (!active_) std::terminate();
    --active_;
    change_lane_active(flight.key, -1);
    executor_->release(flight.id);
  }
  const bool success = result == FlightExecutionResult::kSucceeded && flight.live_waiters;
  flight.state = success ? FlightState::kSucceeded : FlightState::kFailed;
  mark_waiters(flight, success ? WaiterState::kSucceeded : WaiterState::kFailed);
  if (success) {
    if (flight.initial_priority == TransferPriority::kBackground) ++stats_.background_completed;
    else ++stats_.demand_completed;
  }
}

void TransferScheduler::mark_waiters(Flight& flight, WaiterState state) {
  for (auto id : flight.waiters) {
    auto it = waiters_.find(id);
    if (it != waiters_.end() && it->second.state == WaiterState::kPending)
      it->second.state = state;
  }
  flight.live_waiters = 0;
}

void TransferScheduler::poll() {
  ++tick_;
  expire_waiters(std::chrono::steady_clock::now());
  for (auto& item : flights_) {
    auto& flight = item.second;
    if (flight.state == FlightState::kActive || flight.state == FlightState::kDraining)
      finish(flight, executor_->poll(flight.id));
  }
  dispatch();
}

void TransferScheduler::recompute_priority(Flight* flight) {
  if (!flight || terminal(flight->state)) return;
  auto priority = flight->initial_priority;
  bool found = false;
  for (auto id : flight->waiters) {
    const auto it = waiters_.find(id);
    if (it == waiters_.end() || it->second.state != WaiterState::kPending) continue;
    priority = found ? std::max(priority, it->second.original_priority)
                     : it->second.original_priority;
    found = true;
  }
  if (found) flight->priority = priority;
}

void TransferScheduler::expire_waiters(std::chrono::steady_clock::time_point now) {
  std::vector<WaiterId> expired;
  for (const auto& item : waiters_) {
    if (item.second.state == WaiterState::kPending && item.second.deadline <= now)
      expired.push_back(item.first);
  }
  for (auto id : expired) {
    auto& waiter = waiters_.at(id);
    auto& flight = flights_.at(waiter.flight);
    waiter.state = WaiterState::kTimedOut;
    ++stats_.timed_out_waiters;
    if (!flight.live_waiters) std::terminate();
    --flight.live_waiters;
    recompute_priority(&flight);
    if (!flight.live_waiters) {
      if (flight.state == FlightState::kQueued) flight.state = FlightState::kFailed;
      else if (flight.state == FlightState::kActive) {
        executor_->cancel(flight.id);
        flight.state = FlightState::kDraining;
      }
    }
  }
}

size_t TransferScheduler::lane_active(const FlightKey& key) const {
  if (key.target == TransferTarget::kHost) return d2h_active_;
  if (key.target == TransferTarget::kGpu) return h2d_active_;
  return p2p_active_;
}
size_t TransferScheduler::lane_limit(const FlightKey& key) const {
  if (key.target == TransferTarget::kHost) return max_d2h_active_;
  if (key.target == TransferTarget::kGpu) return max_h2d_active_;
  return max_p2p_active_;
}
void TransferScheduler::change_lane_active(const FlightKey& key, int delta) {
  size_t* value = key.target == TransferTarget::kHost ? &d2h_active_
      : key.target == TransferTarget::kGpu ? &h2d_active_ : &p2p_active_;
  if (delta > 0) ++*value;
  else { if (!*value) std::terminate(); --*value; }
  stats_.max_d2h_active = std::max<uint64_t>(stats_.max_d2h_active, d2h_active_);
  stats_.max_h2d_active = std::max<uint64_t>(stats_.max_h2d_active, h2d_active_);
  stats_.max_p2p_active = std::max<uint64_t>(stats_.max_p2p_active, p2p_active_);
}

bool TransferScheduler::drain() {
  while (true) {
    dispatch();
    bool had_physical = false;
    bool safe = true;
    for (auto& item : flights_) {
      auto& flight = item.second;
      if (flight.state == FlightState::kActive || flight.state == FlightState::kDraining ||
          flight.state == FlightState::kQuarantined) {
        had_physical = true;
        finish(flight, executor_->drain(flight.id));
        safe = terminal(flight.state) && safe;
      }
    }
    if (!safe) return false;
    bool queued = false;
    for (const auto& item : flights_) queued |= item.second.state == FlightState::kQueued;
    if (!queued && active_ == 0) return true;
    if (!had_physical && active_ == 0) {
      dispatch();
      if (active_ == 0) return false;
    }
  }
}

bool TransferScheduler::state(WaiterId waiter, WaiterState* state) const {
  const auto it = waiters_.find(waiter);
  if (!state || it == waiters_.end()) return false;
  *state = it->second.state;
  return true;
}

bool TransferScheduler::consume(WaiterId waiter) {
  const auto it = waiters_.find(waiter);
  if (it == waiters_.end() || it->second.state == WaiterState::kPending) return false;
  const auto flight = it->second.flight;
  waiters_.erase(it);
  cleanup_flight(flight);
  return true;
}

void TransferScheduler::cleanup_flight(FlightId id) {
  const auto it = flights_.find(id);
  if (it == flights_.end() || !terminal(it->second.state)) return;
  for (auto waiter : it->second.waiters) if (waiters_.count(waiter)) return;
  for (const auto& item : flights_) {
    if (item.first != id && std::find(item.second.dependencies.begin(),
                                      item.second.dependencies.end(), id) !=
                                item.second.dependencies.end()) return;
  }
  by_key_.erase(it->second.key);
  flights_.erase(it);
}

bool TransferScheduler::terminal(FlightState state) const {
  return state == FlightState::kSucceeded || state == FlightState::kFailed;
}

}  // namespace cache
