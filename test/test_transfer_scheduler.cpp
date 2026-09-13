#include <gtest/gtest.h>

#include <map>
#include <thread>
#include <vector>

#include "cache/transfer_scheduler.h"

namespace {
class ControlledExecutor final : public cache::FlightExecutor {
 public:
  cache::FlightExecutionResult submit(cache::FlightId id,
      const cache::FlightKey& key) noexcept override {
    submissions.push_back(key.page);
    results[id] = cache::FlightExecutionResult::kPending;
    return cache::FlightExecutionResult::kPending;
  }
  cache::FlightExecutionResult poll(cache::FlightId id) noexcept override {
    return results[id];
  }
  bool cancel(cache::FlightId id) noexcept override {
    cancelled.push_back(id);
    return true;
  }
  cache::FlightExecutionResult drain(cache::FlightId id) noexcept override {
    auto& result = results[id];
    if (result == cache::FlightExecutionResult::kPending ||
        result == cache::FlightExecutionResult::kUnknown)
      result = cache::FlightExecutionResult::kFailedSafe;
    return result;
  }
  void release(cache::FlightId id) noexcept override { released.push_back(id); }
  std::map<cache::FlightId, cache::FlightExecutionResult> results;
  std::vector<uint64_t> submissions;
  std::vector<cache::FlightId> cancelled;
  std::vector<cache::FlightId> released;
};

TEST(TransferSchedulerTest, ThirtyTwoWaitersShareOneFlightAndCancelIndependently) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferScheduler scheduler(4, 40, 2, 10, std::move(executor));
  std::vector<cache::WaiterTicket> tickets(32);
  for (auto& ticket : tickets) {
    ASSERT_TRUE(scheduler.submit({77, cache::TransferTarget::kGpu, 0},
        cache::TransferPriority::kPrefill, {}, &ticket));
    EXPECT_EQ(ticket.flight, tickets.front().flight);
  }
  EXPECT_EQ(control->submissions, std::vector<uint64_t>({77}));
  EXPECT_EQ(scheduler.stats().physical_submissions, 1u);
  EXPECT_EQ(scheduler.stats().merged_waiters, 31u);
  for (size_t i = 0; i < 31; ++i) ASSERT_TRUE(scheduler.cancel(tickets[i].waiter));
  EXPECT_TRUE(control->cancelled.empty());
  control->results[tickets.back().flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  for (size_t i = 0; i < tickets.size(); ++i) {
    cache::WaiterState state;
    ASSERT_TRUE(scheduler.state(tickets[i].waiter, &state));
    EXPECT_EQ(state, i + 1 == tickets.size() ? cache::WaiterState::kSucceeded
                                             : cache::WaiterState::kCancelled);
    EXPECT_TRUE(scheduler.consume(tickets[i].waiter));
  }
  EXPECT_EQ(scheduler.flight_count(), 0u);
  EXPECT_EQ(scheduler.waiter_count(), 0u);
}

TEST(TransferSchedulerTest, LastWaiterCancellationDrainsPhysicalFlight) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferScheduler scheduler(1, 2, 1, 4, std::move(executor));
  cache::WaiterTicket ticket;
  ASSERT_TRUE(scheduler.submit({1, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kBackground, {}, &ticket));
  ASSERT_TRUE(scheduler.cancel(ticket.waiter));
  ASSERT_EQ(control->cancelled.size(), 1u);
  EXPECT_EQ(scheduler.active_count(), 1u);
  EXPECT_TRUE(scheduler.drain());
  EXPECT_EQ(scheduler.active_count(), 0u);
  cache::WaiterState state;
  ASSERT_TRUE(scheduler.state(ticket.waiter, &state));
  EXPECT_EQ(state, cache::WaiterState::kCancelled);
  EXPECT_TRUE(scheduler.consume(ticket.waiter));
}

TEST(TransferSchedulerTest, DonationChangesDispatchOrderAndDependenciesFormWave) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferScheduler scheduler(5, 8, 1, 100, std::move(executor));
  cache::WaiterTicket active, low, donated, medium, dependent;
  ASSERT_TRUE(scheduler.submit({1, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {}, &active));
  ASSERT_TRUE(scheduler.submit({2, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kBackground, {}, &low));
  ASSERT_TRUE(scheduler.submit({3, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kPrefill, {}, &medium));
  ASSERT_TRUE(scheduler.submit({2, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {}, &donated));
  ASSERT_EQ(low.flight, donated.flight);
  ASSERT_TRUE(scheduler.submit({4, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {medium.flight}, &dependent));
  control->results[active.flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  ASSERT_EQ(control->submissions, std::vector<uint64_t>({1, 2}));
  control->results[low.flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  ASSERT_EQ(control->submissions, std::vector<uint64_t>({1, 2, 3}));
  control->results[medium.flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  EXPECT_EQ(control->submissions, std::vector<uint64_t>({1, 2, 3, 4}));
  EXPECT_EQ(scheduler.stats().priority_donations, 1u);
}

TEST(TransferSchedulerTest, CapacityRejectsWithoutCreatingIntentOrAddressState) {
  auto executor = std::make_unique<ControlledExecutor>();
  cache::TransferScheduler scheduler(1, 2, 1, 5, std::move(executor));
  cache::WaiterTicket first, second{99, 99};
  ASSERT_TRUE(scheduler.submit({1, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kBackground, {}, &first));
  EXPECT_FALSE(scheduler.submit({2, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kDecode, {}, &second));
  EXPECT_EQ(second.waiter, 99u);
  EXPECT_EQ(second.flight, 99u);
  EXPECT_EQ(scheduler.flight_count(), 1u);
  EXPECT_EQ(scheduler.waiter_count(), 1u);
}

TEST(TransferSchedulerTest, AgingBoundsBackgroundWaitUnderContinuousDecodeArrivals) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferScheduler scheduler(20, 20, 1, 1, std::move(executor));
  cache::WaiterTicket active, background;
  ASSERT_TRUE(scheduler.submit({1, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {}, &active));
  ASSERT_TRUE(scheduler.submit({2, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kBackground, {}, &background));

  cache::FlightId running = active.flight;
  for (uint64_t page = 3; page < 15 &&
       std::find(control->submissions.begin(), control->submissions.end(), 2) ==
           control->submissions.end(); ++page) {
    cache::WaiterTicket demand;
    ASSERT_TRUE(scheduler.submit({page, cache::TransferTarget::kGpu, 0},
        cache::TransferPriority::kDecode, {}, &demand));
    control->results[running] = cache::FlightExecutionResult::kSucceeded;
    scheduler.poll();
    running = control->results.rbegin()->first;
  }
  const auto background_it =
      std::find(control->submissions.begin(), control->submissions.end(), 2);
  ASSERT_NE(background_it, control->submissions.end());
  EXPECT_LE(std::distance(control->submissions.begin(), background_it), 9);
  EXPECT_GE(scheduler.stats().oldest_queue_ticks, 8u);
}

TEST(TransferSchedulerTest, DirectionLanesRunIndependently) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferSchedulerConfig config{8, 8, 3, 1, 1, 1, 10};
  cache::TransferScheduler scheduler(config, std::move(executor));
  cache::WaiterTicket d2h1, d2h2, h2d, p2p;
  ASSERT_TRUE(scheduler.submit({1, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kBackground, {}, &d2h1));
  ASSERT_TRUE(scheduler.submit({2, cache::TransferTarget::kHost, -1},
      cache::TransferPriority::kBackground, {}, &d2h2));
  ASSERT_TRUE(scheduler.submit({3, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {}, &h2d));
  ASSERT_TRUE(scheduler.submit({4, cache::TransferTarget::kPeerGpu, 1},
      cache::TransferPriority::kDecode, {}, &p2p));
  EXPECT_EQ(control->submissions, std::vector<uint64_t>({1, 3, 4}));
  EXPECT_EQ(scheduler.active_count(), 3u);
  EXPECT_EQ(scheduler.stats().max_d2h_active, 1u);
  EXPECT_EQ(scheduler.stats().max_h2d_active, 1u);
  EXPECT_EQ(scheduler.stats().max_p2p_active, 1u);
  control->results[d2h1.flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  EXPECT_EQ(control->submissions, std::vector<uint64_t>({1, 3, 4, 2}));
}

TEST(TransferSchedulerTest, LaneOffOnKeepsTotalCapacityAndProtectsDirections) {
  auto run = [](bool lanes) {
    auto executor = std::make_unique<ControlledExecutor>();
    auto* control = executor.get();
    cache::TransferSchedulerConfig config{8, 8, 2, 1, 1, 1, 10};
    config.direction_lanes_enabled = lanes;
    cache::TransferScheduler scheduler(config, std::move(executor));
    cache::WaiterTicket h2d_a, h2d_b, d2h;
    EXPECT_TRUE(scheduler.submit({1, cache::TransferTarget::kGpu, 0},
        cache::TransferPriority::kDecode, {}, &h2d_a));
    EXPECT_TRUE(scheduler.submit({2, cache::TransferTarget::kGpu, 0},
        cache::TransferPriority::kDecode, {}, &h2d_b));
    EXPECT_TRUE(scheduler.submit({3, cache::TransferTarget::kHost, -1},
        cache::TransferPriority::kBackground, {}, &d2h));
    const bool protected_direction_started =
        std::find(control->submissions.begin(), control->submissions.end(), 3) !=
        control->submissions.end();
    control->results[h2d_a.flight] = cache::FlightExecutionResult::kSucceeded;
    scheduler.poll();
    control->results[h2d_b.flight] = cache::FlightExecutionResult::kSucceeded;
    scheduler.poll();
    control->results[d2h.flight] = cache::FlightExecutionResult::kSucceeded;
    scheduler.poll();
    EXPECT_TRUE(scheduler.consume(h2d_a.waiter));
    EXPECT_TRUE(scheduler.consume(h2d_b.waiter));
    EXPECT_TRUE(scheduler.consume(d2h.waiter));
    return protected_direction_started;
  };
  EXPECT_FALSE(run(false));
  EXPECT_TRUE(run(true));
}

TEST(TransferSchedulerTest, DemandReserveSurvivesBackgroundSaturation) {
  auto executor = std::make_unique<ControlledExecutor>();
  cache::TransferSchedulerConfig config{4, 5, 1, 1, 1, 1, 10, 1, 1};
  cache::TransferScheduler scheduler(config, std::move(executor));
  std::vector<cache::WaiterTicket> background(3);
  for (size_t i = 0; i < background.size(); ++i)
    ASSERT_TRUE(scheduler.submit({i + 1, cache::TransferTarget::kHost, -1},
        cache::TransferPriority::kBackground, {}, &background[i]));
  cache::WaiterTicket rejected;
  EXPECT_EQ(scheduler.submit_typed({9, cache::TransferTarget::kHost, -1},
                cache::TransferPriority::kBackground, {},
                std::chrono::steady_clock::time_point::max(), &rejected),
            cache::SubmitOutcome::kRetryCapacity);
  cache::WaiterTicket demand;
  EXPECT_EQ(scheduler.submit_typed({10, cache::TransferTarget::kGpu, 0},
                cache::TransferPriority::kDecode, {},
                std::chrono::steady_clock::time_point::max(), &demand),
            cache::SubmitOutcome::kAdmitted);
}

TEST(TransferSchedulerTest, TypedJoinAndAbsoluteDeadlineKeepPhysicalFlightSafe) {
  auto executor = std::make_unique<ControlledExecutor>();
  auto* control = executor.get();
  cache::TransferScheduler scheduler(2, 4, 1, 10, std::move(executor));
  cache::WaiterTicket long_lived, expiring;
  EXPECT_EQ(scheduler.submit_typed({9, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kBackground, {},
      std::chrono::steady_clock::time_point::max(), &long_lived),
      cache::SubmitOutcome::kAdmitted);
  EXPECT_EQ(scheduler.submit_typed({9, cache::TransferTarget::kGpu, 0},
      cache::TransferPriority::kDecode, {},
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1), &expiring),
      cache::SubmitOutcome::kJoined);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  scheduler.poll();
  cache::WaiterState state;
  ASSERT_TRUE(scheduler.state(expiring.waiter, &state));
  EXPECT_EQ(state, cache::WaiterState::kTimedOut);
  EXPECT_TRUE(control->cancelled.empty());
  control->results[long_lived.flight] = cache::FlightExecutionResult::kSucceeded;
  scheduler.poll();
  ASSERT_TRUE(scheduler.state(long_lived.waiter, &state));
  EXPECT_EQ(state, cache::WaiterState::kSucceeded);
  EXPECT_EQ(scheduler.stats().timed_out_waiters, 1u);
}
}  // namespace
