import time

from pbe_roles.placement import WorkerSnapshot, choose


def role(worker, queue=0, resident=False, age_ms=0):
    now = time.monotonic_ns()
    return WorkerSnapshot(worker, f"inc-{worker}", "m", "r",
                          now - age_ms * 1_000_000, queue, 0, True,
                          1000, 1000, 1000, 0 if resident else 500,
                          100, .1, 1, .1)


def select(roles):
    return choose(roles, model_revision="m", representation="r",
                  required_bytes=10, prefix_key="key", prefix_bytes=500,
                  prefill_tokens=10, decode_tokens=2)


def test_queue_can_outweigh_locality():
    assert select([role(0, 100, True), role(1)])["worker"] == 1


def test_fresh_locality_wins():
    assert select([role(0, resident=True), role(1)])["worker"] == 0


def test_stale_stats_are_conservative_and_budget_is_filtered():
    stale = role(0, resident=True, age_ms=5000)
    fresh = role(1)
    assert select([stale, fresh])["worker"] == 1
    fresh.admissible = False
    assert not select([stale, fresh])["candidates"][1]["eligible"]
