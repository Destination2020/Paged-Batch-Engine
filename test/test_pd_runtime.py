import time

import pytest

from pbe_roles.pd_runtime import PDRequestState, RoleRegistry


class Process:
    def __init__(self, pid, incarnation):
        self.calls = 0
        self.ready = {"worker_pid": pid, "worker_incarnation": incarnation,
                      "capabilities": ["prefill", "decode"], "gpu_uuid": "gpu",
                      "model_layout": "layout", "dtype": "bf16"}
    def call(self, request):
        self.calls += 1
        return {"ok": True, "worker_pid": self.ready["worker_pid"]}


def test_terminal_request_cannot_be_revived():
    state = PDRequestState("r", 1, time.monotonic_ns() + 1_000_000)
    state.transition("place"); state.transition("reserve")
    state.transition("cancelled")
    with pytest.raises(RuntimeError, match="terminal"):
        state.transition("prefill")
    assert all(item["span_id"] and item["parent_span_id"] and
               item["begin_ns"] <= item["end_ns"] for item in state.trace)
    assert state.trace[-1]["status"] == "cancelled"


def test_registry_rejects_stale_incarnation_and_tracks_replacement():
    registry = RoleRegistry()
    first = Process(10, 100)
    registry.register("p0", first, {"prefill"}, "local")
    replacement = Process(11, 101)
    registry.register("p0", replacement, {"prefill"}, "local")
    assert registry.candidates("prefill")[0].pid == 11
    with pytest.raises(ValueError, match="stale incarnation"):
        registry.register("p0", first, {"prefill"}, "local")


def test_static_registry_refreshes_identity_after_observation_ttl():
    registry = RoleRegistry(max_staleness_ms=1)
    process = Process(10, 100)
    registry.register("p0", process, {"prefill"}, "local")
    time.sleep(0.003)
    assert registry.candidates("prefill")[0].pid == 10
    assert process.calls == 1
