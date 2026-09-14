"""Production coordinator state machine for persistent multi-process P/D roles."""
from __future__ import annotations

import dataclasses
import concurrent.futures
import threading
import time
from typing import Any

from pbe_roles.coordinator import LanguageProcess, encode
from pbe_roles.vision.client import call as vision_call


TERMINAL = {"finished", "cancelled", "failed", "timed_out", "rejected"}
TRANSITIONS = {
    "prepare": {"place", "cancelled", "timed_out", "rejected", "failed"},
    "place": {"reserve", "cancelled", "timed_out", "rejected", "failed"},
    "reserve": {"prefill", "cancelled", "timed_out", "rejected", "failed"},
    "prefill": {"handoff", "cancelled", "timed_out", "failed"},
    "handoff": {"decode", "cancelled", "timed_out", "failed"},
    "decode": {"finished", "cancelled", "timed_out", "failed"},
}


@dataclasses.dataclass
class RoleRecord:
    worker_id: str
    process: LanguageProcess
    capabilities: frozenset[str]
    endpoint: str
    incarnation: int
    pid: int
    gpu_uuid: str
    model_layout: str
    dtype: str
    state: str = "ready"
    observed_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    completed: int = 0
    failed: int = 0


class RoleRegistry:
    """Static registry with restart-generation and readiness checks."""

    def __init__(self, max_staleness_ms: int = 10_000):
        self.max_staleness_ns = max_staleness_ms * 1_000_000
        self._roles: dict[str, RoleRecord] = {}
        self._seen: dict[str, set[int]] = {}
        self._lock = threading.Lock()

    def register(self, worker_id: str, process: LanguageProcess,
                 capabilities: set[str], endpoint: str) -> RoleRecord:
        ready = process.ready
        advertised = set(ready.get("capabilities", []))
        if not capabilities or not capabilities <= advertised:
            raise ValueError(f"capability mismatch for {worker_id}")
        incarnation = int(ready["worker_incarnation"])
        record = RoleRecord(worker_id, process, frozenset(capabilities), endpoint,
                            incarnation, int(ready["worker_pid"]), ready["gpu_uuid"],
                            ready["model_layout"], ready["dtype"])
        with self._lock:
            if incarnation in self._seen.setdefault(worker_id, set()):
                raise ValueError(f"stale incarnation for {worker_id}")
            old = self._roles.get(worker_id)
            if old is not None: old.state = "draining"
            self._seen[worker_id].add(incarnation)
            self._roles[worker_id] = record
        return record

    def candidates(self, capability: str) -> list[RoleRecord]:
        now = time.monotonic_ns()
        with self._lock:
            configured = [r for r in self._roles.values()
                          if capability in r.capabilities and r.state == "ready"]
        # Static endpoints do not become permanently unusable merely because
        # no request arrived within the observation TTL. Revalidate identity
        # outside the registry lock; reservation remains a separate atomic op.
        for record in configured:
            if now - record.observed_ns > self.max_staleness_ns:
                try:
                    self.refresh(record)
                except Exception:
                    pass
        now = time.monotonic_ns()
        with self._lock:
            result = [r for r in configured if r.state == "ready" and
                      now - r.observed_ns <= self.max_staleness_ns]
        if not result:
            raise RuntimeError(f"no fresh ready {capability} worker")
        return sorted(result, key=lambda r: r.worker_id)

    def refresh(self, record: RoleRecord) -> dict[str, Any]:
        status = record.process.call({"op": "status"})
        if not status.get("ok") or status.get("worker_pid") != record.pid:
            record.state = "draining"
            raise RuntimeError(f"worker identity changed: {record.worker_id}")
        record.observed_ns = time.monotonic_ns()
        return status

    def snapshot(self) -> list[dict[str, Any]]:
        return [{"worker_id": r.worker_id, "pid": r.pid,
                 "incarnation": r.incarnation,
                 "capabilities": sorted(r.capabilities), "state": r.state,
                 "completed": r.completed, "failed": r.failed,
                 "gpu_uuid": r.gpu_uuid, "model_layout": r.model_layout,
                 "dtype": r.dtype} for r in sorted(self._roles.values(),
                                                    key=lambda x: x.worker_id)]


@dataclasses.dataclass
class PDRequestState:
    request_id: str
    generation: int
    deadline_ns: int
    stage: str = "prepare"
    terminal: str | None = None
    prefill_worker: str | None = None
    decode_worker: str | None = None
    trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    trial: int | None = None
    arm: str | None = None
    stage_started_ns: int = dataclasses.field(default_factory=time.monotonic_ns)

    def transition(self, target: str, **fields: Any) -> None:
        if self.terminal:
            raise RuntimeError("terminal request cannot transition")
        if target not in TRANSITIONS.get(self.stage, set()):
            raise RuntimeError(f"invalid transition {self.stage}->{target}")
        now = time.monotonic_ns()
        index = len(self.trace)
        self.trace.append({"request_id": self.request_id,
                           "generation": self.generation,
                           "trial": self.trial, "arm": self.arm,
                           "worker_id": fields.pop("worker_id", "coordinator"),
                           "worker_incarnation": fields.pop("worker_incarnation", 0),
                           "stage": self.stage, "next_stage": target,
                           "span_id": f"{self.request_id}:{self.generation}:{index}",
                           "parent_span_id": f"{self.request_id}:{self.generation}:root",
                           "begin_ns": self.stage_started_ns, "end_ns": now,
                           "status": target if target in TERMINAL else "ok", **fields})
        self.stage = target
        self.stage_started_ns = now
        if target in TERMINAL: self.terminal = target


class RequestCancelled(RuntimeError):
    pass


class RequestDeadline(RuntimeError):
    pass


class ProductionPDCoordinator:
    """Client→Vision→P→authorized KV handoff→D→client production path."""

    def __init__(self, registry: RoleRegistry, vision_endpoint: str):
        self.registry = registry
        self.vision_endpoint = vision_endpoint
        self._lock = threading.Lock()
        self._latest: dict[str, int] = {}
        self._active: dict[tuple[str, int], dict[str, Any]] = {}
        self._cancelled: set[tuple[str, int]] = set()
        self._rr_p = self._rr_d = 0

    @staticmethod
    def _remaining_seconds(deadline_ns: int) -> float:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise RequestDeadline("absolute_deadline_exceeded")
        return remaining

    def _check_active(self, state: PDRequestState) -> None:
        key = (state.request_id, state.generation)
        with self._lock:
            cancelled = key in self._cancelled
        if cancelled:
            raise RequestCancelled("external_cancel_accepted")
        if time.monotonic_ns() >= state.deadline_ns:
            raise RequestDeadline("absolute_deadline_exceeded")

    def _set_active_roles(self, key: tuple[str, int], prefill=None, decode=None,
                          reservations=None) -> None:
        with self._lock:
            active = self._active.get(key)
            if active is None:
                return
            if prefill is not None: active["prefill"] = prefill
            if decode is not None: active["decode"] = decode
            if reservations is not None: active["reservations"] = reservations

    def cancel(self, request_id: str, generation: int) -> dict[str, Any]:
        """Accept an external cancellation for an active request generation."""
        key = (request_id, int(generation))
        with self._lock:
            active = self._active.get(key)
            if active is None:
                return {"ok": False, "accepted": False, "reason": "not_active",
                        "request_id": request_id, "generation": generation}
            self._cancelled.add(key)
            workers = []
            for worker in (active.get("prefill"), active.get("decode")):
                if worker is not None and all(worker is not item for item in workers):
                    workers.append(worker)
        acknowledgements = []
        try:
            acknowledgements.append({"role": "vision", "response": vision_call(
                self.vision_endpoint, {"op": "cancel", "request_id": request_id,
                                       "generation": generation})})
        except Exception as error:
            acknowledgements.append({"role": "vision", "error": str(error)})
        for worker in workers:
            try:
                response = worker.process.call({"op": "cancel", "request_id": request_id,
                                                "generation": generation}, timeout=10)
                acknowledgements.append({"role": worker.worker_id, "response": response})
            except Exception as error:
                acknowledgements.append({"role": worker.worker_id, "error": str(error)})
        return {"ok": True, "accepted": True, "request_id": request_id,
                "generation": generation, "acks": acknowledgements}

    def active_state(self, request_id: str, generation: int) -> dict[str, Any] | None:
        """Return an immutable diagnostic snapshot of an active generation."""
        with self._lock:
            active = self._active.get((request_id, int(generation)))
            if active is None:
                return None
            state = active["state"]
            return {"request_id": request_id, "generation": generation,
                    "stage": state.stage, "terminal": state.terminal,
                    "prefill_worker": state.prefill_worker,
                    "decode_worker": state.decode_worker,
                    "reservation_roles": sorted(active["reservations"])}

    def _place(self, forced_prefill: str | None = None,
               forced_decode: str | None = None) -> tuple[RoleRecord, RoleRecord]:
        ps, ds = self.registry.candidates("prefill"), self.registry.candidates("decode")
        if forced_prefill:
            ps = [p for p in ps if p.worker_id == forced_prefill]
        if forced_decode:
            ds = [d for d in ds if d.worker_id == forced_decode]
        if not ps or not ds: raise RuntimeError("forced role is not ready")
        with self._lock:
            p, d = ps[self._rr_p % len(ps)], ds[self._rr_d % len(ds)]
            self._rr_p += 1; self._rr_d += 1
        if p.pid == d.pid: raise RuntimeError("P and D must be independent processes")
        return p, d

    def submit(self, item: dict[str, Any]) -> dict[str, Any]:
        request_id, generation = item["request_id"], int(item.get("generation", 1))
        deadline = int(item.get("deadline_monotonic_ns") or
                       (time.monotonic_ns() + int(item.get("timeout_ms", 120_000))*1_000_000))
        state = PDRequestState(request_id, generation, deadline,
                               trial=item.get("trial"), arm=item.get("arm"))
        key = (request_id, generation)
        with self._lock:
            if generation <= self._latest.get(request_id, 0):
                state.transition("rejected", reason="stale_or_duplicate_generation")
                return {"ok": False, "state": dataclasses.asdict(state)}
            self._latest[request_id] = generation
            self._active[key] = {"state": state, "prefill": None, "decode": None,
                                 "reservations": {}}
        prefill = decode = None; handoff = None
        reservations: dict[str, dict[str, Any]] = {}
        payload: dict[str, Any] = {}
        reclaim_results: list[dict[str, Any]] = []
        try:
            self._check_active(state)
            stamped = dict(item, _deadline_monotonic_ns=deadline)
            state.transition("place")
            _, vision = encode(self.vision_endpoint, stamped)
            if not vision.get("ok"): raise RuntimeError("vision_encode_failed")
            self._check_active(state)
            prefill, decode = self._place(item.get("prefill_worker"),
                                          item.get("decode_worker"))
            state.prefill_worker, state.decode_worker = prefill.worker_id, decode.worker_id
            self._set_active_roles(key, prefill, decode)
            state.transition("reserve", prefill_pid=prefill.pid, decode_pid=decode.pid)
            request = {"request_id": request_id, "generation": generation,
                       "content": vision["content"],
                       "representation": vision["representation"],
                       "feature_content": vision["feature_content"],
                       "feature_representation": vision["feature_representation"],
                       "max_new_tokens": int(item.get("max_new_tokens", 16)),
                       "deadline_monotonic_ns": deadline,
                       "timeout_ms": max(1, (deadline-time.monotonic_ns())//1_000_000)}
            probe = prefill.process.call({"op": "probe", "request": request},
                                         timeout=self._remaining_seconds(deadline))
            if not probe.get("ok") or not probe.get("admissible"):
                raise RuntimeError("pd_prefill_probe_rejected")
            # Atomic worker-owned grants are acquired in a global D→P order.
            # If the second acquisition fails, finally rolls the first back.
            d_reservation = decode.process.call({"op": "pd_reserve", "role": "decode",
                "request_id": request_id, "generation": generation,
                "max_new_tokens": request["max_new_tokens"],
                "deadline_monotonic_ns": deadline},
                timeout=self._remaining_seconds(deadline))
            if not d_reservation.get("ok"):
                raise RuntimeError(d_reservation.get("error", "decode_reserve_failed"))
            reservations["decode"] = d_reservation
            self._set_active_roles(key, reservations=reservations)
            self._check_active(state)
            p_reservation = prefill.process.call({"op": "pd_reserve", "role": "prefill",
                "request_id": request_id, "generation": generation,
                "max_new_tokens": request["max_new_tokens"],
                "deadline_monotonic_ns": deadline, "required": probe["required"]},
                timeout=self._remaining_seconds(deadline))
            if not p_reservation.get("ok"):
                raise RuntimeError(p_reservation.get("error", "prefill_reserve_failed"))
            reservations["prefill"] = p_reservation
            self._set_active_roles(key, reservations=reservations)
            state.transition("prefill", d_reservation=d_reservation,
                             p_reservation=p_reservation, probe=probe)
            self._check_active(state)
            handoff = prefill.process.call({"op": "pd_prefill", "request": request,
                                            "reservation_id": p_reservation["reservation_id"],
                                            "oracle_steps": 0},
                                           timeout=self._remaining_seconds(deadline))
            reservations.pop("prefill", None)
            if not handoff.get("ok"): raise RuntimeError(handoff.get("error", "prefill_failed"))
            state.transition("handoff", worker_id=prefill.worker_id,
                             worker_incarnation=prefill.incarnation, prefill=handoff)
            self._check_active(state)
            if item.get("cancel_before_decode"):
                with self._lock: self._cancelled.add(key)
                raise RequestCancelled("legacy_cancel_before_decode")
            state.transition("decode")
            output = decode.process.call({"op": "pd_decode", "request_id": request_id,
                "generation": generation, "kv_content": handoff["kv_content"],
                "kv_representation": handoff["kv_representation"],
                "expected_provider_incarnation": prefill.incarnation,
                "reservation_id": d_reservation["reservation_id"],
                "deadline_monotonic_ns": deadline,
                "hold_after_attach_ms": int(item.get("hold_after_attach_ms", 0)),
                "max_new_tokens": request["max_new_tokens"]},
                timeout=self._remaining_seconds(deadline))
            reservations.pop("decode", None)
            if not output.get("ok"): raise RuntimeError(output.get("error", "decode_failed"))
            self._check_active(state)
            state.transition("finished", worker_id=decode.worker_id,
                             worker_incarnation=decode.incarnation, decode=output)
            prefill.completed += 1; decode.completed += 1
            payload = {"ok": True, "tokens": output["tokens"], "text": output["text"],
                       "prefill": handoff, "decode": output}
        except (RequestCancelled,) as error:
            if not state.terminal:
                state.transition("cancelled", reason=str(error))
            payload = {"ok": False, "error": str(error)}
        except (RequestDeadline, concurrent.futures.TimeoutError) as error:
            self.cancel(request_id, generation)
            if not state.terminal:
                state.transition("timed_out", reason=str(error) or "rpc_deadline_exceeded")
            payload = {"ok": False, "error": str(error) or "rpc_deadline_exceeded"}
        except Exception as error:
            if not state.terminal:
                if key in self._cancelled or "cancelled" in str(error):
                    state.transition("cancelled", error=str(error))
                elif time.monotonic_ns() >= deadline or "deadline" in str(error):
                    state.transition("timed_out", error=str(error))
                else:
                    state.transition("failed", error=str(error))
            if prefill: prefill.failed += 1
            if decode: decode.failed += 1
            payload = {"ok": False, "error": str(error)}
        finally:
            reclaim_begin = time.monotonic_ns()
            for role, reservation in list(reservations.items())[::-1]:
                worker = decode if role == "decode" else prefill
                if worker is None: continue
                try:
                    released = worker.process.call({"op": "pd_unreserve",
                        "request_id": request_id, "generation": generation,
                        "reservation_id": reservation["reservation_id"]}, timeout=10)
                    reclaim_results.append({"role": role, "reservation": released})
                except Exception as error:
                    reclaim_results.append({"role": role,
                                            "reservation_release_error": str(error)})
            if handoff is not None and prefill is not None:
                try:
                    release = prefill.process.call({"op": "pd_release",
                        "request_id": request_id, "generation": generation}, timeout=10)
                    reclaim_results.append({"role": "prefill", "handoff": release})
                except Exception as error:
                    reclaim_results.append({"role": "prefill",
                                            "handoff_release_error": str(error)})
            state.trace.append({"request_id": request_id, "generation": generation,
                "trial": state.trial, "arm": state.arm, "worker_id": "coordinator",
                "worker_incarnation": 0, "stage": "reclaim",
                "span_id": f"{request_id}:{generation}:reclaim",
                "parent_span_id": f"{request_id}:{generation}:root",
                "begin_ns": reclaim_begin, "end_ns": time.monotonic_ns(),
                "status": "ok" if not any("error" in str(x) for x in reclaim_results)
                                  else "release_error",
                "results": reclaim_results})
            with self._lock:
                self._active.pop(key, None)
                self._cancelled.discard(key)
        payload["state"] = dataclasses.asdict(state)
        payload["reclaim"] = reclaim_results
        return payload
