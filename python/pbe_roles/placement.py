"""Explainable, bounded multi-role placement used by the VLM coordinator."""
from __future__ import annotations

import dataclasses
import time


@dataclasses.dataclass
class WorkerSnapshot:
    worker: int
    incarnation: str
    model_revision: str
    representation: str
    observed_ns: int
    queued_tokens: int
    inflight: int
    admissible: bool
    kv_free_bytes: int
    bundle_free_bytes: int
    staging_free_bytes: int
    missing_prefix_bytes: int
    bandwidth_bytes_per_ms: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    rpc_layout_ms: float


def choose(snapshot: list[WorkerSnapshot], *, model_revision: str,
           representation: str, required_bytes: int, prefix_key: str,
           prefix_bytes: int, prefill_tokens: int, decode_tokens: int,
           excluded: set[int] | None = None, stale_after_ms: float = 1000.0,
           conservative_queue_tokens: int = 1024,
           max_inflight: int = 8, now_ns: int | None = None) -> dict:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    excluded = excluded or set()
    candidates = []
    for role in snapshot:
        age_ms = max(0.0, (now_ns - role.observed_ns) / 1_000_000)
        stale = age_ms > stale_after_ms
        reason = "eligible"
        if role.worker in excluded:
            reason = "excluded_after_reserve_failure"
        elif (role.model_revision, role.representation) != (model_revision, representation):
            reason = "incompatible_model_or_representation"
        elif not role.admissible:
            reason = "insufficient_admission_capacity"
        elif role.inflight >= max_inflight:
            reason = "hotspot_inflight_limit"
        eligible = reason == "eligible"
        queued = max(role.queued_tokens, conservative_queue_tokens) if stale else role.queued_tokens
        missing = role.missing_prefix_bytes
        queue_ms = queued * role.decode_ms_per_token
        transfer_ms = missing / max(role.bandwidth_bytes_per_ms, 1e-9)
        compute_ms = (prefill_tokens * role.prefill_ms_per_token +
                      decode_tokens * role.decode_ms_per_token)
        total_ms = queue_ms + transfer_ms + compute_ms + role.rpc_layout_ms
        candidates.append({"worker": role.worker, "incarnation": role.incarnation,
                           "eligible": eligible, "reason": reason,
                           "stats_age_ms": age_ms, "stale": stale,
                           "capacity": {"kv_free_bytes": role.kv_free_bytes,
                                        "bundle_free_bytes": role.bundle_free_bytes,
                                        "staging_free_bytes": role.staging_free_bytes},
                           "queue_wait_ms": queue_ms, "missing_bytes": missing,
                           "transfer_ms": transfer_ms, "compute_ms": compute_ms,
                           "rpc_layout_ms": role.rpc_layout_ms,
                           "total_ms": total_ms})
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        return {"ok": False, "reason": "no_eligible_worker", "candidates": candidates}
    selected = min(eligible, key=lambda item: (item["total_ms"], item["worker"]))
    return {"ok": True, "worker": selected["worker"],
            "incarnation": selected["incarnation"],
            "predicted_ms": selected["total_ms"], "candidates": candidates}
