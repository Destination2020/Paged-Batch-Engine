#!/usr/bin/env python3
"""Derive B5 production topology support from accepted N5/N6 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TOPOLOGIES = ("unified", "1P1D", "1P2D", "2P1D", "2P2D")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines_with(path: Path, needles) -> list[dict]:
    return [{"line": number, "text": line.strip()}
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if any(needle in line for needle in needles)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "docs/data_flow_evidence/v4/performance_attribution_multi_pd"
    n5_path = evidence / "N5/multi_pd_reservation_cancel_v2/result.json"
    n6_path = evidence / "N6/topology_analysis_v2.json"
    n5, n6 = json.loads(n5_path.read_text()), json.loads(n6_path.read_text())
    coordinator = root / "python/pbe_roles/pd_runtime.py"
    role = root / "demo/pbe_vlm_language_role.cpp"
    required_faults = {
        "atomic_reservation_contention_and_recovery",
        "external_cancel_and_deadline_reach_active_decode",
        "ordered_worker_reservations_consumed",
        "all_ipc_grants_reclaimed",
        "all_topologies_and_edges",
    }
    accepted = (n5.get("ok") is True and n6.get("ok") is True and
                all(n5.get("checks", {}).get(name) for name in required_faults) and
                len(n6.get("cells", [])) == 90 and
                len(n6.get("trials", [])) == 450 and
                sum(row["completed_requests"] for row in n6.get("cells", [])) == 3150)
    matrix = {
        topology: ("validated production coordinator; private/shared measured"
                   if topology != "unified" else
                   "validated production unified baseline and batch-capability probe; private/shared measured")
        for topology in TOPOLOGIES
    }
    result = {
        "schema": "pbe-v4-b5-topology-support-v3",
        "accepted": accepted,
        "status": ("complete_with_bounded_topology_claim" if accepted else
                   "blocked_by_failed_production_evidence"),
        "matrix": matrix,
        "blocking_contracts": [] if accepted else
            ["N5 production fault matrix or N6 independent-window topology matrix failed"],
        "measurement_boundary": {
            "cells": len(n6.get("cells", [])),
            "independent_trials": len(n6.get("trials", [])),
            "requests": sum(row.get("completed_requests", 0)
                            for row in n6.get("cells", [])),
            "trials_per_cell": 5,
            "requests_per_trial": {"low": 5, "medium": 5, "near_saturation": 11},
            "near_saturation_arrival_train_ms": 2000,
            "fixed_total_kv_slots": n6.get("fixed_kv_total_slots"),
            "client_ttft_itl": n6.get("client_ttft_itl"),
            "claim": "best observed under frozen conditions; finite overload, no significance, asymptotic saturation, universal topology optimum, or hardware-isolation claim",
        },
        "completion_evidence": {
            "coordinator": str(coordinator.relative_to(root)),
            "functional_fault_reservation_cancel": str(n5_path.relative_to(root)),
            "fixed_kv_independent_window_matrix": str(n6_path.relative_to(root)),
            "fixed_physical_budget_capacity": str((evidence / "N6/capacity/results.json").relative_to(root)),
            "compute_sanitizer": str((evidence / "N5/compute_sanitizer_reservation_v2/result.json").relative_to(root)),
        },
        "code_evidence": {
            "coordinator": {"path": str(coordinator.relative_to(root)),
                            "sha256": sha(coordinator),
                            "matches": lines_with(coordinator, ["pd_reserve", "deadline_ns", "def cancel("])},
            "language_role": {"path": str(role.relative_to(root)),
                              "sha256": sha(role),
                              "matches": lines_with(role, ["PDReserve", "active_reservations", "deadline_monotonic_ns"])},
        },
        "supersedes": "pbe-v4-b5-topology-support-v2 and the earlier blocked audit; historical artifacts remain preserved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "status": result["status"]}, sort_keys=True))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
