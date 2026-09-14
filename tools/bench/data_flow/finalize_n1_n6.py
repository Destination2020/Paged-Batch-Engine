#!/usr/bin/env python3
"""Create the bounded N1--N6 result index from accepted evidence files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    n1 = load(root / "N1/oracle_free_e4/results.json")
    n2 = load(root / "N2/latency_breakdown.json")
    n3 = load(root / "N3/result.json")
    n4 = load(root / "N4/result.json")
    n5 = load(root / "N5/multi_pd_reservation_cancel_v2/result.json")
    n6 = load(root / "N6/topology_analysis_v2.json")
    capacity = load(root / "N6/capacity/results.json")
    sanitizer = load(root / "N5/compute_sanitizer_reservation_v2/result.json")
    numerical = load(root / "N6/numerical_regression_v2/validation.json")
    phase_ok = {"N1": n1["ok"], "N2": n2["ok"], "N3": n3["ok"],
                "N4": n4["ok"], "N5": n5["ok"] and sanitizer["ok"],
                "N6": n6["ok"] and capacity["ok"] and
                      numerical["event"] == "validated_20_fixtures"}
    result = {
        "schema": "pbe-v4-section21-results-v1",
        "ok": all(phase_ok.values()), "accepted_phases": sum(phase_ok.values()),
        "total_phases": 6, "phases": phase_ok,
        "scope": "single host, one NVIDIA H20-3e GPU, TP=1, Qwen2.5-VL-3B BF16",
        "key_results": {
            "oracle_free_1P1D": n1["summary"],
            "topology_cells": len(n6["cells"]),
            "topology_independent_trials": len(n6["trials"]),
            "topology_requests": sum(x["completed_requests"] for x in n6["cells"]),
            "fixed_total_kv_slots": n6["fixed_kv_total_slots"],
            "capacity_private_last_success_blocks":
                capacity["summary"]["private"]["median_last_success_blocks"],
            "capacity_shared_last_success_blocks":
                capacity["summary"]["shared"]["median_last_success_blocks"],
            "current_numeric_cases": numerical["cases"],
            "current_numeric_exact_sequences": numerical["exact_sequences"],
            "current_numeric_accepted_bf16_divergences":
                len(numerical["accepted_bf16_divergences"]),
        },
        "boundaries": [
            "Client TTFT/ITL and SLO goodput are N/A for the non-streaming interface.",
            "Each topology cell has five independent observation windows; low/medium windows contain five requests and near-saturation windows contain eleven requests.",
            "Near-saturation is a finite 2,000 ms arrival train at 200 ms intervals followed by cohort drain, not an asymptotic saturation claim.",
            "The 18,000 MiB capacity result is an external-allocation bracket; requests accessed at most 48 blocks and it is not maximum live concurrency.",
            "The best observed topology depends on mode/workload/load and is not a universal N:P/D optimum.",
            "N2 preserves unexplained residuals and does not promote correlation to a leaf root cause.",
            "The independent Nsight trace is diagnostic and excluded from formal latency samples.",
            "N6 fixes workload, KV slots and device admission budget, but added role contexts/streams are measured deployment overhead rather than a hard equal stream partition.",
        ],
        "sources": {
            "attribution": "N2/ATTRIBUTION_REPORT.md",
            "topology": "TOPOLOGY_REPORT_V2.md",
            "replay": "REPLAY.md",
            "validator": "checks.json",
            "completion_manifest": "completion/manifest.json",
        },
    }
    (root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "accepted": result["accepted_phases"]},
                     sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
