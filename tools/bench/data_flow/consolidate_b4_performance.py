#!/usr/bin/env python3
"""Consolidate the already-complete E2/E3 trials without rerunning GPUs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (11, 23, 37, 41, 53)
PLACEMENT_POLICIES = ("fixed", "round_robin", "data_aware")
RECOVERY_POLICIES = ("drop_recompute", "fixed_gpu_retain", "dependency_host_checkpoint")


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("docs/data_flow_evidence/v4"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/data_flow_evidence/v4/performance_characterization/B4"),
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    placement_trials = []
    placement_requests = []
    for policy in PLACEMENT_POLICIES:
        for seed in SEEDS:
            source = args.source_root / "E2_final" / policy / f"seed-{seed}" / "result.json"
            trial = load(source)
            placement_trials.append({
                "policy": policy,
                "seed": seed,
                "ok": trial["ok"],
                "requests": trial["requests"],
                "rejected": trial["rejected"],
                "observation_window_ms": trial["observation_window_ms"],
                "throughput_requests_per_s": trial["throughput_requests_per_s"],
                "throughput_definition": trial["throughput_definition"],
                "source": str(source),
            })
            for row in trial["trace"]:
                output = row.get("output", {})
                placement_requests.append({
                    "experiment": "E2_placement",
                    "policy": policy,
                    "seed": seed,
                    "request_id": row["request_id"],
                    "ok": row["result_ok"],
                    "worker": row["worker"],
                    "worker_pid": row["worker_pid"],
                    "coordinator_latency_ms": row["client_language_ms"],
                    "queue_wait_ms": row["actual_queue_wait_ms"],
                    "server_ttft_ms": output.get("ttft_ms"),
                    "server_itl_ms": output.get("itl_ms"),
                    "decision_us": row["decision"].get("decision_us"),
                    "prediction_error_ms": row.get("prediction_error_ms"),
                    "timing_boundary": "coordinator and server; non-streaming client TTFT unavailable",
                    "source": str(source),
                })

    pressure = load(args.source_root / "M9" / "pressure_policy_ab" / "results.json")
    recovery_rows = []
    recovery_requests = []
    for policy in RECOVERY_POLICIES:
        for repeat, row in enumerate(pressure["policies"][policy]["rows"]):
            normalized = {
                "policy": policy,
                "repeat": repeat,
                "latency_ms": row["latency_ms"],
                "server_ttft_ms": row["ttft_ms"],
                "recomputed_tokens": row["recomputed_tokens"],
                "matched_tokens": row["matched_tokens"],
                "host_demoted_blocks": row.get("host_demoted_blocks", 0),
                "host_restored_blocks": row.get("host_restored_blocks", 0),
                "restore_failures": row.get("restore_failures", 0),
                "checkpoint_events": row.get("checkpoint_events", []),
                "source": "docs/data_flow_evidence/v4/M9/pressure_policy_ab/results.json",
            }
            recovery_rows.append(normalized)
            recovery_requests.append({"experiment": "E3_recovery", **normalized})

    placement_summary = {}
    for policy in PLACEMENT_POLICIES:
        rows = [row for row in placement_trials if row["policy"] == policy]
        placement_summary[policy] = {
            "trials": len(rows),
            "requests": sum(row["requests"] for row in rows),
            "rejected": sum(row["rejected"] for row in rows),
            "mean_throughput_requests_per_s": statistics.mean(
                row["throughput_requests_per_s"] for row in rows
            ),
            "median_throughput_requests_per_s": median(rows, "throughput_requests_per_s"),
            "throughput_definition": rows[0]["throughput_definition"],
        }
    recovery_summary = {}
    for policy in RECOVERY_POLICIES:
        rows = [row for row in recovery_rows if row["policy"] == policy]
        recovery_summary[policy] = {
            "trials": len(rows),
            "median_latency_ms": median(rows, "latency_ms"),
            "median_server_ttft_ms": median(rows, "server_ttft_ms"),
            "median_recomputed_tokens": median(rows, "recomputed_tokens"),
            "host_demoted_blocks": sum(row["host_demoted_blocks"] for row in rows),
            "host_restored_blocks": sum(row["host_restored_blocks"] for row in rows),
            "restore_failures": sum(row["restore_failures"] for row in rows),
        }

    placement_ok = (
        len(placement_trials) == 15
        and all(row["ok"] and row["requests"] == 11 and row["rejected"] == 0 for row in placement_trials)
        and all(
            row["throughput_definition"] == "completed_requests / dispatch-to-last-completion wall window"
            for row in placement_trials
        )
    )
    recovery_ok = (
        len(recovery_rows) == 15
        and recovery_summary["dependency_host_checkpoint"]["host_demoted_blocks"]
        == recovery_summary["dependency_host_checkpoint"]["host_restored_blocks"]
        and recovery_summary["dependency_host_checkpoint"]["restore_failures"] == 0
    )
    results = {
        "ok": placement_ok and recovery_ok,
        "method": "existing raw evidence recomputation; no GPU rerun",
        "placement": {"ok": placement_ok, "policies": placement_summary, "trials": placement_trials},
        "recovery": {"ok": recovery_ok, "policies": recovery_summary, "trials": recovery_rows},
        "boundaries": [
            "E2 throughput is completed requests divided by the per-trial dispatch-to-last-completion wall window.",
            "E2 client_language_ms is coordinator-observed full-call latency, not true streaming client TTFT.",
            "E3 timings are server-side and do not include a true streaming client timestamp.",
            "Existing negative results are retained: placement and recovery policies may trade latency for reuse or throughput.",
        ],
    }
    dump(out / "results.json", results)
    dump(out / "commands.json", {
        "replay": "PYTHONPATH=python .venv/bin/python tools/bench/data_flow/consolidate_b4_performance.py",
        "inputs": [
            "docs/data_flow_evidence/v4/E2_final/{fixed,round_robin,data_aware}/seed-{11,23,37,41,53}/result.json",
            "docs/data_flow_evidence/v4/M9/pressure_policy_ab/results.json",
        ],
        "gpu_rerun": False,
    })
    with (out / "placement_requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in placement_requests:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "recovery_requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in recovery_requests:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"ok": results["ok"], "placement_requests": len(placement_requests), "recovery_trials": len(recovery_rows)}))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
