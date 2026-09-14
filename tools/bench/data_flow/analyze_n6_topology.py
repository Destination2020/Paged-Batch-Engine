#!/usr/bin/env python3
"""Aggregate N6 v2 independent-window topology evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TOPOLOGIES = ("unified", "1P1D", "1P2D", "2P1D", "2P2D")
MODES = ("private", "shared")
LOADS = ("low", "medium", "near_saturation")


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def overlap_peak(records):
    events = []
    for row in records:
        events.extend(((row["started"], 1), (row["completed"], -1)))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def split_spans_ms(row):
    response = row["response"]
    if "state" not in response:
        output = next((item for item in response["outputs"]
                       if item.get("request_id") == row["request_id"]),
                      response["outputs"][0])
        model = float(output["latency_ms"])
        return {"model_ms": model, "handoff_attach_ms": 0.0,
                "pre_model_and_queue_ms": row["client_ms"] - model}
    spans = {item["stage"]: (item["end_ns"] - item["begin_ns"]) / 1e6
             for item in response["state"]["trace"]
             if "begin_ns" in item and "end_ns" in item}
    explained = spans["prefill"] + spans["decode"]
    return {"model_ms": explained,
            "handoff_attach_ms": float(response["decode"].get("attach_rpc_ms", 0)),
            "pre_model_and_queue_ms": row["client_ms"] - explained}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    trial_rows, raw_lookup = [], {}
    raw_by_mode = {mode: [] for mode in MODES}
    for mode in MODES:
        for topology in TOPOLOGIES:
            path = args.root / "N6" / "raw_v2" / mode / topology / "result.json"
            raw = json.loads(path.read_text())
            raw_by_mode[mode].append(raw)
            raw_lookup[(mode, topology)] = raw
            for cell in raw["cells"]:
                members = [record for record in raw["records"]
                           if record["workload"] == cell["workload"] and
                           record["load"] == cell["load"] and
                           record["trial"] == cell["trial"]]
                durations = [record["client_ms"] for record in members]
                spans = [split_spans_ms(record) for record in members]
                split = topology != "unified"
                window_s = cell["observation_window_ms"] / 1000
                trial_rows.append({
                    "mode": mode, "topology": topology,
                    "workload": cell["workload"], "load": cell["load"],
                    "trial": cell["trial"], "trial_id": cell["trial_id"],
                    "requests": len(members),
                    "completed": sum(bool(record["ok"]) for record in members),
                    "observation_window_ms": cell["observation_window_ms"],
                    "arrival_train_ms": cell["arrival_train_ms"],
                    "offered_interval_ms": cell["offered_interval_ms"],
                    "throughput_requests_per_s": sum(bool(record["ok"])
                                                      for record in members) / window_s,
                    "output_tokens_per_s": sum(len(record["tokens"])
                                                for record in members) / window_s,
                    "p50_e2e_ms": statistics.median(durations),
                    "p95_e2e_ms": percentile(durations, 95),
                    "peak_client_inflight": overlap_peak(members),
                    "p50_model_path_ms": statistics.median(x["model_ms"] for x in spans),
                    "p50_pre_model_and_queue_ms": statistics.median(
                        x["pre_model_and_queue_ms"] for x in spans),
                    "p50_attach_rpc_ms": statistics.median(
                        x["handoff_attach_ms"] for x in spans),
                    "actual_prompt_tokens": sorted(set(
                        (record["response"]["decode"]["prompt_tokens"] if split else
                         next((item for item in record["response"]["outputs"]
                               if item.get("request_id") == record["request_id"]),
                              record["response"]["outputs"][0])["semantic_prompt_tokens"])
                        for record in members)),
                    "actual_output_tokens": sorted(set(len(record["tokens"])
                                                       for record in members))})

    rows = []
    workloads = sorted({row["workload"] for row in trial_rows})
    for mode in MODES:
        for topology in TOPOLOGIES:
            for workload in workloads:
                for load in LOADS:
                    trials = [row for row in trial_rows if row["mode"] == mode and
                              row["topology"] == topology and
                              row["workload"] == workload and row["load"] == load]
                    throughputs = [row["throughput_requests_per_s"] for row in trials]
                    token_rates = [row["output_tokens_per_s"] for row in trials]
                    rows.append({"mode": mode, "topology": topology,
                        "workload": workload, "load": load,
                        "independent_trials": len(trials),
                        "requests_per_trial": sorted(set(row["requests"] for row in trials)),
                        "completed_requests": sum(row["completed"] for row in trials),
                        "throughput_requests_per_s": statistics.median(throughputs),
                        "throughput_requests_per_s_min": min(throughputs),
                        "throughput_requests_per_s_max": max(throughputs),
                        "output_tokens_per_s": statistics.median(token_rates),
                        "p50_e2e_ms": statistics.median(row["p50_e2e_ms"] for row in trials),
                        "p95_e2e_ms": statistics.median(row["p95_e2e_ms"] for row in trials),
                        "peak_client_inflight": max(row["peak_client_inflight"] for row in trials),
                        "p50_model_path_ms": statistics.median(
                            row["p50_model_path_ms"] for row in trials),
                        "p50_pre_model_and_queue_ms": statistics.median(
                            row["p50_pre_model_and_queue_ms"] for row in trials),
                        "p50_attach_rpc_ms": statistics.median(
                            row["p50_attach_rpc_ms"] for row in trials),
                        "actual_prompt_tokens": sorted({token for row in trials
                            for token in row["actual_prompt_tokens"]}),
                        "actual_output_tokens": sorted({token for row in trials
                            for token in row["actual_output_tokens"]})})

    best = []
    for mode in MODES:
        for workload in workloads:
            for load in LOADS:
                candidates = [row for row in rows if row["mode"] == mode and
                              row["workload"] == workload and row["load"] == load]
                winner = max(candidates, key=lambda row: row["throughput_requests_per_s"])
                best.append({key: winner[key] for key in (
                    "mode", "workload", "load", "topology",
                    "throughput_requests_per_s", "throughput_requests_per_s_min",
                    "throughput_requests_per_s_max", "p95_e2e_ms")})

    memory = []
    for (mode, topology), raw in raw_lookup.items():
        status = raw["status"]
        memory.append({"mode": mode, "topology": topology,
            "resident_mib": raw["resident_mib"], "workers": len(status),
            "all_device_budgets_valid": all(item["budget_invariant"] and
                item["device_memory"]["within_admission_limit"] for item in status),
            "total_worker_granted_slots": sum(
                item["pool"]["worker_granted_slots"] for item in status),
            "workspace_reserved_mib": sum(
                item["device_memory"]["workspace_reserved_bytes"]
                for item in status) / 2**20,
            "private_model_nonworkspace_mib": sum(
                item["device_memory"]["private_model_nonworkspace_bytes"]
                for item in status) / 2**20})

    result = {"schema": "pbe-v4-n6-topology-analysis-v2",
        "measurement_boundary": "first actual send through last terminal response in each cohort",
        "client_ttft_itl": "N/A; interface is non-streaming",
        "trial_boundary": "five independent observation windows per cell; five requests at low/medium and eleven at near-saturation",
        "queue_metric": "peak client inflight reconstructed within each trial window",
        "fixed_kv_total_slots": 128,
        "resource_fairness": {
            "fixed": ["GPU UUID", "model and dtype", "request/arrival seed",
                      "128 total granted KV slots", "device admission limit"],
            "process_overhead_accounting": "Additional role context, streams and workspace remain measured deployment cost.",
            "not_hard_partitioned": "CPU and CUDA streams share the host/device pool; MPS/MIG is disabled.",
            "claim_boundary": "Fixed-workload/fixed-KV/device-budget comparison, not hardware isolation."},
        "cells": rows, "trials": trial_rows, "best_observed": best, "memory": memory,
        "ok": len(rows) == 90 and len(trial_rows) == 450 and all(
            row["independent_trials"] == 5 and
            row["requests_per_trial"] == ([11] if row["load"] == "near_saturation" else [5]) and
            row["completed_requests"] == (55 if row["load"] == "near_saturation" else 25)
            for row in rows)}
    capacity_path = args.root / "N6" / "capacity" / "results.json"
    if capacity_path.exists():
        capacity = json.loads(capacity_path.read_text())
        result["fixed_physical_budget_capacity"] = {
            "hard_budget_mib": capacity["hard_budget_mib"],
            "safety_margin_mib": capacity["safety_margin_mib"],
            "private_last_success_blocks": capacity["summary"]["private"]["median_last_success_blocks"],
            "private_first_failure_blocks": capacity["summary"]["private"]["median_first_failure_blocks"],
            "shared_last_success_blocks": capacity["summary"]["shared"]["median_last_success_blocks"],
            "shared_first_failure_blocks": capacity["summary"]["shared"]["median_first_failure_blocks"],
            "max_request_accessed_blocks": max(row["measurement"]["prompt_blocks_accessed"]
                                                for row in capacity["records"]),
            "claim_boundary": capacity["capacity_claim_boundary"]}
    output = args.root / "N6" / "topology_analysis_v2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for mode in MODES:
        summary = {"schema": "pbe-v4-n6-topology-matrix-v2", "mode": mode,
            "independent_trials_per_cell": 5,
            "requests_per_trial": {"low": 5, "medium": 5, "near_saturation": 11},
            "fixed_total_kv_slots": 128, "arrival_seed": 20260914,
            "timing": "first send through last terminal response in each cohort",
            "results": raw_by_mode[mode], "ok": all(item["ok"] for item in raw_by_mode[mode])}
        (args.root / "N6" / f"results-v2-{mode}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")

    near = [row for row in rows if row["load"] == "near_saturation"]
    lines = ["# N6 topology report v2", "",
        "Formal samples use no online oracle or profiler. Client TTFT/ITL is N/A because the API is non-streaming.", "",
        "## Five-window fixed-total-KV comparison", "",
        "| mode | workload | best observed topology | median req/s | trial range | median p95 E2E ms |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for mode in MODES:
        for workload in workloads:
            candidates = [row for row in near if row["mode"] == mode and
                          row["workload"] == workload]
            winner = max(candidates, key=lambda row: row["throughput_requests_per_s"])
            lines.append(f"| {mode} | {workload} | {winner['topology']} | "
                         f"{winner['throughput_requests_per_s']:.4f} | "
                         f"{winner['throughput_requests_per_s_min']:.4f}–"
                         f"{winner['throughput_requests_per_s_max']:.4f} | "
                         f"{winner['p95_e2e_ms']:.2f} |")
    lines += ["", "Each cell has five independent observation windows. Low/medium windows contain five requests; near-saturation contains eleven requests at a non-zero 200 ms interval for a 2,000 ms arrival train followed by cohort drain. This is a finite sustained-overload measurement, not an asymptotic saturation claim.", "",
        "## Interpretation boundary", "",
        "Best-observed topology is conditional on this GPU, model, workload and seed. Added process context/stream/workspace cost remains measured; CPU and CUDA streams use shared pools, and MPS/MIG is disabled.", "",
        "## Fixed-physical-memory capacity control", "",
        "The historical 18,000 MiB ladder remains an external-allocation bracket, not maximum live-request or SLO capacity.", ""]
    (args.root / "TOPOLOGY_REPORT_V2.md").write_text("\n".join(lines))
    print(json.dumps({"ok": result["ok"], "cells": len(rows),
                      "trials": len(trial_rows), "output": str(output)}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
