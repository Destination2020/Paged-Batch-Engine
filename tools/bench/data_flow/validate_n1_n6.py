#!/usr/bin/env python3
"""Evidence-derived acceptance gate for V4 section 21 (N1--N6)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODES = ("private", "shared")
TOPOLOGIES = ("unified", "1P1D", "1P2D", "2P1D", "2P2D")
WORKLOADS = ("long_input_short_output", "short_input_long_output", "mixed")
LOADS = ("low", "medium", "near_saturation")


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(root, manifest):
    if not manifest.exists():
        return False
    data = load(manifest)
    entries = data["sources"] + data["binaries"] + data["inputs"]
    return bool(entries) and all((root / item["path"]).is_file() and
                                 (root / item["path"]).stat().st_size == item["bytes"] and
                                 sha256(root / item["path"]) == item["sha256"]
                                 for item in entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path)
    args = parser.parse_args()
    n1 = load(args.evidence / "N1/oracle_free_e4/results.json")
    n2 = load(args.evidence / "N2/latency_breakdown.json")
    profiler = n2.get("diagnostic_profiler", {})
    profiler_trace = args.root.resolve() / profiler.get("trace", "missing")
    profiler_stats = args.root.resolve() / profiler.get("stats", "missing")
    n3 = load(args.evidence / "N3/result.json")
    n4 = load(args.evidence / "N4/result.json")
    n5 = load(args.evidence / "N5/multi_pd_reservation_cancel_v2/result.json")
    analysis = load(args.evidence / "N6/topology_analysis_v2.json")
    capacity_path = args.evidence / "N6/capacity/results.json"
    capacity = load(capacity_path) if capacity_path.exists() else {"ok": False}

    all_records = []
    raw_results = []
    for mode in MODES:
        for topology in TOPOLOGIES:
            raw = load(args.evidence / "N6/raw_v2" / mode / topology / "result.json")
            raw_results.append(raw)
            all_records.extend((mode, topology, item) for item in raw["records"])

    split = [(mode, topology, row) for mode, topology, row in all_records
             if topology != "unified"]
    cell_keys = {(mode, topology, row["workload"], row["load"])
                 for mode, topology, row in all_records}
    trial_keys = {(mode, topology, row["workload"], row["load"], row["trial"])
                  for mode, topology, row in all_records}
    role_contract = all(
        row["response"]["ok"] and row["response"]["state"]["terminal"] == "finished" and
        row["response"]["state"]["prefill_worker"] !=
        row["response"]["state"]["decode_worker"] and
        row["response"]["prefill"]["worker_pid"] !=
        row["response"]["decode"]["worker_pid"] and
        row["response"]["prefill"]["consumed_reservation_id"] > 0 and
        row["response"]["decode"]["consumed_reservation_id"] > 0 and
        row["response"]["decode"]["attach_grant"] > 0 and
        row["response"]["decode"]["provider_incarnation"] ==
        row["response"]["prefill"]["provider_incarnation"] and
        row["response"]["decode"]["consumer_incarnation"] > 0 and
        row["response"]["decode"]["metadata_allocation"]["generation"] > 0 and
        bool(row["response"]["decode"]["authorized_pages"])
        for _, _, row in split)
    no_prefix_recompute = all(
        row["response"]["decode"]["actual_computed_prompt_tokens"] == 0 and
        row["response"]["decode"]["scheduled_prefill_tokens"] == 0 and
        row["response"]["decode"]["prefill_tokens_saved"] ==
        row["response"]["decode"]["prompt_tokens"]
        for _, _, row in split)
    statuses = [item for raw in raw_results for item in raw["status"]]
    budget = all(item["budget_invariant"] and
                 item["device_memory"]["within_admission_limit"] and
                 item["pd"]["active_handoffs"] == 0 and
                 item["pd"]["active_reservations"] == 0 and
                 item["budget"]["kv"]["used"] == 0
                 for item in statuses)
    fixed_slots = all(sum(item["pool"]["worker_granted_slots"]
                          for item in raw["status"]) == 128
                      for raw in raw_results)
    topology_workers = all(raw["all_workers_executed"] for raw in raw_results)
    unified_batching = all(
        raw.get("unified_batch_capability_ok") and
        raw.get("unified_batch_capability_probe", {}).get("ok") and
        raw["unified_batch_capability_probe"].get("admitted") == 5
        for raw in raw_results if raw["topology"] == "unified")
    independent_windows = all(
        raw.get("five_independent_windows_per_cell") and
        raw.get("finite_sustained_overload") and len(raw["cells"]) == 45 and
        len(raw["records"]) == 315 and
        all(cell["requests"] == (11 if cell["load"] == "near_saturation" else 5)
            and cell["completed"] == cell["requests"] and
            (cell["offered_interval_ms"] == 200 and cell["arrival_train_ms"] >= 2000
             if cell["load"] == "near_saturation" else
             cell["offered_interval_ms"] > 0)
            for cell in raw["cells"])
        for raw in raw_results)

    edges = {(item["topology"], item["edge"]) for item in n5["paths"]}
    required_edges = {("1P1D", "p0->d0"), ("1P2D", "p0->d0"),
                      ("1P2D", "p0->d1"), ("2P1D", "p0->d0"),
                      ("2P1D", "p1->d0"), ("2P2D", "p0->d0"),
                      ("2P2D", "p0->d1"), ("2P2D", "p1->d0"),
                      ("2P2D", "p1->d1")}
    capacity_rows = capacity.get("records", [])
    capacity_valid = (capacity.get("ok", False) and capacity.get("hard_budget_mib") == 18000 and
        capacity.get("safety_margin_mib") == 256 and
        capacity.get("same_probe_ladder") == [64, 1024, 2048, 4096, 8192, 12288, 16384] and
        all(row["measurement"]["prompt_blocks_accessed"] > 0 and
            row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256
            for row in capacity_rows) and
        all(capacity["summary"][mode]["repeats"] == 5 for mode in MODES))

    log_root = args.evidence / "logs"
    cpp_log = (log_root / "cpp_affected_tests.log").read_text()
    python_log = (log_root / "python_affected_tests.log").read_text()
    sanitizer_path = args.evidence / "N5/compute_sanitizer_reservation_v2/result.json"
    sanitizer = load(sanitizer_path) if sanitizer_path.exists() else {"ok": False}
    numerical_path = args.root / n1["numeric_gate"]["path"]
    numerical = load(numerical_path)
    current_numerical = load(args.evidence / "N6/numerical_regression_v2/validation.json")

    checks = {
        "N1_oracle_free_five_per_arm": n1["ok"] and
            n1["checks"]["formal_oracle_steps_zero"] and
            n1["checks"]["five_trials_per_arm"],
        "N1_external_numeric_same_history_gate":
            numerical["event"] == "validated_20_fixtures" and
            numerical["cases"] == 20 and numerical["budget_invariant"] and
            numerical["numerical_contract"]["same_history_required"],
        "N6_current_binary_20_case_numeric_gate":
            current_numerical["event"] == "validated_20_fixtures" and
            current_numerical["cases"] == 20 and
            current_numerical["budget_invariant"] and
            current_numerical["numerical_contract"] ==
                numerical["numerical_contract"],
        "N2_per_request_critical_path_before_aggregation": n2["ok"] and
            all(name in n2 for name in ("B2", "E3", "E4")) and
            all(n2["checks"][name] for name in (
                "all_request_breakdowns_conserve_e2e",
                "b2_reconciles_50_paired_requests_before_aggregation",
                "e4_oracle_free_5_pairs",
                "residuals_not_presented_as_leaf_causes")) and
            "unresolved" in n2["E4"]["classification"],
        "N2_independent_real_profiler_excluded_from_formal_latency":
            n2["checks"]["independent_real_profiler_trace_excluded_from_formal_latency"] and
            not profiler["formal_latency_sample"] and
            profiler_trace.is_file() and profiler_stats.is_file() and
            sha256(profiler_trace) == profiler["trace_sha256"] and
            sha256(profiler_stats) == profiler["stats_sha256"],
        "N3_static_registry_and_1P1D": n3["ok"] and all(n3["checks"].values()),
        "N4_dynamic_authorization_and_1P2D": n4["ok"] and all(n4["checks"].values()),
        "N3_N5_all_production_edges": n5["ok"] and required_edges <= edges,
        "N3_N5_dynamic_authorization_and_lifecycle": all(n5["checks"].values()),
        "N5_compute_sanitizer_real_1P1D": sanitizer.get("ok", False) and
            sanitizer.get("independent_role_pids", False) and
            sanitizer.get("zero_prefix_recompute", False),
        "N6_all_90_cells_450_independent_trials_3150_requests":
            len(cell_keys) == 90 and len(trial_keys) == 450 and
            len(all_records) == 3150 and independent_windows,
        "N6_oracle_and_diagnostics_absent": all(
            raw["oracle_steps_in_timing"] == 0 and not raw["diagnostics_in_timing"]
            for raw in raw_results),
        "N6_outputs_complete": all(row["ok"] and bool(row["tokens"])
                                   for _, _, row in all_records),
        "N6_real_roles_and_provider_generation": role_contract and topology_workers,
        "N6_unified_production_batch_capability": unified_batching,
        "N6_decode_does_not_recompute_prefix": no_prefix_recompute,
        "N6_fixed_total_kv_and_device_budget": fixed_slots and budget,
        "N6_resource_fairness_boundary_disclosed":
            "resource_fairness" in analysis and
            "MPS/MIG is disabled" in
                analysis["resource_fairness"]["not_hard_partitioned"] and
            "not hardware isolation" in
                analysis["resource_fairness"]["claim_boundary"],
        "N6_window_throughput_and_queue_recomputed": analysis["ok"] and
            len(analysis["cells"]) == 90 and len(analysis["trials"]) == 450 and
            all(row["independent_trials"] == 5 and
                row["completed_requests"] ==
                    (55 if row["load"] == "near_saturation" else 25)
                for row in analysis["cells"]) and
            all(row["peak_client_inflight"] >= 1 and
                abs(row["throughput_requests_per_s"] - row["completed"] /
                    (row["observation_window_ms"] / 1000)) < 1e-12
                for row in analysis["trials"]),
        "N6_common_hard_budget_capacity_ladder": capacity_valid,
        "affected_cpp_tests": "[  PASSED  ] 71 tests." in cpp_log and
            "[  SKIPPED ] 2 tests" in cpp_log,
        "affected_python_tests": "8 passed" in python_log,
        "completion_manifest_matches_current_files": bool(args.completion_manifest) and
            check_manifest(args.root.resolve(), args.completion_manifest),
    }
    result = {"schema": "pbe-v4-n1-n6-validation-v1", "checks": checks,
              "counts": {"topology_cells": len(cell_keys),
                         "topology_independent_trials": len(trial_keys),
                         "topology_requests": len(all_records),
                         "split_requests": len(split),
                         "n5_real_paths": len(edges),
                         "capacity_probe_records": len(capacity_rows)},
              "numeric_contract": n1["numeric_gate"]["contract"],
              "capacity_claim_boundary": capacity.get("capacity_claim_boundary"),
              "ok": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "checks": checks}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
