#!/usr/bin/env python3
"""Independent consistency gate for V4 section 20 B1-B6 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lines(path):
    return [line for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path,
                        default=Path("docs/data_flow_evidence/v4/performance_characterization"))
    args = parser.parse_args()
    root, out = args.root, args.evidence
    b2, b3 = load(out/"B2/results.json"), load(out/"B3/results.json")
    b4, b5 = load(out/"B4/results.json"), load(out/"B5/support_matrix.json")
    result, manifest = load(out/"results.json"), load(out/"manifest.json")
    checks = {}

    checks["b1_inventory_and_protocol"] = (
        len(load(out/"evidence_inventory.json")["items"]) == 10 and
        load(out/"protocol.json")["frozen_before_supplemental_runs"] is True)
    records = b2["records"]
    cell_arms = Counter((row["cell"], row["arm"]) for row in records)
    checks["b2_40_trials_5_each"] = len(records) == 40 and set(cell_arms.values()) == {5}
    checks["b2_400_requests"] = sum(len(row["requests"]) for row in records) == 400
    checks["b2_runtime_and_numeric_gates"] = b2["ok"] and all(
        row["ok"] and row["identity_ok"] and
        row["cache_behavior_ok"] and row["resource_recovered"] for row in records)
    checks["b2_paired_numerics"] = (
        b2["paired_cross_arm_numerics"]["ok"] and
        b2["paired_cross_arm_numerics"]["comparisons"] == 250 and
        b2["paired_cross_arm_numerics"]["accepted"] == 250 and
        load(out/"B2_divergence_audit/results.json")["ok"] and
        len(b2["within_arm_repeatability_failures"]) == 2)
    ids = defaultdict(set)
    for row in records:
        ids[row["cell"]].add(tuple(row["canonical_media_identities"]))
    checks["b2_identity_independent_of_cache_policy"] = (
        b2["identity_across_cache_policies_ok"] and all(len(value) == 1 for value in ids.values()))
    checks["b2_non_streaming_boundary_honest"] = all(
        request["actual_send_ns"] is not None and request["terminal_ns"] is not None and
        request["terminal_ns"] > request["actual_send_ns"] and
        request["first_visible_token_ns"] is None and
        "non-streaming" in request["timing_boundary"]
        for row in records for request in row["requests"])

    ladder = load(out/"protocol.json")["b3"]["probe_blocks"]
    checks["b3_same_budget_and_ladder"] = (
        b3["same_probe_ladder"] == ladder and b3["hard_budget_mib"] == 18000)
    checks["b3_five_brackets_each"] = all(
        cell["repeats"] == 5 and len(cell["brackets"]) == 5
        for cell in b3["summary"].values())
    checks["b3_classification_consistent"] = all(
        (row["capacity_classification"] == "success") ==
        (row["startup_peak_mib"] + b3["safety_margin_mib"] <= b3["hard_budget_mib"] and
         row["measurement"]["admitted"] >= 1 and row["measurement"]["all_decode_ok"])
        for row in b3["records"])
    checks["b3_resource_recovery"] = b3["ok"] and all(
        row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256 for row in b3["records"])
    checks["b3_claim_is_bounded"] = (
        "not maximum" in b3["capacity_claim_boundary"] and
        result["phases"]["B3"]["slo_goodput"].startswith("N/A"))

    checks["b4_existing_trials_recomputed"] = (
        b4["ok"] and len(b4["placement"]["trials"]) == 15 and
        len(b4["recovery"]["trials"]) == 15 and
        all("completed_requests / dispatch-to-last-completion" in row["throughput_definition"]
            for row in b4["placement"]["trials"]))
    dependency = b4["recovery"]["policies"]["dependency_host_checkpoint"]
    checks["b4_dependency_restore_accounting"] = (
        dependency["host_demoted_blocks"] == dependency["host_restored_blocks"] and
        dependency["restore_failures"] == 0)
    checks["b5_explicitly_unaccepted"] = (
        b5["accepted"] is False and len(b5["blocking_contracts"]) >= 4 and
        all("unsupported" in b5["matrix"][topology]
            for topology in ("1P2D", "2P1D", "2P2D")))

    with (out/"summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    formula_ok = True
    for row in summary:
        base, candidate = float(row["baseline_value"]), float(row["candidate_value"])
        expected = (candidate-base)/base*100 if base else None
        actual = None if row["candidate_change_percent"] == "" else float(row["candidate_change_percent"])
        formula_ok &= ((expected is None and actual is None) or abs(expected-actual) < 1e-9)
        improvement = actual if row["higher_is_better"] == "True" else -actual
        formula_ok &= abs(improvement-float(row["improvement_percent"])) < 1e-9
    checks["b6_percentages_recomputed"] = formula_ok and len(summary) >= 10
    checks["b6_raw_counts"] = (
        len(lines(out/"raw/requests.jsonl")) == result["raw_counts"]["requests"] and
        len(lines(out/"raw/trials.jsonl")) == result["raw_counts"]["trials"] and
        len(lines(out/"raw/resources.jsonl")) == result["raw_counts"]["resource_samples"])
    required = ["PERFORMANCE_REPORT.md", "RESUME_METRICS.md", "summary.csv", "results.json",
                "manifest.json", "raw/requests.jsonl", "raw/trials.jsonl", "raw/resources.jsonl"]
    plot_names = ["b2_reuse_throughput", "b3_fixed_kv_memory",
                  "b3_capacity_bracket", "b5_topology_support"]
    checks["b6_deliverables"] = (all((out/name).is_file() for name in required) and
        all((out/"plots"/f"{name}.{extension}").is_file()
            for name in plot_names for extension in ("svg", "png")))
    checks["b6_manifest_source_hashes"] = all(
        (root/item["path"]).is_file() and sha256(root/item["path"]) == item["sha256"]
        for item in manifest["sources"])
    gpu_apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    checks["b6_gpu_processes_released"] = not gpu_apps
    checks["overall_exactly_5_of_6"] = (
        result["ok"] and result["accepted_phases"] == 5 and
        result["phases"]["B5"]["accepted"] is False)

    output = {"schema": "pbe-v4-b1-b6-validation-v1", "ok": all(checks.values()),
              "checks": checks, "accepted_phases": result["accepted_phases"],
              "total_phases": 6}
    (out/"validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
