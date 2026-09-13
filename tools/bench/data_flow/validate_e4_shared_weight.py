#!/usr/bin/env python3
"""Executable E4 acceptance gate over raw real-process evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-repeats", type=int, default=5)
    args = parser.parse_args()
    result = json.loads((args.evidence / "results.json").read_text())
    manifest = json.loads((args.evidence / "weight_manifest.json").read_text())
    rows = result["records"]
    shared = [row for row in rows if row["mode"] == "shared"]
    expected_trials = args.required_repeats * 4
    identities = {(row["prefill_status"]["weights"]["owner_incarnation"],
                   row["prefill_status"]["weights"]["allocation_id"],
                   row["prefill_status"]["weights"]["generation"])
                  for row in shared}
    checks = {
        "runner_gate": result.get("ok") is True,
        "trial_count": len(rows) == expected_trials,
        "raw_trial_files": len(list((args.evidence / "trials").glob("*/result.json"))) == expected_trials,
        "five_repeats_each": all(sum(row["experiment"] == experiment and
            row["mode"] == mode for row in rows) == args.required_repeats
            for experiment in ("fixed_kv", "fixed_budget") for mode in ("private", "shared")),
        "model_content_identity": sha256(args.model) == result["model_sha256"] ==
            manifest["model_sha256"],
        "layout_identity": result["layout_identity"] == manifest["layout_identity"] ==
            "pbe-qwen2-bf16-v1",
        "manifest_exact_coverage": manifest["validation"]["file_exactly_covered"] and
            manifest["validation"]["overlap_count_excluding_declared_aliases"] == 0 and
            manifest["validation"]["declared_alias_count"] == 1 and
            manifest["validation"]["physical_slab_allocations"] == 1,
        "fresh_owner_each_shared_trial": len(identities) == len(shared),
        "single_physical_upload_each": all(row["service_weight_stats"]["upload_count"] == 1 and
            row["service_weight_stats"]["allocation_count"] == 1 and
            row["service_weight_stats"]["physical_bytes"] == manifest["file_bytes"]
            for row in shared),
        "two_independent_import_leases": all(
            row["prefill_status"]["weights"]["import_lease_id"] !=
            row["decode_status"]["weights"]["import_lease_id"] and
            row["service_weight_stats"]["active_leases"] == 2 for row in shared),
        "same_generation_and_allocation": all(
            row["prefill_status"]["weights"][key] == row["decode_status"]["weights"][key]
            for row in shared for key in ("owner_incarnation", "allocation_id", "generation")),
        "real_attention_views": all(row[role]["weights"]["attention_views_bound"] and
            row[role]["weights"]["embedding_view_bound"] and
            row[role]["weights"]["output_view_bound"] and
            row[role]["weights"]["bound_tensor_views"] ==
            manifest["validation"]["immutable_operator_tensor_views"]
            for row in shared for role in ("prefill_status", "decode_status")),
        "no_private_full_weight_copy": all(row[role]["device_memory"][
            "model_process_allocation_bytes"] < manifest["file_bytes"] // 2
            for row in shared for role in ("prefill_status", "decode_status")),
        "real_pd_handoff": all(row["measurement"].get("independent_role_pids", True) and
            row["measurement"].get("handoff_valid", True) and
            row["measurement"].get("decode_actual_computed_prompt_tokens", 0) == 0 and
            row["measurement"].get("decode_scheduled_prefill_tokens", 0) == 0 and
            row["measurement"].get("output_matches_prefill_oracle", True) for row in rows),
        "survivor_and_new_importer": all(row["lifecycle"]["decode_survived_prefill_exit"] and
            row["lifecycle"]["new_importer_joined"] for row in rows),
        "normal_lease_recovery": all(row["weight_after_workers"]["active_leases"] == 0
                                     for row in shared),
        "normal_kv_grant_recovery": all(row["ipc_after_workers"]["free_slots"] ==
            row["ipc_after_workers"]["total_slots"] and
            row["ipc_after_workers"]["active_grants"] == 0 for row in rows),
        "physical_budget_capacity": result["checks"]["capacity_kv_increment_within_weight_savings"] and
            result["checks"]["fixed_budget_observed_not_higher"],
        "capacity_pages_really_accessed": result["checks"]["shared_capacity_accesses_new_pages"],
        "capacity_admission_improves": result["checks"]["shared_capacity_admits_more"],
        "device_memory_recovered": result["checks"]["cleanup_near_baseline"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"schema": "pbe-e4-acceptance-v1", "ok": all(checks.values()),
              "checks": checks, "trials": len(rows), "shared_trials": len(shared),
              "model_sha256": result["model_sha256"],
              "physical_accounting": result["physical_accounting"],
              "summary": result["summary"]}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    lifecycle = []
    memory = []
    for row in rows:
        lifecycle.append({"trial": row["trial"], "mode": row["mode"],
                          "experiment": row["experiment"], **row["lifecycle"],
                          "ipc_after_workers": row["ipc_after_workers"],
                          "weight_after_workers": row.get("weight_after_workers")})
        for sample in row["memory_timeline"]:
            memory.append({"trial": row["trial"], "mode": row["mode"],
                           "experiment": row["experiment"], **sample})
    (args.evidence / "lifecycle_trace.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True)+"\n" for item in lifecycle))
    (args.evidence / "memory_timeline.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True)+"\n" for item in memory))
    print(json.dumps(output, sort_keys=True))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
