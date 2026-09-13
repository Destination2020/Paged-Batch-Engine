#!/usr/bin/env python3
"""Executable acceptance gate for persistent multimodal P/D deployment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run_dir / "results.json").read_text())
    ipc = (args.run_dir / "ipc-after.log").read_text()
    data = (args.run_dir / "data-after.log").read_text()
    pd = [row for row in result["records"]
          if row["mode"] == "persistent_prefill_decode"]
    checks = {
        "five_repetitions_each": len(result["records"]) == 10 and len(pd) == 5,
        "persistent_and_warm": result["cold_start_excluded"] and
            result["all_three_model_processes_concurrently_resident"],
        "same_workload_and_owned_kv_budget": result["same_multimodal_bundle_and_model"] and
            result["same_owned_kv_slots_per_topology"] == 64,
        "independent_prefill_decode_pid": all(
            row["prefill_pid"] != row["decode_pid"] for row in pd),
        "valid_handoff_and_distinct_grants": all(
            row["handoff_valid"] and row["grants_are_distinct"] and
            row["handoff_content"] for row in pd),
        "decode_did_not_recompute_prompt": all(
            row["prefill_tokens_saved"] == row["prompt_tokens"] and
            row["decode_actual_computed_prompt_tokens"] == 0 and
            row["decode_scheduled_prefill_tokens"] == 0 for row in pd),
        "outputs_match_warmed_unified_oracle": all(
            row["output_matches_oracle"] for row in result["records"]),
        "handoffs_released": all(row["handoff_released"] for row in pd) and
            result["worker_status"]["persistent_prefill_decode"][0]["pd"]["active_handoffs"] == 0,
        "worker_kv_requests_recovered_to_model_baseline": all(
            status["pd"]["kv_active_requests"] ==
            status["pd"]["kv_request_baseline"] and
            status["pd"]["serving_active_requests"] == 0
            for statuses in result["worker_status"].values() for status in statuses),
        "ipc_grants_and_leases_recovered":
            "total_slots=128 free_slots=128 active_grants=0" in ipc and "leases=0" in data,
    }
    output = {"schema": "pbe.v4.persistent-multimodal-pd-gate.v1",
              "ok": result["ok"] and all(checks.values()), "checks": checks,
              "prefill_pids": sorted(set(row["prefill_pid"] for row in pd)),
              "decode_pids": sorted(set(row["decode_pid"] for row in pd)),
              "handoffs": len(pd), "prompt_tokens_saved_each":
                  sorted(set(row["prefill_tokens_saved"] for row in pd))}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
