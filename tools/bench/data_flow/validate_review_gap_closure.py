#!/usr/bin/env python3
"""Executable final gate for the E2/E3/M9 review-gap closure."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    placement = load(root / "E2_final/runtime_state_validation.json")
    lane_cuda = load(root / "M9/transfer_scheduler_lane_final/validation.json")
    lane_model = load(root / "M9/model_transfer_scheduler_lane_final_gpu0_v2/results.json")
    replica_count = load(root / "M9/persistent_deployment_final_gpu0_v2/results.json")
    deployment = load(root / "M9/persistent_pd_deployment_final_gpu0_v4/results.json")
    deployment_gate = load(root / "M9/persistent_pd_deployment_final_gpu0_v4/validation.json")
    focused_log = (root / "E3/dependency_eviction_integration_final2.log").read_text()

    statuses = lane_model["worker_status"]
    recovery = [statuses[mode]["recovery_dependency"] for mode in statuses]
    transfer = [statuses[mode]["transfer_scheduler"] for mode in statuses]
    e3_ok = ("[  PASSED  ] 19 tests." in focused_log and
             all(x["committed_demotions"] > 0 and x["restores"] > 0 and
                 x["publish_rejections"] == 0 for x in recovery))
    lane_model_ok = (lane_model["ok"] and lane_model["repeats_each"] == 5 and
                     all(row["background_pages_submitted"] == 2 and
                         row["semantic_matched_tokens"] > 0 for row in lane_model["records"]) and
                     statuses["lanes_on"]["transfer_scheduler"]["max_d2h_active"] == 1 and
                     statuses["lanes_off"]["transfer_scheduler"]["max_d2h_active"] == 2 and
                     all(x["physical_submissions"] > 0 for x in transfer))
    pd_rows = [row for row in deployment["records"]
               if row["mode"] == "persistent_prefill_decode"]
    deployment_ok = (deployment["ok"] and deployment_gate["ok"] and
                     deployment["cold_start_excluded"] and
                     deployment["all_three_model_processes_concurrently_resident"] and
                     deployment["same_owned_kv_slots_per_topology"] == 64 and
                     deployment["completed_requests"] == 10 and len(pd_rows) == 5 and
                     all(row["prefill_pid"] != row["decode_pid"] and
                         row["handoff_valid"] and row["grants_are_distinct"] and
                         row["prefill_tokens_saved"] == row["prompt_tokens"] and
                         row["decode_actual_computed_prompt_tokens"] == 0 and
                         row["decode_scheduled_prefill_tokens"] == 0 and
                         row["output_matches_oracle"] and row["handoff_released"]
                         for row in pd_rows))
    output = {
        "schema": "pbe.v4.review-gap-closure.v1",
        "ok": placement["ok"] and e3_ok and lane_cuda["ok"] and lane_model_ok and deployment_ok,
        "e2_actual_runtime_state": placement,
        "e3_actual_kv_eviction_and_recovery": {
            "ok": e3_ok, "focused_tests": 19, "serving_workers": recovery},
        "m9_production_scheduler_lane": {
            "ok": lane_cuda["ok"] and lane_model_ok,
            "cuda_full_payload_validation": lane_cuda,
            "model_repeats_each": lane_model["repeats_each"],
            "model_result_is_negative": lane_model["summaries"]["lanes_on"]["median_client_ms"] >
                                        lane_model["summaries"]["lanes_off"]["median_client_ms"]},
        "m9_full_replica_count_cost": {
            "ok": replica_count["ok"],
            "counts_as_prefill_decode_handoff_acceptance": False,
            "scope": "one versus two complete Language replicas",
            "summaries": replica_count["summaries"]},
        "m9_warm_persistent_prefill_decode_deployment": {
            "ok": deployment_ok,
            "cold_start_excluded": deployment["cold_start_excluded"],
            "all_three_model_processes_concurrently_resident":
                deployment["all_three_model_processes_concurrently_resident"],
            "completed_requests": deployment["completed_requests"],
            "gate": deployment_gate,
            "summaries": deployment["summaries"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": output["ok"], "output": str(args.output)}, sort_keys=True))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
