#!/usr/bin/env python3
"""Derive N3/N4 acceptance views from the immutable N5 real-process record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    faults = {item["case"]: item for item in source["faults"]}
    paths = source["paths"]
    registry = source["registry"]

    one = [item for item in paths if item["topology"] == "1P1D"]
    n3_checks = {
        "production_coordinator_path": len(one) == 1 and one[0]["result"]["ok"],
        "independent_prefill_decode_pids": len(one) == 1 and
            one[0]["result"]["prefill"]["worker_pid"] !=
            one[0]["result"]["decode"]["worker_pid"],
        "static_ready_registry_contract": all(item["state"] == "ready" and
            item["incarnation"] > 0 and item["gpu_uuid"] and item["dtype"] == "bf16"
            for item in registry),
        "worker_owned_ordered_reservations":
            source["checks"].get("ordered_worker_reservations_consumed", False) and
            source["checks"].get("atomic_reservation_contention_and_recovery", False),
        "active_decode_deadline_and_external_cancel":
            source["checks"].get("external_cancel_and_deadline_reach_active_decode", False),
        "duplicate_terminal": "duplicate_generation" in faults,
        "resource_reclaimed": source["post_worker_ipc"]["active_grants"] == 0,
    }
    n3 = {"schema": "pbe-v4-n3-derived-acceptance-v1", "source": str(args.input),
          "checks": n3_checks, "path": one, "registry": registry,
          "faults": [faults[name] for name in
                     ("atomic_reservation_contention", "external_cancel_during_decode",
                      "absolute_deadline_during_decode", "duplicate_generation")],
          "ok": all(n3_checks.values())}

    one_two = [item for item in paths if item["topology"] == "1P2D"]
    decoders = {item["result"]["decode"]["worker_pid"] for item in one_two}
    n4_checks = {
        "production_1P2D_both_decoders": len(one_two) == 2 and len(decoders) == 2,
        "request_level_provider_generation_grants": all(
            item["result"]["decode"]["attach_grant"] > 0 and
            item["result"]["decode"]["provider_incarnation"] ==
            item["result"]["prefill"]["provider_incarnation"]
            for item in one_two),
        "decode_zero_prefix_recompute": all(
            item["result"]["decode"]["actual_computed_prompt_tokens"] == 0
            for item in one_two),
        "fanout_cow_branch_isolation": "fanout_cow_and_branch_isolation" in faults,
        "provider_exit_and_new_decode_generation": all(name in faults for name in
            ("provider_normal_exit_after_attach", "new_decode_incarnation")),
        "stale_provider_rejected": "provider_generation_authorization" in faults,
        "resource_reclaimed": source["post_worker_ipc"]["active_grants"] == 0,
    }
    n4_names = ("provider_generation_authorization", "fanout_cow_and_branch_isolation",
                "provider_normal_exit_after_attach", "new_decode_incarnation")
    n4 = {"schema": "pbe-v4-n4-derived-acceptance-v1", "source": str(args.input),
          "checks": n4_checks, "paths": one_two,
          "faults": [faults[name] for name in n4_names],
          "ok": all(n4_checks.values())}
    for name, value in (("N3", n3), ("N4", n4)):
        output = args.output_root / name / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"N3": n3["ok"], "N4": n4["ok"]}, sort_keys=True))
    return 0 if n3["ok"] and n4["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
