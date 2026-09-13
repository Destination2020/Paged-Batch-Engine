#!/usr/bin/env python3
"""Probe private/shared P/D capacity with one frozen physical hard budget."""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from run_e4_shared_weight_ab import file_sha256, run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hard-budget-mib", type=int, default=18000)
    parser.add_argument("--safety-margin-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    args = parser.parse_args()
    config_root = json.loads((args.model_dir / "config.json").read_text())
    config = config_root.get("text_config") or config_root
    args.layers = config["num_hidden_layers"]
    args.kv_heads = config["num_key_value_heads"]
    args.head_size = config["hidden_size"] // config["num_attention_heads"]
    args.model_sha256 = file_sha256(args.model_bin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ladder = [64, 1024, 2048, 4096, 8192, 12288, 16384]
    pairs = [(mode, repetition) for repetition in range(args.repeats)
             for mode in ("private", "shared")]
    random.Random(20260913).shuffle(pairs)
    records = []
    commands = []
    order = 0
    for mode, repetition in pairs:
        prior_success_blocks = None
        for blocks in ladder:
            args.capacity_blocks = blocks
            command = {
                "runner": "run_e4_shared_weight_ab.run_trial",
                "experiment": "fixed_budget", "mode": mode,
                "repetition": repetition, "capacity_blocks": blocks,
                "hard_budget_mib": args.hard_budget_mib,
                "safety_margin_mib": args.safety_margin_mib,
            }
            commands.append(command)
            print(json.dumps({"event": "capacity_probe", "order": order, **command}), flush=True)
            row = run_trial(args, "fixed_budget", mode, repetition, order)
            row["hard_budget_mib"] = args.hard_budget_mib
            row["within_hard_budget"] = (
                row["startup_peak_mib"] + args.safety_margin_mib <= args.hard_budget_mib)
            row["previous_success_blocks"] = prior_success_blocks
            row["capacity_classification"] = ("success" if row["within_hard_budget"] and
                row["measurement"]["admitted"] >= 1 and
                row["measurement"]["all_decode_ok"] else "first_failure")
            row["access_boundary"] = {
                "allocated_blocks": blocks,
                "request_accessed_blocks": row["measurement"]["prompt_blocks_accessed"],
                "claim": "allocation bracket only; not maximum live-request concurrency",
            }
            records.append(row)
            order += 1
            if row["capacity_classification"] == "first_failure":
                break
            prior_success_blocks = blocks
    cells = {}
    for mode in ("private", "shared"):
        mode_rows = [row for row in records if row["mode"] == mode]
        brackets = []
        for repetition in range(args.repeats):
            rows = [row for row in mode_rows if row["repetition"] == repetition]
            success = [row for row in rows if row["capacity_classification"] == "success"]
            failure = [row for row in rows if row["capacity_classification"] == "first_failure"]
            brackets.append({"repetition": repetition,
                "last_success_blocks": success[-1]["kv_blocks"] if success else None,
                "last_success_peak_mib": success[-1]["startup_peak_mib"] if success else None,
                "first_failure_blocks": failure[0]["kv_blocks"] if failure else None,
                "first_failure_peak_mib": failure[0]["startup_peak_mib"] if failure else None,
                "failure_reason": "peak_plus_safety_margin_exceeds_hard_budget"
                    if failure else "ladder_ceiling",
            })
        last_success_values = [item["last_success_blocks"] for item in brackets
                               if item["last_success_blocks"] is not None]
        first_failure_values = [item["first_failure_blocks"] for item in brackets
                                if item["first_failure_blocks"] is not None]
        cells[mode] = {
            "repeats": len(brackets), "brackets": brackets,
            "median_last_success_blocks": statistics.median(last_success_values)
                if last_success_values else None,
            "median_first_failure_blocks": statistics.median(first_failure_values)
                if first_failure_values else None,
            "right_censored_at_ladder_ceiling": not first_failure_values,
        }
    result = {"schema": "pbe-v4-b3-common-budget-capacity-v1",
              "hard_budget_mib": args.hard_budget_mib,
              "safety_margin_mib": args.safety_margin_mib,
              "same_probe_ladder": ladder, "random_seed": 20260913,
              "records": records, "summary": cells,
              "capacity_claim_boundary":
                  "Brackets allocatable external KV under the hard budget; request-level access is reported separately and does not claim maximum SLO concurrency.",
              "ok": all(cell["repeats"] == args.repeats for cell in cells.values()) and
                    all(row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256
                        for row in records)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output.parent / "commands.json").write_text(
        json.dumps({"replay_argv": [sys.executable, *sys.argv],
                    "random_seed": 20260913, "probes": commands},
                   indent=2, sort_keys=True) + "\n")
    (args.output.parent / "memory_timeline.jsonl").write_text("".join(
        json.dumps({"trial": row["trial"], **sample}, sort_keys=True) + "\n"
        for row in records for sample in row["memory_timeline"]))
    print(json.dumps({"event": "complete", "ok": result["ok"],
                      "records": len(records), "summary": cells}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
