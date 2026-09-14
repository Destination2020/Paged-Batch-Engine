#!/usr/bin/env python3
"""N1 paired 1P1D fixed-KV trial with the numerical oracle out of timing."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

from run_e4_shared_weight_ab import LAYOUT, file_sha256, run_trial


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--numeric-gate", type=Path,
        default=Path("docs/data_flow_evidence/v4/M5_numerics_final/validation_final.json"))
    parser.add_argument("--reuse-records", action="store_true")
    cli = parser.parse_args()
    config_root = json.loads((cli.model_dir / "config.json").read_text())
    config = config_root.get("text_config") or config_root
    output = cli.output_dir / "results.json"
    args = SimpleNamespace(**vars(cli), output=output,
        layers=config["num_hidden_layers"],
        kv_heads=config["num_key_value_heads"],
        head_size=config["hidden_size"] // config["num_attention_heads"],
        model_sha256=file_sha256(cli.model_bin), compute_sanitizer=False)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    order = [(mode, repetition) for repetition in range(cli.repeats)
             for mode in ("private", "shared")]
    random.Random(20260914).shuffle(order)
    records = []
    if cli.reuse_records and output.exists():
        records = json.loads(output.read_text())["records"]
    else:
        for index, (mode, repetition) in enumerate(order):
            print(json.dumps({"event": "trial_start", "index": index,
                              "mode": mode, "repetition": repetition}), flush=True)
            records.append(run_trial(args, "fixed_kv", mode, repetition, index))

    summary = {}
    for mode in ("private", "shared"):
        rows = [row for row in records if row["mode"] == mode]
        wall = [row["measurement"]["wall_ms"] for row in rows]
        summary[mode] = {
            "trials": len(rows), "requests": len(rows),
            "wall_ms_p50": statistics.median(wall),
            "wall_ms_p95": percentile(wall, 95),
            "turnaround_throughput_requests_per_s": len(rows) / (sum(wall) / 1000),
            "steady_mib_p50": statistics.median(row["steady_mib"] for row in rows),
            "startup_peak_mib_p50": statistics.median(
                row["startup_peak_mib"] for row in rows),
        }
    numeric_gate = json.loads(cli.numeric_gate.read_text())
    numeric_contract = numeric_gate.get("numerical_contract", {})
    numeric_ok = (numeric_gate.get("cases") == 20 and
                  numeric_gate.get("event") == "validated_20_fixtures" and
                  numeric_contract.get("same_history_required") is True and
                  numeric_contract.get("both_choices_in_shared_top2") is True)
    for row in records:
        expected = row["measurement"]["expected_tokens"]
        actual = row["measurement"]["tokens"]
        row["measurement"]["exact_sequence_match_diagnostic_only"] = expected == actual
        row["measurement"]["common_prefix_tokens"] = next(
            (i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
            min(len(expected), len(actual)))
        row["measurement"]["external_numeric_gate"] = str(cli.numeric_gate)
    checks = {
        "five_trials_per_arm": all(summary[mode]["trials"] == cli.repeats
                                    for mode in summary),
        "formal_oracle_steps_zero": all(
            row["measurement"]["oracle_steps_in_timing"] == 0 and
            row["measurement"]["timed_prefill_oracle_tokens"] == []
            for row in records),
        "oracle_and_diagnostics_outside_timing": all(
            not row["measurement"]["oracle_in_timing"] and
            not row["measurement"]["diagnostics_in_timing"]
            for row in records),
        # BF16 acceptance is logits/top-2 based at the first same-history
        # divergence; whole generated sequences are intentionally diagnostic.
        "external_numeric_gate": numeric_ok,
        "real_independent_pd": all(
            row["measurement"]["independent_role_pids"] and
            row["measurement"]["handoff_valid"] and
            row["measurement"]["decode_actual_computed_prompt_tokens"] == 0
            for row in records),
        "same_fixed_kv": all(row["kv_blocks"] == 128 for row in records),
        "resources_recovered": all(
            row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256
            for row in records),
    }
    result = {"schema": "pbe-v4-n1-oracle-free-e4-v1",
              "ok": all(checks.values()), "random_order_seed": 20260914,
              "model_sha256": args.model_sha256, "layout_identity": LAYOUT,
              "numeric_gate": {"path": str(cli.numeric_gate),
                               "contract": numeric_contract,
                               "cases": numeric_gate.get("cases"),
                               "accepted_divergences": numeric_gate.get(
                                   "accepted_bf16_divergences", [])},
              "timing_contract": {
                  "oracle_in_timing": False, "diagnostics_in_timing": False,
                  "timing_boundary": "pd_prefill RPC send through pd_decode RPC response",
                  "cleanup_boundary": "pd_release and lifecycle probes after response window",
                  "client_ttft": "N/A: non-streaming interface",
                  "throughput_name": "closed-loop request turnaround throughput"},
              "checks": checks, "summary": summary, "records": records}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (cli.output_dir / "commands.json").write_text(json.dumps({
        "replay": "PYTHONPATH=tools/bench/data_flow:python .venv/bin/python "
                  "tools/bench/data_flow/run_n1_oracle_free_e4.py --build build-v3 "
                  "--model-bin /tmp/Paged-Batch-Engine-models/"
                  "Qwen2.5-VL-3B-Instruct.pbe-bf16-v1.bin --tokenizer "
                  "/tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct/tokenizer.json "
                  "--model-dir /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct "
                  "--image /tmp/pbe-e1-different.png --device 0 --repeats 5 "
                  "--output-dir docs/data_flow_evidence/v4/"
                  "performance_attribution_multi_pd/N1/oracle_free_e4",
        "order": [{"order": i, "mode": mode, "repetition": rep}
                  for i, (mode, rep) in enumerate(order)]}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "ok": result["ok"],
                      "summary": summary}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
