#!/usr/bin/env python3
"""One real 1P1D model handoff under Compute Sanitizer memcheck."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_e4_shared_weight_ab import file_sha256, run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    cli = parser.parse_args()
    root = json.loads((cli.model_dir / "config.json").read_text())
    config = root.get("text_config") or root
    args = SimpleNamespace(**vars(cli), layers=config["num_hidden_layers"],
        kv_heads=config["num_key_value_heads"],
        head_size=config["hidden_size"] // config["num_attention_heads"],
        model_sha256=file_sha256(cli.model_bin), compute_sanitizer=True)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    row = run_trial(args, "fixed_kv", "private", 0, 0)
    measurement = row["measurement"]
    result = {
        "schema": "pbe-v4-n5-real-1p1d-compute-sanitizer-v1",
        "tool": "compute-sanitizer --tool memcheck --error-exitcode 99 --leak-check full",
        "scope": "real BF16 model, independent P and D language-role processes",
        "independent_role_pids": measurement["independent_role_pids"],
        "handoff_valid": measurement["handoff_valid"],
        "zero_prefix_recompute":
            measurement["decode_actual_computed_prompt_tokens"] == 0 and
            measurement["decode_scheduled_prefill_tokens"] == 0,
        "resource_recovered": row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256,
        "worker_protocol": [row["prefill_ready"]["protocol"],
                            row["decode_status"]["worker_pid"]],
        "raw_trial": row,
    }
    result["ok"] = all(result[key] for key in (
        "independent_role_pids", "handoff_valid", "zero_prefix_recompute",
        "resource_recovered"))
    cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "output": str(cli.output)}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
