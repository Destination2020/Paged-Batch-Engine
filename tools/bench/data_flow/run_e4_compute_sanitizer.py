#!/usr/bin/env python3
"""Run one real shared-weight P/D lifecycle under CUDA memcheck."""

import argparse
import hashlib
import json
from pathlib import Path

from run_e4_shared_weight_ab import run_trial


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    args = parser.parse_args()
    root = json.loads((args.model_dir / "config.json").read_text())
    config = root.get("text_config") or root
    args.layers = config["num_hidden_layers"]
    args.kv_heads = config["num_key_value_heads"]
    args.head_size = config["hidden_size"] // config["num_attention_heads"]
    args.model_sha256 = file_sha256(args.model_bin)
    args.compute_sanitizer = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row = run_trial(args, "fixed_kv", "shared", 0, 0)
    checks = {"real_pd": row["measurement"]["independent_role_pids"] and
              row["measurement"]["handoff_valid"],
              "attention_shared": row["prefill_status"]["weights"]["attention_views_bound"] and
              row["decode_status"]["weights"]["attention_views_bound"],
              "numeric": row["measurement"]["output_matches_prefill_oracle"],
              "lifecycle": row["lifecycle"]["decode_survived_prefill_exit"] and
              row["lifecycle"]["new_importer_joined"],
              "leases_recovered": row["weight_after_workers"]["active_leases"] == 0}
    result = {"schema": "pbe-e4-compute-sanitizer-v1",
              "configuration": {
                  "tool": "memcheck", "error_exitcode": 99,
                  "leak_check": "full", "report_api_errors": "no",
                  "report_api_errors_reason":
                      "exclude third-party cuBLAS cuCtxGetLimit capability probes"
              },
              "ok": all(checks.values()), "checks": checks, "trial": row}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"ok": result["ok"], "checks": checks}, sort_keys=True))
    if not result["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
