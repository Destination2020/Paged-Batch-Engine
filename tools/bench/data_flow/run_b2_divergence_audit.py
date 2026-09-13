#!/usr/bin/env python3
"""Capture paired teacher-history logits for B2's observed step-2 divergence."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_b2_cache_matrix import make_variants


def request(identity, image, diagnostic_key=""):
    item = {"request_id": identity, "generation": 1,
            "parts": [{"type": "image", "path": str(image), "size": 280},
                      {"type": "text", "text": " ".join(
                          ["Describe every visible component, relationship, direction, label, and layout detail."] * 12) +
                       " Finish with the central object."}],
            "max_new_tokens": 8}
    if diagnostic_key:
        item.update({"diagnostic_key": diagnostic_key,
                     "diagnostic_mode": identity,
                     "diagnostic_dump_logits_step": 2})
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    variants = make_variants(args.image, out / "inputs")
    target = variants[0]
    spec = {"rounds": [[request("warmup", variants[10])],
                       [request("cache-fill", target)]] +
                      [[request(f"audit-{index}", target, f"paired-step2-{index}")]
                       for index in range(5)]}
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    commands = []
    for arm in ("00", "11"):
        trial = out / f"arm{arm}"; trial.mkdir(exist_ok=True)
        invocation = ["tools/launch/persistent_vlm_online.sh", str(args.build),
                      str(args.model_bin), str(args.tokenizer), str(args.model_dir),
                      str(args.image)]
        environment = dict(os.environ, PBE_DEVICE=str(args.device), PBE_RUN_DIR=str(trial),
                           PBE_SPEC=str(out / "spec.json"), PBE_WEIGHT_MODE="shared",
                           PBE_DETERMINISTIC_VISION="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                           PYTHONPATH="python")
        if arm == "00":
            environment["PBE_DISABLE_FEATURE_CACHE"] = "1"
            environment["PBE_DISABLE_RADIX_CACHE"] = "1"
        command = {"arm": arm, "argv": invocation,
                   "environment": {key: environment[key] for key in
                    ("PBE_DEVICE", "PBE_RUN_DIR", "PBE_SPEC", "PBE_WEIGHT_MODE",
                     "PBE_DETERMINISTIC_VISION", "CUBLAS_WORKSPACE_CONFIG")}}
        if arm == "00": command["environment"].update(
            PBE_DISABLE_FEATURE_CACHE="1", PBE_DISABLE_RADIX_CACHE="1")
        commands.append(command)
        with (trial / "launch.log").open("w") as log:
            completed = subprocess.run(invocation, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=600)
        if completed.returncode:
            raise RuntimeError(f"arm {arm} failed: {completed.returncode}")
    (out / "commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")

    def dumps(arm):
        result = json.loads((out / f"arm{arm}" / "result.json").read_text())
        values = {}
        for round_result in result["rounds"]:
            for item in round_result.get("numerical_diagnostics", []):
                if item.get("diagnostic_key", "").startswith("paired-step2-") and "logits" in item:
                    values[item["diagnostic_key"]] = item
        return values
    left, right = dumps("00"), dumps("11")
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key, {}), right.get(key, {})
        same_history = all(a.get(field) == b.get(field) for field in
                           ("step", "history_tokens", "prompt_tokens", "positions", "rope_delta"))
        errors = [abs(x-y) for x, y in zip(a.get("logits", []), b.get("logits", []))]
        choices = {a.get("selected_token"), b.get("selected_token")}
        accepted = (same_history and len(errors) > 0 and max(errors) < .75 and
                    sum(errors)/len(errors) < .20 and
                    a.get("top2_margin", 1) <= .25 and b.get("top2_margin", 1) <= .25 and
                    choices <= set(a.get("top2_ids", [])) and choices <= set(b.get("top2_ids", [])))
        rows.append({"key": key, "same_history": same_history,
                     "baseline_token": a.get("selected_token"),
                     "candidate_token": b.get("selected_token"),
                     "baseline_top2_ids": a.get("top2_ids"),
                     "candidate_top2_ids": b.get("top2_ids"),
                     "baseline_margin": a.get("top2_margin"),
                     "candidate_margin": b.get("top2_margin"),
                     "logits_max_abs": max(errors) if errors else None,
                     "logits_mean_abs": sum(errors)/len(errors) if errors else None,
                     "accepted": accepted})
    result = {"schema": "pbe-v4-b2-step2-paired-audit-v1", "rows": rows,
              "comparisons": len(rows), "ok": len(rows) == 5 and all(x["accepted"] for x in rows),
              "performance_sample": False,
              "reason": "teacher-history evidence for first output divergence observed in formal 90% stress trials"}
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "comparisons": len(rows)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
