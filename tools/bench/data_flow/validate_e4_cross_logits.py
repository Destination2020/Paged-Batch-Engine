#!/usr/bin/env python3
"""Validate private/shared teacher-history logits at observed divergence steps."""

import argparse
import json
from pathlib import Path


def dumps(document):
    return {item["diagnostic_key"]: item for round_result in document["rounds"]
            for item in round_result["numerical_diagnostics"] if "logits" in item}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = json.loads(args.private.read_text())
    shared = json.loads(args.shared.read_text())
    left, right = dumps(private), dumps(shared)
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key, {}), right.get(key, {})
        same_history = all(a.get(field) == b.get(field) for field in
                           ("step", "history_tokens", "prompt_tokens", "positions", "rope_delta"))
        logits_a, logits_b = a.get("logits", []), b.get("logits", [])
        errors = [abs(x-y) for x, y in zip(logits_a, logits_b)]
        token_a, token_b = a.get("selected_token"), b.get("selected_token")
        top_union = set(a.get("top2_ids", [])) | set(b.get("top2_ids", []))
        selected_equal = token_a == token_b
        accepted = same_history and len(logits_a) == len(logits_b) > 0 and \
            max(errors) < .75 and sum(errors)/len(errors) < .20 and \
            (selected_equal or (min(a["top2_margin"], b["top2_margin"]) <= .25 and
                                token_a in top_union and token_b in top_union))
        rows.append({"case": key, "step": a.get("step"), "same_history": same_history,
            "private_token": token_a, "shared_token": token_b,
            "private_top2_ids": a.get("top2_ids"),
            "shared_top2_ids": b.get("top2_ids"),
            "top2_union": sorted(top_union),
            "private_margin": a.get("top2_margin"), "shared_margin": b.get("top2_margin"),
            "logits_max_abs": max(errors) if errors else None,
            "logits_mean_abs": sum(errors)/len(errors) if errors else None,
            "selected_equal": selected_equal, "accepted": accepted})
    checks = {"both_runs_ok": private.get("ok") and shared.get("ok"),
              "four_observed_cases": len(rows) == 4,
              "same_history_all": all(row["same_history"] for row in rows),
              "frozen_logits_rule_all": all(row["accepted"] for row in rows)}
    result = {"schema": "pbe-e4-cross-mode-logits-v1", "ok": all(checks.values()),
              "checks": checks, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
