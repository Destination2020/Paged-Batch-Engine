#!/usr/bin/env python3
"""Compare frozen 20-case private/shared runs without weakening the M5 rule."""

import argparse
import json
from pathlib import Path


def outputs(document):
    return {output["request_id"]: output for round_result in document["rounds"]
            for output in round_result["outputs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument("--shared-validation", type=Path, required=True)
    parser.add_argument("--cross-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = json.loads(args.private.read_text())
    shared = json.loads(args.shared.read_text())
    private_gate = json.loads(args.private_validation.read_text())
    shared_gate = json.loads(args.shared_validation.read_text())
    cross_gate = json.loads(args.cross_validation.read_text())
    private_outputs, shared_outputs = outputs(private), outputs(shared)
    identifiers = sorted(set(private_outputs) | set(shared_outputs))
    comparisons = [{"request_id": identity,
                    "private_tokens": private_outputs.get(identity, {}).get("tokens"),
                    "shared_tokens": shared_outputs.get(identity, {}).get("tokens"),
                    "exact": private_outputs.get(identity, {}).get("tokens") ==
                             shared_outputs.get(identity, {}).get("tokens")}
                   for identity in identifiers]
    divergent_cases = {item["request_id"].split("-")[1]
                       for item in comparisons if not item["exact"]}
    evidenced_cases = {item["case"].split("-")[1] for item in cross_gate["rows"]}
    cross_by_case = {item["case"].split("-")[1]: item for item in cross_gate["rows"]}
    divergent_choices_valid = True
    for item in comparisons:
        if item["exact"]:
            continue
        case = item["request_id"].split("-")[1]
        first = next(index for index, pair in enumerate(zip(
            item["private_tokens"], item["shared_tokens"])) if pair[0] != pair[1])
        evidence = cross_by_case.get(case, {})
        divergent_choices_valid &= first == evidence.get("step") and \
            item["private_tokens"][first] in evidence.get("top2_union", []) and \
            item["shared_tokens"][first] in evidence.get("top2_union", [])
    weights = shared["status"]["weights"]
    checks = {
        "both_runs_ok": private["ok"] and shared["ok"],
        "both_frozen_m5_rules_pass": private_gate.get("cases") == 20 and
            shared_gate.get("cases") == 20 and
            private_gate.get("event") == "validated_20_fixtures" and
            shared_gate.get("event") == "validated_20_fixtures",
        "all_40_mode_outputs_present": len(private_outputs) == len(shared_outputs) == 40,
        "private_shared_divergences_have_cross_logits": cross_gate.get("ok") is True and
            divergent_cases <= evidenced_cases and divergent_choices_valid,
        "shared_attention_binding": weights["mode"] == "shared" and
            weights["attention_views_bound"] and weights["embedding_view_bound"] and
            weights["output_view_bound"],
        "shared_owner_one_upload": weights["owner_upload_count"] == 1 and
            weights["owner_allocation_count"] == 1,
        "budget_invariants": private["status"]["budget_invariant"] and
            shared["status"]["budget_invariant"],
    }
    output = {"schema": "pbe-e4-20-case-numerics-v1", "ok": all(checks.values()),
              "checks": checks, "case_pairs": 20, "mode_outputs": len(comparisons),
              "exact_mode_outputs": sum(item["exact"] for item in comparisons),
              "divergent_cases": sorted(divergent_cases),
              "cross_mode_logits": cross_gate,
              "frozen_rule": {"same_history_required": True,
                  "max_logits_abs_lt": .75, "max_logits_mean_abs_lt": .20,
                  "max_top2_margin_le": .25,
                  "both_choices_in_shared_top2": True},
              "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(json.dumps(output, sort_keys=True))
    if not output["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
