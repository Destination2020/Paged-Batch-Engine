#!/usr/bin/env python3
"""Validate mixed results against the same persistent role's single-request oracle."""
import argparse
import json


MAX_LOGITS_ABS = 0.75
MAX_LOGITS_MEAN_ABS = 0.20
MAX_DIVERGENCE_MARGIN = 0.25


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    result = json.load(open(args.result))
    by_case = {}
    diagnostics = {}
    mixed_pid = set()
    mixed_decode = 0
    for round_result in result["rounds"]:
        assert round_result["ok"] and round_result["rejected"] == 0
        assert round_result["budget_invariant"]
        mixed_decode += round_result["mixed_prefill_decode_steps"]
        mixed_pid.add(round_result["worker_pid"])
        for output in round_result["outputs"]:
            assert not output["failed"] and len(output["tokens"]) == 8
            case = int(output["request_id"].split("-")[1])
            kind = output["request_id"].split("-")[2]
            by_case.setdefault(case, {})[kind] = output["tokens"]
        for item in round_result.get("numerical_diagnostics", []):
            if item.get("comparison") == "against_baseline":
                diagnostics[(item["diagnostic_key"], item["step"])] = item
    require(len(mixed_pid) == 1, "language role was not persistent")
    require(len(by_case) == 20, f"expected 20 cases, got {len(by_case)}")
    exact_sequences = 0
    accepted_divergences = []
    equal_tokens = 0
    total_tokens = 0
    for case, values in sorted(by_case.items()):
        mixed = values["mixed"]
        single = values["single"]
        total_tokens += len(mixed)
        equal_tokens += sum(a == b for a, b in zip(mixed, single))
        if mixed == single:
            exact_sequences += 1
            continue
        first = next(step for step, pair in enumerate(zip(mixed, single))
                     if pair[0] != pair[1])
        item = diagnostics.get((f"case-{case:02d}", first))
        require(item is not None,
                f"case {case:02d} first divergence step {first} lacks logits evidence")
        require(item.get("same_history") is True,
                f"case {case:02d} divergence was not compared under the same history")
        require(item["baseline_selected_token"] == mixed[first] and
                item["selected_token"] == single[first],
                f"case {case:02d} diagnostic tokens do not match outputs")
        require(item["logits_max_abs"] < MAX_LOGITS_ABS and
                item["logits_mean_abs"] < MAX_LOGITS_MEAN_ABS,
                f"case {case:02d} same-history logits exceed frozen error limits")
        require(item["baseline_top2_margin"] <= MAX_DIVERGENCE_MARGIN and
                item["top2_margin"] <= MAX_DIVERGENCE_MARGIN,
                f"case {case:02d} divergence has a high-confidence margin")
        candidates = {mixed[first], single[first]}
        require(candidates.issubset(set(item["baseline_top2_ids"])) and
                candidates.issubset(set(item["top2_ids"])),
                f"case {case:02d} divergent choices are not the shared top-2 candidates")
        accepted_divergences.append({
            "case": case, "step": first,
            "mixed_token": mixed[first], "single_token": single[first],
            "mixed_margin": item["baseline_top2_margin"],
            "single_margin": item["top2_margin"],
            "logits_max_abs": item["logits_max_abs"],
            "logits_mean_abs": item["logits_mean_abs"],
        })
    require(mixed_decode > 0, "no mixed prefill/decode step was observed")
    print(json.dumps({"event": "validated_20_fixtures", "language_pid": next(iter(mixed_pid)),
                      "cases": 20, "exact_sequences": exact_sequences,
                      "equal_tokens": equal_tokens, "total_tokens": total_tokens,
                      "accepted_bf16_divergences": accepted_divergences,
                      "numerical_contract": {
                          "same_history_required": True,
                          "max_logits_abs_lt": MAX_LOGITS_ABS,
                          "max_logits_mean_abs_lt": MAX_LOGITS_MEAN_ABS,
                          "max_top2_margin_le": MAX_DIVERGENCE_MARGIN,
                          "both_choices_in_shared_top2": True,
                      },
                      "mixed_prefill_decode_steps": mixed_decode,
                      "budget_invariant": True}, sort_keys=True))


if __name__ == "__main__":
    main()
