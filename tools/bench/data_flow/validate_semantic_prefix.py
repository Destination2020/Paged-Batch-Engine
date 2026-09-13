#!/usr/bin/env python3
"""Executable acceptance gate for the E1 real-VLM semantic prefix trace."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    with open(args.result, encoding="utf-8") as stream:
        result = json.load(stream)
    assert result["ok"] and len(result["rounds"]) == 5
    outputs = {
        item["request_id"]: item
        for round_result in result["rounds"]
        for item in round_result["outputs"]
    }
    producer = outputs["producer"]
    exact = outputs["exact-replay"]
    related = outputs["same-image-different-question"]
    processor = outputs["different-processor"]
    different = outputs["different-image"]
    assert producer["semantic_matched_tokens"] == 0
    assert exact["semantic_matched_tokens"] > 0
    assert exact["semantic_actual_computed_prompt_tokens"] < exact["semantic_prompt_tokens"]
    assert related["semantic_matched_tokens"] > 0
    assert processor["semantic_matched_tokens"] == 0
    assert different["semantic_matched_tokens"] == 0
    first_divergence = next((index for index, pair in enumerate(
        zip(producer["tokens"], exact["tokens"])) if pair[0] != pair[1]), None)
    if first_divergence is not None:
        diagnostics = [item for round_result in result["rounds"]
                       for item in round_result.get("numerical_diagnostics", [])]
        diagnostic = next(item for item in diagnostics
                          if item.get("comparison") == "against_baseline" and
                          item["step"] == first_divergence)
        assert diagnostic["same_history"]
        assert diagnostic["logits_max_abs"] < 0.75
        assert diagnostic["logits_mean_abs"] < 0.20
        assert (diagnostic["top2_margin"] <= 0.25 or
                diagnostic["baseline_top2_margin"] <= 0.25)
        choices = {producer["tokens"][first_divergence],
                   exact["tokens"][first_divergence]}
        shared_top2 = set(diagnostic["top2_ids"] + diagnostic["baseline_top2_ids"])
        assert choices <= shared_top2
    radix = result["status"]["radix_cache"]
    assert radix["cache_hits"] >= 2 and radix["tokens_reused"] > 0
    assert result["status"]["budget_invariant"]
    print(json.dumps({
        "ok": True,
        "exact_matched_tokens": exact["semantic_matched_tokens"],
        "different_question_matched_tokens": related["semantic_matched_tokens"],
        "negative_cases_rejected": 2,
        "first_divergence": first_divergence,
        "radix_cache": radix,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
