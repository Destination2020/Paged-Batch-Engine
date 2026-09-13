#!/usr/bin/env python3
"""Executable gate for E3 physical Host restore and multimodal checkpoint."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("trace")
    args = parser.parse_args()
    result = json.load(open(args.result, encoding="utf-8"))
    trace = [json.loads(line) for line in open(args.trace, encoding="utf-8")]
    assert result["ok"] and len(result["rounds"]) == 2
    producer = result["rounds"][0]["outputs"][0]
    recovered = result["rounds"][1]["outputs"][0]
    assert recovered["semantic_matched_tokens"] > 0
    first_divergence = next((i for i, pair in enumerate(zip(
        producer["tokens"], recovered["tokens"])) if pair[0] != pair[1]), None)
    diagnostics = result["rounds"][1]["numerical_diagnostics"]
    if first_divergence is not None:
        diagnostic = next(x for x in diagnostics if x["step"] == first_divergence)
        assert diagnostic["same_history"]
        assert diagnostic["logits_max_abs"] < 0.75
        assert diagnostic["logits_mean_abs"] < 0.20
        assert diagnostic["top2_margin"] <= 0.25 or diagnostic["baseline_top2_margin"] <= 0.25
        assert recovered["tokens"][first_divergence] == diagnostic["selected_token"]
        assert producer["tokens"][first_divergence] == diagnostic["baseline_selected_token"]
        assert set((diagnostic["selected_token"],
                    diagnostic["baseline_selected_token"])) <= set(
                        diagnostic["top2_ids"] + diagnostic["baseline_top2_ids"])
    demotions = [x["result"] for x in trace if x.get("stage") == "tiered_cache_demotion"]
    assert len(demotions) == 1 and demotions[0]["host_demoted_blocks"] > 0
    checkpoints = result["rounds"][1]["checkpoint_events"]
    assert checkpoints and checkpoints[0]["ok"] and checkpoints[0]["restored"]
    assert checkpoints[0]["exact_multimodal_dependency"]
    assert checkpoints[0]["position_values"] > 0 and checkpoints[0]["sampling_counter"] > 0
    assert checkpoints[0]["private_tail_tokens"] > 0
    assert checkpoints[0]["private_tail_restored"]
    radix = result["status"]["radix_cache"]
    assert radix["host_restore_requests"] > 0
    assert radix["host_restored_blocks"] == radix["host_demoted_blocks"]
    assert radix["host_restore_failures"] == 0
    assert result["status"]["budget_invariant"]
    print(json.dumps({"ok": True, "first_divergence": first_divergence,
                      "demotion": demotions[0],
                      "checkpoint": checkpoints[0], "radix_cache": radix},
                     sort_keys=True))


if __name__ == "__main__":
    main()
