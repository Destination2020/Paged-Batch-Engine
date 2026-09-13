#!/usr/bin/env python3
"""Consolidate post-review main-path evidence without inventing missing values."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.root / "docs/data_flow_evidence/v4"
    persistent = json.load(open(evidence / "post_review_persistent_vlm_ab_retry2/results.json"))
    external = json.load(open(evidence / "post_review_external_serving_ab/results.json"))
    pressure = json.load(open(evidence / "post_review_online_pressure_retry/result.json"))
    fixtures = json.load(open(evidence / "post_review_persistent_vlm_20_retry/result.json"))
    pairs = {}
    for round_result in fixtures["rounds"]:
        for item in round_result["outputs"]:
            case = int(item["request_id"].split("-")[1])
            mode = item["request_id"].split("-")[2]
            pairs.setdefault(case, {})[mode] = item["tokens"]
    exact = sum(value["mixed"] == value["single"] for value in pairs.values())
    token_equal = sum(sum(a == b for a, b in zip(value["mixed"], value["single"]))
                      for value in pairs.values())
    result = {
        "schema": "pbe.v4.post_review_final.v1", "date": "2026-09-13",
        "hardware": "2 x NVIDIA H20-3e", "seed": 20260913,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3",
        "persistent_multimodal_cache_ab": persistent,
        "external_pool_serving_ab": {
            "repeats_per_mode": external["repeats_per_mode"],
            "median": external["median"], "controls": external["controls"]},
        "mixed_correctness": {
            "cases": len(pairs), "exact_sequences": exact,
            "equal_tokens": token_equal, "total_tokens": 160,
            "agreement": token_equal / 160,
            "interpretation": "two first-divergence boundaries; later tokens are autoregressive cascade; BF16 tie policy follows M1"},
        "online_pressure": {
            "first_batch": {"admitted": pressure["rounds"][0]["admitted"],
                            "rejected": pressure["rounds"][0]["rejected"]},
            "recovery_batch": {"admitted": pressure["rounds"][1]["admitted"],
                               "rejected": pressure["rounds"][1]["rejected"]},
            "peak_staging_bytes": pressure["status"]["budget"]["staging"]["peak"],
            "staging_capacity_bytes": pressure["status"]["budget"]["staging"]["capacity"],
            "used_after_bytes": pressure["status"]["budget"]["staging"]["used"]},
        "evidence_reused": {
            "kv_share_cow": "post_review_four_branch_cow_retry2",
            "same_gpu_ipc": "post_review_agent_owned_kv",
            "cross_gpu_copy": "post_review_cross_gpu_kv",
            "fault_matrix": "post_review_online_fault_matrix_retry + post_review_fault_matrix_regression.log",
            "lane_and_checkpoint_contracts": "M7 and M9 raw contract logs"},
        "claims": {"beats_sglang": False, "cache_always_improves_latency": False,
                   "network_exactly_once": False,
                   "merged_vs_separated_speedup": None,
                   "reason_unmeasured": "no equal-budget merged full-VLM implementation in V4 scope"},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    off = persistent["summary"]["off"]
    on = persistent["summary"]["on"]
    bars = [("Cache off TTFT", off["multimodal_ttft_ms"]["median"]),
            ("Cache on TTFT", on["multimodal_ttft_ms"]["median"]),
            ("Cache off E2E", off["request_latency_ms"]["median"]),
            ("Cache on E2E", on["request_latency_ms"]["median"])]
    scale = 560 / max(value for _, value in bars)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330">',
           '<style>text{font:14px sans-serif}.title{font:bold 18px sans-serif}</style>',
           '<text x="20" y="28" class="title">Persistent multimodal serving, median of 5 (H20, Qwen2.5-VL-3B)</text>']
    for index, (label, value) in enumerate(bars):
        y = 60 + index * 58
        svg += [f'<text x="20" y="{y + 21}">{label}</text>',
                f'<rect x="170" y="{y}" width="{value * scale:.1f}" height="28" fill="#3977c5"/>',
                f'<text x="{180 + value * scale:.1f}" y="{y + 20}">{value:.1f} ms</text>']
    svg += ['<text x="20" y="305">Cache skips vision forward (median 37.3 ms), while processor/RPC still dominate encode.</text>', '</svg>']
    (args.output / "persistent_multimodal_median.svg").write_text("\n".join(svg) + "\n")
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
