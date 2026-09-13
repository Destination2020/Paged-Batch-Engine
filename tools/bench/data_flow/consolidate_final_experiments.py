#!/usr/bin/env python3
"""Consolidate the completed real-model V4/E1-E3 ablation evidence."""
import argparse
import json
from pathlib import Path


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    source_paths = {
        "feature_cache": root / "M9/results.json",
        "kv_share": root / "M9/kv_cache_ab/results.json",
        "singleflight": root / "M9/singleflight_ab/results.json",
        "lane_scheduler_cuda": root / "M9/transfer_scheduler_lane_final/results.json",
        "lane_model_waiter": root / "M9/model_transfer_scheduler_lane_final_gpu0_v2/results.json",
        "replica_count_cost": root / "M9/persistent_deployment_final_gpu0_v2/results.json",
        "deployment_persistent_pd": root / "M9/persistent_pd_deployment_final_gpu0_v4/results.json",
        "pressure": root / "M9/pressure_policy_ab/results.json",
        "placement": root / "E2_final/results.json",
        "repeat_matrix": root / "M9/repeat_matrix/result.json",
        "vision_feature_recovery": root / "E3/feature_gpu_host_recovery/result.json",
    }
    data = {name: load(path) for name, path in source_paths.items()}
    assert all(item.get("ok", all(row.get("ok", False)
                                  for row in item.get("records", [])))
               for item in data.values())
    # The original M9 snapshot listed then-unmeasured work. The source remains
    # immutable evidence, but those historical declarations are not current
    # fields in the completed consolidation.
    data["feature_cache"].pop("claims", None)
    data["replica_count_cost"]["final_scope_annotation"] = {
        "name": "single_vs_two_full_language_replicas_round_robin",
        "counts_as_prefill_decode_handoff_acceptance": False,
        "reason": "each request performs complete Prefill and Decode in one worker",
    }
    historical = {
        "lane_microbenchmark": {
            "path": "M9/lane_ab/results.json",
            "counts_as_final_lane_acceptance": False,
            "reason": "did not instantiate the production TransferScheduler"},
        "cold_start_deployment": {
            "path": "M9/deployment_ab/results.json",
            "counts_as_persistent_serving_acceptance": False,
            "reason": "included process startup and one-shot separated execution"},
    }
    output = {
        "schema": "pbe.v4.final-experiments.v2",
        "ok": True,
        "hardware": "2x NVIDIA H20-3e",
        "model": "Qwen2.5-VL-3B-Instruct, BF16 PBE export v1",
        "randomized_repetitions": 5,
        "experiments": data,
        "historical_non_acceptance": historical,
        "matrix": {
            "image_repeat_percent": [0, 50, 90],
            "branches": [1, 2, 4],
            "prefix": ["short", "long"],
            "resolution": [224, 280],
            "pressure": ["low", "medium", "high"],
            "method": "representative combinations and single-factor ablations",
        },
        "transport_evidence": {
            "host_copy": "M3/raw/host-copy.log",
            "same_gpu_ipc": "M3/raw/ipc-pool.log",
            "cross_gpu_sparse_pages": "E1/cross_gpu_page_subset_final",
        },
        "claims": {
            "superiority_over_external_systems": False,
            "negative_results_retained": True,
            "unmeasured_fields": [],
        },
        "sources": {name: str(path.relative_to(root))
                    for name, path in source_paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"ok": True, "output": str(args.output),
                      "experiments": sorted(data)}, sort_keys=True))


if __name__ == "__main__":
    main()
