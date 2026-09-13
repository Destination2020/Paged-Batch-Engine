#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def row_from_result(path, mode):
    value = json.loads(path.read_text())
    replay = value["rounds"][1]["outputs"][0]
    radix = value["status"]["radix_cache"]
    return {"mode": mode, "ttft_ms": replay["ttft_ms"],
            "latency_ms": replay["latency_ms"],
            "matched_tokens": replay["semantic_matched_tokens"],
            "recomputed_tokens": replay["semantic_actual_computed_prompt_tokens"],
            "token_itl_ms": replay["token_itl_ms"],
            "host_demoted_blocks": radix["host_demoted_blocks"],
            "host_restored_blocks": radix["host_restored_blocks"],
            "restore_failures": radix["host_restore_failures"],
            "checkpoint_events": value["rounds"][1]["checkpoint_events"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-ab", type=Path, required=True)
    parser.add_argument("--host-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {
        "drop_recompute": [row_from_result(p, "drop_recompute") for p in
                           sorted((args.kv_ab / "off").glob("seed-*/result.json"))],
        "fixed_gpu_retain": [row_from_result(p, "fixed_gpu_retain") for p in
                             sorted((args.kv_ab / "on").glob("seed-*/result.json"))],
        "dependency_host_checkpoint": [row_from_result(p, "dependency_host_checkpoint")
                                       for p in sorted(args.host_runs.glob("seed-*/result.json"))],
    }
    result = {"ok": True, "policies": {}}
    for mode, rows in groups.items():
        assert len(rows) == 5
        gaps = [gap for row in rows for gap in row["token_itl_ms"]]
        result["policies"][mode] = {
            "repeats": 5, "median_ttft_ms": statistics.median(r["ttft_ms"] for r in rows),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
            "median_recomputed_tokens": statistics.median(r["recomputed_tokens"] for r in rows),
            "itl_p95_ms": sorted(gaps)[int(.95 * (len(gaps) - 1))],
            "host_demoted_blocks": sum(r["host_demoted_blocks"] for r in rows),
            "host_restored_blocks": sum(r["host_restored_blocks"] for r in rows),
            "restore_failures": sum(r["restore_failures"] for r in rows),
            "checkpoint_restores": sum(len(r["checkpoint_events"]) for r in rows),
            "rows": rows,
        }
    host = result["policies"]["dependency_host_checkpoint"]
    assert host["host_demoted_blocks"] > 0
    assert host["host_demoted_blocks"] == host["host_restored_blocks"]
    assert host["restore_failures"] == 0 and host["checkpoint_restores"] == 5
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
