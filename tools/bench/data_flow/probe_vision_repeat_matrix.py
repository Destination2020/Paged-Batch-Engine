#!/usr/bin/env python3
"""Run real persistent-Vision 0/50/90-percent cache-hit workloads."""
import argparse
import json
import time

from pbe_roles.vision.client import call


def request(endpoint: str, image: str, scenario: str, index: int, size: int) -> dict:
    started = time.perf_counter()
    result = call(endpoint, {
        "op": "encode", "request_id": f"{scenario}-{index}", "generation": 1,
        "timeout_ms": 30000,
        "parts": [{"type": "image", "path": image, "size": size},
                  {"type": "text", "text": f"Describe image for {scenario} case {index}."}],
    })
    assert result.get("ok"), result
    return {
        "index": index, "size": size,
        "cache_hit": result["feature_cache_hit"],
        "forward_ms": result["forward_ms"],
        "wall_ms": (time.perf_counter() - started) * 1000,
        "bundle_bytes": result["bundle_bytes"],
        "worker_pid": result["worker_pid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workloads = {
        "repeat_0": [28 + 28 * i for i in range(10)],
        # First access to 308 is cold, the following five hit; four sizes are unique.
        "repeat_50": [308] * 6 + [336, 364, 392, 420],
        # First access is cold and the remaining nine hit.
        "repeat_90": [448] * 10,
    }
    scenarios = {}
    worker_pids = set()
    for name, sizes in workloads.items():
        rows = [request(args.endpoint, args.image, name, i, size)
                for i, size in enumerate(sizes)]
        worker_pids.update(row["worker_pid"] for row in rows)
        hits = sum(row["cache_hit"] for row in rows)
        scenarios[name] = {"requests": len(rows), "hits": hits,
                           "hit_percent": 100 * hits / len(rows), "rows": rows}
    assert len(worker_pids) == 1
    assert [scenarios[name]["hit_percent"] for name in workloads] == [0, 50, 90]
    output = {"ok": True, "same_worker": True, "worker_pid": worker_pids.pop(),
              "scenarios": scenarios}
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
