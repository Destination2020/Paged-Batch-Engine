#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import time

from pbe_roles.vision.client import call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    requests = [{"op": "encode", "request_id": f"duplicate-{index}",
                 "generation": 1, "timeout_ms": 30000,
                 "parts": [{"type": "image", "path": args.image, "size": 280},
                           {"type": "text", "text": f"Question {index}."}]}
                for index in range(8)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda request: call(args.endpoint, request), requests))
    wall_ms = (time.perf_counter() - started) * 1000
    assert all(result.get("ok") for result in results)
    output = {"ok": True, "wall_ms": wall_ms,
              "logical_forward_batch_size": max(r["forward_batch_size"] for r in results),
              "physical_forward_batch_size": max(r["physical_forward_batch_size"] for r in results),
              "singleflight_saved": max(r["singleflight_saved"] for r in results),
              "forward_ms": max(r["forward_ms"] for r in results), "results": results}
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
