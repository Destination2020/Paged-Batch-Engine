#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"ok": True, "modes": {}}
    for mode in ("off", "on"):
        rows = [json.loads(path.read_text()) for path in sorted(
            (args.runs / mode).glob("seed-*/result.json"))]
        assert len(rows) == 5 and all(row["ok"] for row in rows)
        result["modes"][mode] = {
            "repeats": 5,
            "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
            "median_forward_ms": statistics.median(row["forward_ms"] for row in rows),
            "physical_forward_batch_sizes": [row["physical_forward_batch_size"] for row in rows],
            "singleflight_saved": [row["singleflight_saved"] for row in rows],
        }
    assert result["modes"]["off"]["physical_forward_batch_sizes"] == [8] * 5
    assert result["modes"]["on"]["physical_forward_batch_sizes"] == [1] * 5
    assert result["modes"]["on"]["singleflight_saved"] == [7] * 5
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
