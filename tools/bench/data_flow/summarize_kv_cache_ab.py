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
        rows = []
        for path in sorted((args.runs / mode).glob("seed-*/result.json")):
            value = json.loads(path.read_text())
            replay = value["rounds"][1]["outputs"][0]
            rows.append({"ttft_ms": replay["ttft_ms"],
                         "multimodal_ttft_ms": replay["multimodal_ttft_ms"],
                         "latency_ms": replay["latency_ms"],
                         "matched_tokens": replay["semantic_matched_tokens"],
                         "computed_tokens": replay["semantic_actual_computed_prompt_tokens"],
                         "token_itl_ms": replay["token_itl_ms"],
                         "lookup_us": replay["semantic_lookup_us"]})
        assert len(rows) == 5
        token_itl = [gap for row in rows for gap in row["token_itl_ms"]]
        result["modes"][mode] = {
            "repeats": len(rows),
            "median_ttft_ms": statistics.median(r["ttft_ms"] for r in rows),
            "median_multimodal_ttft_ms": statistics.median(
                r["multimodal_ttft_ms"] for r in rows),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
            "median_matched_tokens": statistics.median(r["matched_tokens"] for r in rows),
            "median_computed_tokens": statistics.median(r["computed_tokens"] for r in rows),
            "median_lookup_us": statistics.median(r["lookup_us"] for r in rows),
            "token_itl_p50_ms": statistics.median(token_itl),
            "token_itl_p95_ms": sorted(token_itl)[int(.95 * (len(token_itl) - 1))],
            "token_itl_p99_ms": sorted(token_itl)[int(.99 * (len(token_itl) - 1))],
            "rows": rows,
        }
    assert result["modes"]["off"]["median_matched_tokens"] == 0
    assert result["modes"]["on"]["median_matched_tokens"] > 0
    assert result["modes"]["on"]["median_computed_tokens"] < \
        result["modes"]["off"]["median_computed_tokens"]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
