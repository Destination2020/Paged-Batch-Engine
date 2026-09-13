#!/usr/bin/env python3
import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--output")
    args = parser.parse_args()
    value = json.load(open(args.result, encoding="utf-8"))
    groups = {mode: [row["d2h_completion_ms"] for row in value["records"]
                     if row["mode"] == mode]
              for mode in ("lanes_off", "lanes_on")}
    assert value["ok"] and value["streams"] == 2 and value["max_total_active"] == 2
    assert all(len(rows) == 5 for rows in groups.values())
    result = {"ok": True, "bytes_per_transfer": value["bytes_per_transfer"],
              "same_streams": 2, "same_max_total_active": 2,
              "median_d2h_completion_ms": {mode: statistics.median(rows)
                                            for mode, rows in groups.items()},
              "randomized_order": [row["mode"] for row in value["records"]]}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        open(args.output, "w", encoding="utf-8").write(text)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
