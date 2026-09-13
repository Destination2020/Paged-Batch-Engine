#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, min(len(values) - 1, int(len(values) * fraction) - 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.runs.glob("*/seed-*/result.json")):
        value = json.loads(path.read_text())
        groups.setdefault(value["policy"], []).append(value)
    assert set(groups) == {"fixed", "round_robin", "data_aware"}
    result = {"ok": True, "policies": {}}
    for policy, runs in groups.items():
        assert len(runs) >= 5 and all(run["ok"] for run in runs)
        assert all(len(run["worker_status"]) == 2 and
                   all(status["pool"]["worker_granted_slots"] == 32 and
                       status["pool"]["active_grants"] == 2 and
                       status["device_memory"]["within_admission_limit"]
                       for status in run["worker_status"])
                   for run in runs)
        records = [record for run in runs for record in run["trace"]]
        ttft = [record["output"]["ttft_ms"] for record in records]
        itl = [record["output"]["itl_ms"] for record in records]
        client = [record["client_language_ms"] for record in records]
        decision = [record["decision"].get("decision_us", 0) for record in records]
        prediction = [abs(record["prediction_error_ms"]) for record in records]
        result["policies"][policy] = {
            "repeats": len(runs), "requests": len(records),
            "mean_client_language_ms": statistics.mean(client),
            "p95_client_language_ms": percentile(client, .95),
            "mean_ttft_ms": statistics.mean(ttft), "p95_ttft_ms": percentile(ttft, .95),
            "mean_itl_ms": statistics.mean(itl),
            "mean_decision_us": statistics.mean(decision),
            "mean_abs_prediction_error_ms": statistics.mean(prediction),
            "rejected": sum(run["rejected"] for run in runs),
            "feature_copy_bytes": sum(sum(status["feature_copy_bytes"]
                                           for status in run["worker_status"])
                                      for run in runs),
            "mean_throughput_requests_per_s": statistics.mean(
                run["throughput_requests_per_s"] for run in runs),
        }
    data_records = [record for run in groups["data_aware"] for record in run["trace"]]
    queue_choices = [r for r in data_records if r["request_id"] == "cache-versus-queue"]
    stale_choices = [r for r in data_records if r["request_id"] == "stale-stat-choice"]
    budget_choices = [r for r in data_records if r["request_id"] == "budget-fallback"]
    assert all(r["worker"] == 1 for r in queue_choices)
    assert all(r["worker"] == 1 for r in stale_choices)
    assert all(r["worker"] == 1 for r in budget_choices)
    assert all(run["restarts"] and run["restarts"][0]["old_pid"] !=
               run["restarts"][0]["new_pid"] for run in groups["data_aware"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
