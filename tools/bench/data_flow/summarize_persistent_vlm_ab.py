#!/usr/bin/env python3
import argparse
import json
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("output")
    args = parser.parse_args()
    source = json.load(open(args.result))
    trace_path = __import__("pathlib").Path(args.result).with_name("trace.jsonl")
    vision = {}
    for line in trace_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("stage") == "encode":
            vision[event["request_id"]] = event["result"]
    records = []
    reference_tokens = None
    for round_result in source["rounds"][1:]:
        output = round_result["outputs"][0]
        _, mode, repetition = output["request_id"].split("-")
        if reference_tokens is None:
            reference_tokens = output["tokens"]
        assert output["tokens"] == reference_tokens
        records.append({
            "mode": mode, "repetition": int(repetition),
            "multimodal_ttft_ms": output["multimodal_ttft_ms"],
            "itl_ms": output["itl_ms"],
            "request_latency_ms": round_result["round_latency_ms"],
            "language_latency_ms": output["latency_ms"],
            "encode_join_ms": round_result["encode_join_ms"],
            "bundle_bytes": round_result["acquired_bundle_bytes"],
            "kv_admitted_bytes": round_result["admitted_kv_bytes"],
            "staging_admitted_bytes": round_result["admitted_staging_bytes"],
            "worker_pid": round_result["worker_pid"],
            "vision_forward_ms": vision[output["request_id"]]["forward_ms"],
            "feature_cache_hit": vision[output["request_id"]]["feature_cache_hit"],
        })
    summary = {}
    for mode in ("off", "on"):
        subset = [record for record in records if record["mode"] == mode]
        summary[mode] = {"n": len(subset)}
        for metric in ("multimodal_ttft_ms", "itl_ms", "request_latency_ms",
                       "encode_join_ms", "vision_forward_ms"):
            values = [record[metric] for record in subset]
            summary[mode][metric] = {
                "median": statistics.median(values), "min": min(values),
                "max": max(values), "p95": percentile(values, .95),
                "p99": percentile(values, .99),
            }
    result = {"schema": "pbe.persistent_vlm_ab.v1", "seed": 20260913,
              "repetitions": 5, "same_output_tokens": True,
              "same_language_pid": len({x["worker_pid"] for x in records}) == 1,
              "summary": summary, "records": records}
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
