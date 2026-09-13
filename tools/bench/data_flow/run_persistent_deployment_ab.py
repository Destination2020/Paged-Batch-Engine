#!/usr/bin/env python3
"""Warm one-vs-two complete Language-replica cost A/B (not P/D split)."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def language(args, slots):
    return LanguageProcess([str(args.language_binary), str(args.model_bin),
        str(args.tokenizer), args.data_endpoint, str(args.device),
        str(64 << 20), str(32 << 20), str(slots)])


def request(item, encoded, generation):
    return {"request_id": item["request_id"], "generation": generation,
            "content": encoded["content"], "representation": encoded["representation"],
            "feature_content": encoded["feature_content"],
            "feature_representation": encoded["feature_representation"],
            "max_new_tokens": item["max_new_tokens"], "timeout_ms": 120000}


def measured_budget(statuses):
    return {"model_process_allocation_bytes": sum(
                x["device_memory"]["model_process_allocation_bytes"] for x in statuses),
            "workspace_reserved_bytes": sum(
                x["device_memory"]["workspace_reserved_bytes"] for x in statuses),
            "worker_allocatable_kv_bytes": sum(
                x["pool"]["worker_allocatable_kv_bytes"] for x in statuses),
            "weight_processes": len(statuses),
            "all_within_device_admission_limit": all(
                x["device_memory"]["within_admission_limit"] for x in statuses)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vision-endpoint", required=True)
    p.add_argument("--language-binary", type=Path, required=True)
    p.add_argument("--model-bin", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--data-endpoint", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    item = {"request_id": "persistent-deployment", "generation": 1,
            "parts": [{"type": "image", "path": str(args.image), "size": 224},
                      {"type": "text", "text":
                       "Describe the image briefly and name its main subject."}],
            "max_new_tokens": 16, "timeout_ms": 120000}
    item["_deadline_monotonic_ns"] = time.monotonic_ns() + 120_000_000_000
    _, encoded = encode(args.vision_endpoint, item)
    if not encoded.get("ok"):
        raise RuntimeError("vision encode failed")
    unified = [language(args, 64)]
    split = [language(args, 32), language(args, 32)]
    topologies = {"single_full_language_replica": unified,
                  "two_full_language_replicas_round_robin": split}
    records = []
    try:
        # Warm every resident model before opening the observation window.
        for mode, workers in topologies.items():
            for index, worker in enumerate(workers):
                warm = dict(item, request_id=f"warm-{mode}-{index}", max_new_tokens=2)
                result = worker.call({"op": "infer", "requests": [
                    request(warm, encoded, 1)]}, timeout=180)
                if not result.get("ok") or not result.get("outputs"):
                    raise RuntimeError(f"warmup failed: {mode}: {result}")
        order = [mode for _ in range(args.repeats) for mode in topologies]
        random.Random(20260913).shuffle(order)
        counts = {mode: 0 for mode in topologies}
        window_start = time.perf_counter()
        for order_index, mode in enumerate(order):
            repetition = counts[mode]
            counts[mode] += 1
            workers = topologies[mode]
            worker_index = repetition % len(workers)
            started = time.perf_counter()
            result = workers[worker_index].call({"op": "infer", "requests": [
                request(item, encoded, repetition + 2)]}, timeout=180)
            completed = time.perf_counter()
            if not result.get("ok") or not result.get("outputs"):
                raise RuntimeError(f"request failed: {mode}: {result}")
            output = result["outputs"][0]
            records.append({"mode": mode, "repetition": repetition,
                "order": order_index, "worker_index": worker_index,
                "client_ms": (completed - started) * 1000,
                "ttft_ms": output["ttft_ms"], "itl_ms": output["itl_ms"],
                "token_itl_ms": output["token_itl_ms"], "tokens": output["tokens"]})
        window_end = time.perf_counter()
        statuses = {mode: [worker.call({"op": "status"}) for worker in workers]
                    for mode, workers in topologies.items()}
    finally:
        for workers in topologies.values():
            for worker in workers:
                worker.close()
    summaries = {}
    for mode in topologies:
        rows = [x for x in records if x["mode"] == mode]
        token_itls = [value for row in rows for value in row["token_itl_ms"]]
        summaries[mode] = {"repeats": len(rows),
            "median_client_ms": statistics.median(x["client_ms"] for x in rows),
            "p95_client_ms": percentile([x["client_ms"] for x in rows], 95),
            "p99_client_ms": percentile([x["client_ms"] for x in rows], 99),
            "client_stdev_ms": statistics.pstdev(x["client_ms"] for x in rows),
            "median_ttft_ms": statistics.median(x["ttft_ms"] for x in rows),
            "p95_ttft_ms": percentile([x["ttft_ms"] for x in rows], 95),
            "p99_ttft_ms": percentile([x["ttft_ms"] for x in rows], 99),
            "median_itl_ms": statistics.median(x["itl_ms"] for x in rows),
            "token_itl_p50_ms": percentile(token_itls, 50),
            "token_itl_p95_ms": percentile(token_itls, 95),
            "token_itl_p99_ms": percentile(token_itls, 99),
            "effective_service_throughput_requests_per_s":
                len(rows) / (sum(x["client_ms"] for x in rows) / 1000),
            "shared_window_contribution_requests_per_s":
                len(rows) / (window_end - window_start),
            "measured_budget": measured_budget(statuses[mode])}
    by_mode = {mode: [x["tokens"] for x in records if x["mode"] == mode]
               for mode in topologies}
    completed = len(records) == args.repeats * len(topologies) and all(
        row["tokens"] and row["token_itl_ms"] for row in records)
    budgets_valid = all(measured_budget(statuses[mode])[
        "all_within_device_admission_limit"] for mode in topologies)
    result = {"ok": completed and budgets_valid,
        "schema": "pbe.v4.full-language-replica-count-ab.v1",
        "experiment_scope": "one complete infer replica versus two complete infer replicas",
        "counts_as_prefill_decode_handoff_acceptance": False,
        "cold_start_excluded": True, "both_topologies_concurrently_resident": True,
        "completed_requests": len(records),
        "token_equality_is_not_the_acceptance_gate": True,
        "bf16_numerical_acceptance_evidence": "M5_numerics_final/validation_final.json",
        "same_device": args.device, "same_kv_slots_per_topology": 64,
        "observation_window_ms": (window_end - window_start) * 1000,
        "aggregate_throughput_requests_per_s":
            len(records) / (window_end - window_start),
        "arrival_order_seed": 20260913,
        "internally_stable_tokens": {mode: all(v == values[0] for v in values)
                                     for mode, values in by_mode.items()},
        "cross_topology_same_tokens": by_mode["single_full_language_replica"][0] ==
                                      by_mode["two_full_language_replicas_round_robin"][0],
        "summaries": summaries, "records": records, "worker_status": statuses}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
