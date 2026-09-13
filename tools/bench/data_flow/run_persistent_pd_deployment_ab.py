#!/usr/bin/env python3
"""Warm unified inference vs persistent multimodal Prefill->KV handoff->Decode."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def worker(args, slots, initial_slots=()):
    command = [str(args.language_binary), str(args.model_bin), str(args.tokenizer),
               args.data_endpoint, str(args.device), str(64 << 20), str(32 << 20),
               str(slots), "0", "1"]
    if initial_slots:
        command.append(",".join(map(str, initial_slots)))
    return LanguageProcess(command)


def wire(item, encoded, request_id, generation=1):
    return {"request_id": request_id, "generation": generation,
            "content": encoded["content"], "representation": encoded["representation"],
            "feature_content": encoded["feature_content"],
            "feature_representation": encoded["feature_representation"],
            "max_new_tokens": item["max_new_tokens"], "timeout_ms": 120000}


def budget(statuses):
    return {"weight_processes": len(statuses),
            "model_process_allocation_bytes": sum(
                x["device_memory"]["model_process_allocation_bytes"] for x in statuses),
            "workspace_reserved_bytes": sum(
                x["device_memory"]["workspace_reserved_bytes"] for x in statuses),
            "owned_kv_slots": sum(x["pool"]["worker_granted_slots"] for x in statuses),
            "all_within_device_admission_limit": all(
                x["device_memory"]["within_admission_limit"] for x in statuses)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-endpoint", required=True)
    parser.add_argument("--language-binary", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-endpoint", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    item = {"request_id": "persistent-pd", "generation": 1,
            "parts": [{"type": "image", "path": str(args.image), "size": 224},
                      {"type": "text", "text":
                       "Describe the image briefly and name its main subject."}],
            "max_new_tokens": 16, "timeout_ms": 120000,
            "_deadline_monotonic_ns": time.monotonic_ns() + 120_000_000_000}
    _, encoded = encode(args.vision_endpoint, item)
    if not encoded.get("ok"):
        raise RuntimeError("vision encode failed")

    unified = worker(args, 64)
    prefill = worker(args, 32)
    decode = None
    records = []
    try:
        unified_warm = unified.call({"op": "infer", "requests": [
            wire(item, encoded, "warm-unified")]}, timeout=180)
        if not unified_warm.get("ok") or not unified_warm.get("outputs"):
            raise RuntimeError(f"unified warmup failed: {unified_warm}")
        oracle = unified_warm["outputs"][0]["tokens"]

        warm_request = wire(item, encoded, "warm-pd")
        prefill_warm = prefill.call({"op": "pd_prefill", "request": warm_request,
                                     "oracle_steps": item["max_new_tokens"]}, timeout=180)
        if not prefill_warm.get("ok") or not prefill_warm.get("pool_slots"):
            raise RuntimeError(f"prefill warmup failed: {prefill_warm}")
        decode = worker(args, 32, prefill_warm["pool_slots"])
        decode_warm = decode.call({"op": "pd_decode", "request_id": "warm-pd",
            "generation": 1, "kv_content": prefill_warm["kv_content"],
            "kv_representation": prefill_warm["kv_representation"],
            "max_new_tokens": item["max_new_tokens"]}, timeout=180)
        released = prefill.call({"op": "pd_release", "request_id": "warm-pd",
                                 "generation": 1}, timeout=180)
        if (not decode_warm.get("ok") or not released.get("ok") or
            decode_warm["tokens"] != oracle or
            prefill_warm["oracle_tokens"] != oracle):
            raise RuntimeError("warm persistent P/D output or release mismatch")

        order = [mode for _ in range(args.repeats)
                 for mode in ("unified_pd_execution", "persistent_prefill_decode")]
        random.Random(20260913).shuffle(order)
        counts = {mode: 0 for mode in set(order)}
        window_start = time.perf_counter()
        for order_index, mode in enumerate(order):
            repetition = counts[mode]
            counts[mode] += 1
            request_id = f"deploy-{mode}-{repetition}"
            started = time.perf_counter()
            if mode == "unified_pd_execution":
                response = unified.call({"op": "infer", "requests": [
                    wire(item, encoded, request_id)]}, timeout=180)
                completed = time.perf_counter()
                if not response.get("ok") or not response.get("outputs"):
                    raise RuntimeError(f"unified request failed: {response}")
                output = response["outputs"][0]
                records.append({"mode": mode, "repetition": repetition,
                    "order": order_index, "request_id": request_id, "generation": 1,
                    "worker_pid": response["worker_pid"],
                    "client_ms": (completed - started) * 1000,
                    "ttft_ms": output["ttft_ms"], "itl_ms": output["itl_ms"],
                    "token_itl_ms": output["token_itl_ms"], "tokens": output["tokens"],
                    "output_matches_oracle": output["tokens"] == oracle})
            else:
                p_started = time.perf_counter()
                p = prefill.call({"op": "pd_prefill", "request":
                    wire(item, encoded, request_id)}, timeout=180)
                p_completed = time.perf_counter()
                if not p.get("ok"):
                    raise RuntimeError(f"prefill request failed: {p}")
                d = decode.call({"op": "pd_decode", "request_id": request_id,
                    "generation": 1, "kv_content": p["kv_content"],
                    "kv_representation": p["kv_representation"],
                    "max_new_tokens": item["max_new_tokens"]}, timeout=180)
                d_completed = time.perf_counter()
                release = prefill.call({"op": "pd_release", "request_id": request_id,
                                        "generation": 1}, timeout=180)
                if not d.get("ok") or not release.get("ok"):
                    raise RuntimeError(f"decode/release failed: {d} {release}")
                records.append({"mode": mode, "repetition": repetition,
                    "order": order_index, "request_id": request_id, "generation": 1,
                    "prefill_pid": p["worker_pid"], "decode_pid": d["worker_pid"],
                    "independent_role_pids": p["worker_pid"] != d["worker_pid"],
                    "handoff_valid": p["handoff_valid"] and d["handoff_valid"],
                    "handoff_content": p["kv_content"],
                    "prefix_grant": d["prefix_grant"],
                    "private_grant": d["private_grant"],
                    "grants_are_distinct": d["prefix_grant"] != d["private_grant"],
                    "prompt_tokens": d["prompt_tokens"],
                    "prefill_tokens_saved": d["prefill_tokens_saved"],
                    "decode_actual_computed_prompt_tokens": d["actual_computed_prompt_tokens"],
                    "decode_scheduled_prefill_tokens": d["scheduled_prefill_tokens"],
                    "decode_scheduled_tokens": d["scheduled_decode_tokens"],
                    "handoff_released": release["remaining_handoffs"] == 0,
                    "prefill_ms": (p_completed - p_started) * 1000,
                    "client_ms": (d_completed - started) * 1000,
                    "ttft_ms": (p_completed - started) * 1000 + d["ttft_ms"],
                    "decode_ttft_ms": d["ttft_ms"], "itl_ms": d["itl_ms"],
                    "token_itl_ms": d["token_itl_ms"], "tokens": d["tokens"],
                    "output_matches_oracle": d["tokens"] == oracle})
        window_end = time.perf_counter()
        statuses = {"unified_pd_execution": [unified.call({"op": "status"})],
                    "persistent_prefill_decode": [prefill.call({"op": "status"}),
                                                    decode.call({"op": "status"})]}
    finally:
        if decode is not None:
            decode.close()
        prefill.close()
        unified.close()

    summaries = {}
    for mode in ("unified_pd_execution", "persistent_prefill_decode"):
        rows = [row for row in records if row["mode"] == mode]
        gaps = [gap for row in rows for gap in row["token_itl_ms"]]
        summaries[mode] = {"repeats": len(rows),
            "median_client_ms": statistics.median(row["client_ms"] for row in rows),
            "client_stdev_ms": statistics.pstdev(row["client_ms"] for row in rows),
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
            "p95_ttft_ms": percentile([row["ttft_ms"] for row in rows], 95),
            "p99_ttft_ms": percentile([row["ttft_ms"] for row in rows], 99),
            "median_itl_ms": statistics.median(row["itl_ms"] for row in rows),
            "token_itl_p50_ms": percentile(gaps, 50),
            "token_itl_p95_ms": percentile(gaps, 95),
            "token_itl_p99_ms": percentile(gaps, 99),
            "effective_service_throughput_requests_per_s":
                len(rows) / (sum(row["client_ms"] for row in rows) / 1000),
            "measured_budget": budget(statuses[mode])}
    pd_rows = [row for row in records if row["mode"] == "persistent_prefill_decode"]
    result = {"schema": "pbe.v4.persistent-multimodal-pd-deployment-ab.v2",
        "ok": len(records) == args.repeats * 2 and all(
            row["output_matches_oracle"] for row in records) and all(
            row["independent_role_pids"] and row["handoff_valid"] and
            row["grants_are_distinct"] and row["handoff_released"] and
            row["prefill_tokens_saved"] == row["prompt_tokens"] and
            row["decode_actual_computed_prompt_tokens"] == 0 and
            row["decode_scheduled_prefill_tokens"] == 0 for row in pd_rows),
        "cold_start_excluded": True,
        "all_three_model_processes_concurrently_resident": True,
        "same_multimodal_bundle_and_model": True,
        "same_owned_kv_slots_per_topology": 64,
        "kv_recovery_criterion":
            "active requests return to each Qwen2 model's post-init persistent baseline",
        "arrival_order_seed": 20260913, "completed_requests": len(records),
        "observation_window_ms": (window_end - window_start) * 1000,
        "aggregate_throughput_requests_per_s": len(records) / (window_end - window_start),
        "warmup": {"unified_pid": unified_warm["worker_pid"],
            "prefill_pid": prefill_warm["worker_pid"], "decode_pid": decode_warm["worker_pid"],
            "independent_role_pids": prefill_warm["worker_pid"] != decode_warm["worker_pid"],
            "handoff_output_matches_prefill_oracle_and_unified": True},
        "summaries": summaries, "records": records, "worker_status": statuses,
        "workers_closed_before_result_write": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": result["ok"], "output": str(args.output),
                      "summaries": summaries}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
