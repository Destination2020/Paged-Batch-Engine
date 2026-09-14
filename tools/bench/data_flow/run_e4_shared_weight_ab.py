#!/usr/bin/env python3
"""E4 real-process persistent P/D private/shared trials and capacity gate."""

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode
from pbe_roles.vision.client import call as vision_call

LAYOUT = "pbe-qwen2-bf16-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_socket(path, process, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(path).is_socket():
            return
        if process.poll() is not None:
            raise RuntimeError(f"process exited before socket {path}: {process.returncode}")
        time.sleep(0.05)
    raise TimeoutError(path)


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def gpu_used_mib(device):
    result = subprocess.run([
        "nvidia-smi", "-i", str(device), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], check=True, text=True,
        stdout=subprocess.PIPE)
    return int(result.stdout.strip().splitlines()[0])


def service_stats(binary, endpoint, kind):
    result = subprocess.run([str(binary), kind, endpoint], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    values = {"exit_code": result.returncode, "raw": result.stdout.strip()}
    for field in result.stdout.strip().split():
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        try:
            values[key] = float(value) if "." in value else int(value)
        except ValueError:
            values[key] = value
    return values


def request(encoded, request_id, max_new_tokens=8):
    return {"request_id": request_id, "generation": 1,
            "content": encoded["content"],
            "representation": encoded["representation"],
            "feature_content": encoded["feature_content"],
            "feature_representation": encoded["feature_representation"],
            "max_new_tokens": max_new_tokens, "timeout_ms": 120000}


def worker_command(args, endpoint, slots, initial, mode):
    command = [str(args.build / "demo/pbe_vlm_language_role"), str(args.model_bin),
               str(args.tokenizer), endpoint, str(args.device), str(64 << 20),
               str(32 << 20), str(slots), "0", "1"]
    if mode == "shared":
        command.extend([",".join(map(str, initial)), "shared", args.model_sha256,
                        LAYOUT])
    elif initial:
        command.append(",".join(map(str, initial)))
    if getattr(args, "compute_sanitizer", False):
        command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
                   "--leak-check", "full", "--report-api-errors", "no", *command]
    return command


class MemoryMonitor:
    def __init__(self, device, output):
        self.device = device
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.output.append({"monotonic_ns": time.monotonic_ns(),
                                    "device": self.device,
                                    "used_mib": gpu_used_mib(self.device)})
            except Exception as error:
                self.output.append({"monotonic_ns": time.monotonic_ns(),
                                    "error": str(error)})
            self.stop_event.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def run_trial(args, experiment, mode, repetition, order):
    trial_name = f"{experiment}-{order:02d}-{mode}-{repetition}"
    trial_dir = args.output.parent / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"/tmp/pbe-e4-data-{os.getpid()}-{order}.sock"
    vision_endpoint = f"/tmp/pbe-e4-vision-{os.getpid()}-{order}.sock"
    blocks = getattr(args, "capacity_blocks", None)
    if blocks is None:
        blocks = 128 if experiment == "fixed_kv" else (64 if mode == "private" else 8192)
    slots = blocks // 2
    service = args.build / "demo/pbe_data_service"
    service_command = [str(service),
        "serve-gpu-weights" if mode == "shared" else "serve-gpu",
        endpoint, str(2 << 30), str(args.device), str(args.layers), str(blocks),
        "16", str(args.kv_heads), str(args.head_size), "4"]
    if mode == "shared":
        service_command.extend([str(args.model_bin), args.model_sha256, LAYOUT,
                                str(64 << 20)])
    service_command.append("4096")
    timeline = []
    baseline_mib = gpu_used_mib(args.device)
    data_log = (trial_dir / "data.log").open("w")
    vision_log = (trial_dir / "vision.log").open("w")
    data_process = subprocess.Popen(service_command, stdout=data_log, stderr=subprocess.STDOUT,
                                    text=True)
    vision_process = None
    prefill = decode = newcomer = None
    record = {"trial": trial_name, "experiment": experiment, "mode": mode,
              "repetition": repetition, "order": order, "kv_blocks": blocks,
              "worker_slots_each": slots, "gpu_baseline_mib": baseline_mib,
              "service_command": service_command}
    try:
        with MemoryMonitor(args.device, timeline):
            wait_socket(endpoint, data_process, 180)
            record["after_data_service_mib"] = gpu_used_mib(args.device)
            vision_command = [str(args.python), "python/pbe_roles/vision/persistent_worker.py",
                "--listen", vision_endpoint, "--endpoint", endpoint, "--model",
                str(args.model_dir), "--device", f"cuda:{args.device}",
                "--batch-window-ms", "20", "--max-batch", "8"]
            environment = dict(os.environ, PYTHONPATH="python")
            vision_process = subprocess.Popen(vision_command, stdout=vision_log,
                                              stderr=subprocess.STDOUT, env=environment,
                                              text=True)
            wait_socket(vision_endpoint, vision_process, 180)
            short_item = {"request_id": "encode-short", "generation": 1,
                "parts": [{"type": "image", "path": str(args.image), "size": 224},
                          {"type": "text", "text":
                           "Describe the image briefly and name its main subject."}],
                "max_new_tokens": 8, "timeout_ms": 120000,
                "_deadline_monotonic_ns": time.monotonic_ns() + 120_000_000_000}
            _, short_encoded = encode(vision_endpoint, short_item)
            if not short_encoded.get("ok"):
                raise RuntimeError(f"vision encode failed: {short_encoded}")
            record["after_vision_mib"] = gpu_used_mib(args.device)
            prefill = LanguageProcess(worker_command(args, endpoint, slots, (), mode))
            record["prefill_ready"] = prefill.ready

            if experiment == "fixed_kv":
                # Generate the deterministic numerical oracle during warm-up,
                # outside the measured serving interval.  Radix reuse is
                # disabled for these workers and the warm handoff is released
                # before the measured request, so no oracle KV survives into
                # the formal sample.
                warm = request(short_encoded, f"{trial_name}-warm", 16)
                p_warm = prefill.call({"op": "pd_prefill", "request": warm,
                                       "oracle_steps": 16}, timeout=180)
                if not p_warm.get("ok"):
                    raise RuntimeError(f"warm prefill failed: {p_warm}")
                decode = LanguageProcess(worker_command(
                    args, endpoint, slots, p_warm["pool_slots"], mode))
                d_warm = decode.call({"op": "pd_decode", "request_id": warm["request_id"],
                    "generation": 1, "kv_content": p_warm["kv_content"],
                    "kv_representation": p_warm["kv_representation"],
                    "expected_provider_incarnation": p_warm["provider_incarnation"],
                    "max_new_tokens": 16}, timeout=180)
                release = prefill.call({"op": "pd_release", "request_id": warm["request_id"],
                                        "generation": 1}, timeout=180)
                if not d_warm.get("ok") or not release.get("ok") or \
                        d_warm["tokens"] != p_warm["oracle_tokens"]:
                    raise RuntimeError("warm P/D handoff mismatch")
                expected_tokens = p_warm["oracle_tokens"]
                measured = request(short_encoded, f"{trial_name}-measured", 16)
                started = time.perf_counter()
                p = prefill.call({"op": "pd_prefill", "request": measured,
                                  "oracle_steps": 0}, timeout=180)
                p_done = time.perf_counter()
                d = decode.call({"op": "pd_decode", "request_id": measured["request_id"],
                    "generation": 1, "kv_content": p["kv_content"],
                    "kv_representation": p["kv_representation"],
                    "expected_provider_incarnation": p["provider_incarnation"],
                    "max_new_tokens": 16}, timeout=180)
                finished = time.perf_counter()
                released = prefill.call({"op": "pd_release",
                    "request_id": measured["request_id"], "generation": 1}, timeout=180)
                record["measurement"] = {"ok": p.get("ok") and d.get("ok") and
                    released.get("ok"), "prefill_pid": p["worker_pid"],
                    "decode_pid": d["worker_pid"],
                    "independent_role_pids": p["worker_pid"] != d["worker_pid"],
                    "handoff_valid": p["handoff_valid"] and d["handoff_valid"],
                    "prompt_tokens": d["prompt_tokens"],
                    "decode_actual_computed_prompt_tokens": d["actual_computed_prompt_tokens"],
                    "decode_scheduled_prefill_tokens": d["scheduled_prefill_tokens"],
                    "prefill_tokens_saved": d["prefill_tokens_saved"],
                    "output_matches_prefill_oracle": d["tokens"] == expected_tokens,
                    "oracle_steps_in_timing": 0,
                    "oracle_in_timing": False,
                    "diagnostics_in_timing": False,
                    "timing_boundary": "pd_prefill RPC send through pd_decode RPC response",
                    "cleanup_boundary": "pd_release and lifecycle probes after response window",
                    "oracle_source": "warm-up pd_prefill fork; compared after timed response",
                    "timed_prefill_oracle_tokens": p["oracle_tokens"],
                    "expected_tokens": expected_tokens,
                    "tokens": d["tokens"], "prefill_ms": (p_done-started)*1000,
                    "wall_ms": (finished-started)*1000, "ttft_ms": d["ttft_ms"] +
                    (p_done-started)*1000, "itl_ms": d["itl_ms"],
                    "token_itl_ms": d["token_itl_ms"]}
            else:
                long_text = "Explain the visual evidence carefully, without guessing. " * 32
                long_items = []
                for index in range(2):
                    item = dict(short_item)
                    item["request_id"] = f"encode-capacity-{index}"
                    item["parts"] = [short_item["parts"][0],
                                     {"type": "text", "text": long_text + str(index)}]
                    item["max_new_tokens"] = 8
                    item["_deadline_monotonic_ns"] = time.monotonic_ns() + 120_000_000_000
                    _, encoded = encode(vision_endpoint, item)
                    if not encoded.get("ok"):
                        raise RuntimeError(f"capacity encode failed: {encoded}")
                    long_items.append(request(encoded, f"{trial_name}-capacity-{index}"))
                started = time.perf_counter()
                p0 = prefill.call({"op": "pd_prefill", "request": long_items[0]}, timeout=180)
                if not p0.get("ok"):
                    raise RuntimeError(f"first capacity prefill failed: {p0}")
                decode = LanguageProcess(worker_command(
                    args, endpoint, slots, p0["pool_slots"], mode))
                p1 = prefill.call({"op": "pd_prefill", "request": long_items[1]}, timeout=180)
                accepted = [p0] + ([p1] if p1.get("ok") else [])
                decoded = []
                for item, publication in zip(long_items, accepted):
                    decoded.append(decode.call({"op": "pd_decode",
                        "request_id": item["request_id"], "generation": 1,
                        "kv_content": publication["kv_content"],
                        "kv_representation": publication["kv_representation"],
                        "expected_provider_incarnation": publication["provider_incarnation"],
                        "max_new_tokens": 8}, timeout=180))
                    prefill.call({"op": "pd_release", "request_id": item["request_id"],
                                  "generation": 1}, timeout=180)
                finished = time.perf_counter()
                prompt_blocks = sum((value["prompt_tokens"] + 15) // 16 for value in accepted)
                record["measurement"] = {"offered": 2, "admitted": len(accepted),
                    "rejected": 2-len(accepted), "second_response": p1,
                    "prompt_blocks_accessed": prompt_blocks,
                    "accessed_slots_beyond_private_32": prompt_blocks > 32,
                    "all_decode_ok": all(value.get("ok") for value in decoded),
                    "wall_ms": (finished-started)*1000,
                    "throughput_requests_per_s": len(decoded)/(finished-started),
                    "tokens": [value.get("tokens") for value in decoded]}

            record["steady_mib"] = gpu_used_mib(args.device)
            record["prefill_status"] = prefill.call({"op": "status"})
            record["decode_status"] = decode.call({"op": "status"})
            record["service_weight_stats"] = service_stats(service, endpoint, "weight-stats") \
                if mode == "shared" else None
            # Lifecycle is outside the timed window: P exits, D keeps computing,
            # then an independent new importer joins the still-ready allocation.
            prefill.close()
            prefill = None
            d_continues = decode.call({"op": "infer", "requests": [
                request(short_encoded, f"{trial_name}-decode-survivor")]}, timeout=180)
            newcomer = LanguageProcess(worker_command(args, endpoint, slots, (), mode))
            n_continues = newcomer.call({"op": "infer", "requests": [
                request(short_encoded, f"{trial_name}-new-importer")]}, timeout=180)
            record["lifecycle"] = {"decode_survived_prefill_exit": d_continues.get("ok"),
                "new_importer_joined": n_continues.get("ok"),
                "decode_pid": d_continues.get("worker_pid"),
                "new_importer_pid": n_continues.get("worker_pid"),
                "weight_stats_after_join": service_stats(service, endpoint, "weight-stats")
                    if mode == "shared" else None}
            newcomer.close(); newcomer = None
            decode.close(); decode = None
            record["ipc_after_workers"] = service_stats(service, endpoint, "ipc-stats")
            if mode == "shared":
                record["weight_after_workers"] = service_stats(service, endpoint, "weight-stats")
        record["startup_peak_mib"] = max(
            (sample["used_mib"] for sample in timeline if "used_mib" in sample),
            default=record["steady_mib"])
    finally:
        for process in (newcomer, decode, prefill):
            if process is not None:
                try: process.close()
                except Exception: process.process.kill()
        if vision_process is not None and vision_process.poll() is None:
            try: vision_call(vision_endpoint, {"op": "shutdown"})
            except Exception: vision_process.kill()
            try: vision_process.wait(timeout=30)
            except subprocess.TimeoutExpired: vision_process.kill()
        if data_process.poll() is None:
            subprocess.run([str(service), "shutdown", endpoint], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: data_process.wait(timeout=30)
        except subprocess.TimeoutExpired: data_process.kill()
        data_log.close(); vision_log.close()
        Path(endpoint).unlink(missing_ok=True)
        Path(vision_endpoint).unlink(missing_ok=True)
    time.sleep(1)
    record["gpu_recovered_mib"] = gpu_used_mib(args.device)
    record["memory_timeline"] = timeline
    (trial_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    return record


def summarize(rows):
    by_key = {}
    for experiment in ("fixed_kv", "fixed_budget"):
        for mode in ("private", "shared"):
            subset = [row for row in rows if row["experiment"] == experiment and row["mode"] == mode]
            values = [row["measurement"]["wall_ms"] for row in subset]
            by_key[f"{experiment}.{mode}"] = {"trials": len(subset),
                "wall_ms_p50": statistics.median(values),
                "wall_ms_p95": percentile(values, 95),
                "steady_mib_p50": statistics.median(row["steady_mib"] for row in subset),
                "startup_peak_mib_p50": statistics.median(row["startup_peak_mib"] for row in subset),
                "throughput_requests_per_s": sum(
                    row["measurement"].get("admitted", 1) for row in subset) /
                    (sum(row["measurement"]["wall_ms"] for row in subset) / 1000),
                "error_rate": sum(row["measurement"].get("rejected", 0) for row in subset) /
                    sum(row["measurement"].get("offered", 1) for row in subset),
                "admitted_total": sum(row["measurement"].get("admitted", 1) for row in subset),
                "rejected_total": sum(row["measurement"].get("rejected", 0) for row in subset)}
    return by_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    args = parser.parse_args()
    root_config = json.loads((args.model_dir / "config.json").read_text())
    config = root_config.get("text_config") or root_config
    args.layers = config["num_hidden_layers"]
    args.kv_heads = config["num_key_value_heads"]
    args.head_size = config["hidden_size"] // config["num_attention_heads"]
    args.model_sha256 = file_sha256(args.model_bin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    order = [(experiment, mode, repetition)
             for experiment in ("fixed_kv", "fixed_budget")
             for repetition in range(args.repeats) for mode in ("private", "shared")]
    random.Random(20260913).shuffle(order)
    rows = []
    for index, (experiment, mode, repetition) in enumerate(order):
        print(json.dumps({"event": "trial_start", "index": index,
                          "experiment": experiment, "mode": mode,
                          "repetition": repetition}), flush=True)
        rows.append(run_trial(args, experiment, mode, repetition, index))
    fixed = [row for row in rows if row["experiment"] == "fixed_kv"]
    capacity_shared = [row for row in rows if row["experiment"] == "fixed_budget" and
                       row["mode"] == "shared"]
    capacity_private = [row for row in rows if row["experiment"] == "fixed_budget" and
                        row["mode"] == "private"]
    shared = [row for row in rows if row["mode"] == "shared"]
    block_bytes = args.layers * 2 * 16 * args.kv_heads * args.head_size * 2
    private_capacity_bytes = 64 * block_bytes
    shared_capacity_bytes = 8192 * block_bytes
    private_budget_steady = statistics.median(row["steady_mib"] for row in capacity_private)
    shared_budget_steady = statistics.median(row["steady_mib"] for row in capacity_shared)
    checks = {
        "repetitions_each_cell": all(sum(row["experiment"] == experiment and row["mode"] == mode
            for row in rows) == args.repeats for experiment in ("fixed_kv", "fixed_budget")
            for mode in ("private", "shared")),
        "real_pd_all": all(row["measurement"].get("independent_role_pids", True) and
            row["measurement"].get("handoff_valid", True) and
            row["measurement"].get("decode_actual_computed_prompt_tokens", 0) == 0
            for row in rows),
        "numeric_oracle_all": all(row["measurement"].get("output_matches_prefill_oracle", True)
                                  for row in rows),
        "one_shared_allocation_upload": all(row["service_weight_stats"]["allocation_count"] == 1 and
            row["service_weight_stats"]["upload_count"] == 1 for row in shared),
        "same_allocation_two_importers": all(row["prefill_status"]["weights"]["allocation_id"] ==
            row["decode_status"]["weights"]["allocation_id"] and
            row["prefill_status"]["weights"]["owner_incarnation"] ==
            row["decode_status"]["weights"]["owner_incarnation"] for row in shared),
        "attention_uses_shared_views": all(row["prefill_status"]["weights"]["attention_views_bound"]
            and row["decode_status"]["weights"]["attention_views_bound"] for row in shared),
        "leases_recovered": all(row.get("weight_after_workers", {}).get("active_leases", 0) == 0
                                for row in shared),
        "lifecycle_survivor_and_join": all(row["lifecycle"]["decode_survived_prefill_exit"] and
            row["lifecycle"]["new_importer_joined"] for row in rows),
        "fixed_kv_same_slots": all(row["kv_blocks"] == 128 for row in fixed),
        "shared_capacity_accesses_new_pages": all(
            row["measurement"]["accessed_slots_beyond_private_32"] for row in capacity_shared),
        "shared_capacity_admits_more": sum(row["measurement"]["admitted"] for row in capacity_shared) >
            sum(row["measurement"]["admitted"] for row in capacity_private),
        "capacity_kv_increment_within_weight_savings":
            shared_capacity_bytes - private_capacity_bytes <= args.model_bin.stat().st_size,
        "fixed_budget_observed_not_higher": shared_budget_steady <= private_budget_steady + 256,
        "cleanup_near_baseline": all(row["gpu_recovered_mib"] <= row["gpu_baseline_mib"] + 256
                                     for row in rows),
    }
    result = {"schema": "pbe-e4-shared-weight-ab-v1", "ok": all(checks.values()),
              "random_order_seed": 20260913, "model_sha256": args.model_sha256,
              "layout_identity": LAYOUT, "checks": checks, "summary": summarize(rows),
              "physical_accounting": {"weight_file_bytes": args.model_bin.stat().st_size,
                  "private_capacity_kv_bytes": private_capacity_bytes,
                  "shared_capacity_kv_bytes": shared_capacity_bytes,
                  "kv_increment_bytes": shared_capacity_bytes-private_capacity_bytes,
                  "private_capacity_steady_mib_p50": private_budget_steady,
                  "shared_capacity_steady_mib_p50": shared_budget_steady},
              "records": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"event": "complete", "ok": result["ok"],
                      "checks": checks, "output": str(args.output)}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
