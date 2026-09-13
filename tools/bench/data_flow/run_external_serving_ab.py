#!/usr/bin/env python3
"""Five-repeat fair serving A/B for process-owned vs Agent-owned KV."""
from __future__ import annotations

import argparse
import json
import random
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path


def parse_summary(text: str) -> dict:
    line = next(line for line in text.splitlines()
                if line.startswith("FINAL_SUMMARY "))
    fields = dict(re.findall(r"([a-zA-Z0-9_]+)=([^ ]+)", line))
    keys = ("throughput_tps", "ttft_ms", "itl_ms", "itl_p95_ms",
            "latency_ms", "wall_ms")
    return {key: float(fields[key]) for key in keys} | {
        "completed_requests": int(fields["completed_requests"]),
        "failed_requests": int(fields["failed_requests"]),
        "total_prefill_tokens": int(fields["total_prefill_tokens"]),
        "total_decode_tokens": int(fields["total_decode_tokens"]),
    }


def wait_socket(path: Path, process: subprocess.Popen):
    for _ in range(500):
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError("data service exited before ready")
        time.sleep(0.01)
    raise TimeoutError("data service socket was not ready")


def call_service(binary: Path, *args: str):
    return subprocess.run([str(binary), *map(str, args)], check=True,
                          text=True, capture_output=True)


def median(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    layers = text_config["num_hidden_layers"]
    kv_heads = text_config["num_key_value_heads"]
    head_size = text_config["hidden_size"] // text_config["num_attention_heads"]
    blocks = 64
    service_binary = args.build / "demo" / "pbe_data_service"
    serving_binary = args.build / "demo" / "serving_qwen"
    common = [str(serving_binary), str(args.model_bin), str(args.tokenizer),
              "What is paged attention?",
              "Give one benefit of continuous batching.",
              f"--device-id={args.device}",
              f"--kv-cache-blocks-per-layer={blocks}",
              "--max-new-tokens=16", "--max-batched-tokens=128",
              "--prefill-chunk-cap=64", "--radix-cache=off",
              "--warmup-rounds=0", "--quiet=1", "--step-trace=0"]
    order = [mode for mode in ("local", "external")
             for _ in range(args.repeats)]
    random.Random(20260913).shuffle(order)
    counters = {"local": 0, "external": 0}
    records = []
    for ordinal, mode in enumerate(order):
        repeat = counters[mode]
        counters[mode] += 1
        service = None
        endpoint = Path(f"/tmp/pbe-serving-ab-{mode}-{ordinal}-{time.time_ns()}.sock")
        command = list(common)
        service_log = None
        try:
            if mode == "external":
                service_log = (raw / f"{ordinal:02d}_{mode}_{repeat}_service.log").open("w")
                service = subprocess.Popen([
                    str(service_binary), "serve-gpu", str(endpoint),
                    str(2 << 30), str(args.device), str(layers), str(blocks),
                    "16", str(kv_heads), str(head_size), "4", "512"],
                    stdout=service_log, stderr=subprocess.STDOUT, text=True)
                wait_socket(endpoint, service)
                command.append(f"--data-service-endpoint={endpoint}")
            started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True)
            process_ms = (time.perf_counter() - started) * 1000.0
            log = completed.stdout + completed.stderr
            (raw / f"{ordinal:02d}_{mode}_{repeat}.log").write_text(log)
            if completed.returncode != 0:
                raise RuntimeError(f"{mode} serving failed: {completed.returncode}")
            result = parse_summary(log)
            result.update(mode=mode, repeat=repeat, ordinal=ordinal,
                          process_ms=process_ms)
            if mode == "external":
                if "PBE_SERVING_EXTERNAL_KV_BOUND" not in log or \
                        "PBE_SERVING_EXTERNAL_KV_RELEASED" not in log:
                    raise RuntimeError("external lifecycle marker missing")
                stats = call_service(service_binary, "ipc-stats", endpoint).stdout.strip()
                if "free_slots=64 active_grants=0" not in stats:
                    raise RuntimeError(f"external pool did not recover: {stats}")
                result["final_pool_stats"] = stats
            records.append(result)
        finally:
            if service is not None:
                subprocess.run([str(service_binary), "shutdown", str(endpoint)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                service.wait(timeout=10)
            if service_log is not None:
                service_log.close()

    grouped = {mode: [record for record in records if record["mode"] == mode]
               for mode in ("local", "external")}
    summary = {}
    for mode, rows in grouped.items():
        summary[mode] = {
            key: median([row[key] for row in rows])
            for key in ("throughput_tps", "ttft_ms", "itl_ms",
                        "itl_p95_ms", "latency_ms", "wall_ms", "process_ms")
        }
    output = {
        "date": "2026-09-13", "seed": 20260913,
        "repeats_per_mode": args.repeats, "randomized_order": order,
        "controls": {"blocks_per_layer": blocks, "max_new_tokens": 16,
                     "max_batched_tokens": 128, "prefill_chunk_cap": 64,
                     "requests": 2, "device": args.device,
                     "model": str(args.model_bin)},
        "records": records, "median": summary,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
