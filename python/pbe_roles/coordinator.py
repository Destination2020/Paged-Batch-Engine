#!/usr/bin/env python3
"""Typed Encode -> Join -> PBE Prefill/Decode coordinator for the V4 demo."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from pbe_roles.vision.client import call as vision_call


class LanguageProcess:
    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        ready = self._read()
        if ready.get("event") != "ready":
            raise RuntimeError(f"language role did not become ready: {ready}")
        self.ready = ready
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._next_op_id = 1
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"language role exited with {self.process.poll()}")
            # Wrappers such as compute-sanitizer may emit banners on stdout.
            # Preserve them in the raw log but only feed protocol JSON to the
            # asynchronous response dispatcher.
            if line.lstrip().startswith("{"):
                return json.loads(line)
            print(line.rstrip(), file=sys.stderr)

    def _read_loop(self):
        try:
            while True:
                response = self._read()
                op_id = response.get("op_id")
                with self._pending_lock:
                    future = self._pending.pop(op_id, None)
                if future is not None:
                    future.set_result(response)
        except Exception as error:
            with self._pending_lock:
                pending = list(self._pending.values())
                self._pending.clear()
            for future in pending:
                future.set_exception(error)

    def send(self, request: dict) -> concurrent.futures.Future:
        future = concurrent.futures.Future()
        with self._write_lock:
            op_id = self._next_op_id
            self._next_op_id += 1
            payload = dict(request)
            payload["op_id"] = op_id
            with self._pending_lock:
                self._pending[op_id] = future
            try:
                self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
            except Exception:
                with self._pending_lock:
                    self._pending.pop(op_id, None)
                raise
        return future

    def call(self, request: dict, timeout: float = 120.0) -> dict:
        return self.send(request).result(timeout=timeout)

    def close(self):
        if self.process.poll() is None:
            self.call({"op": "shutdown"})
        if self.process.wait(timeout=30) != 0:
            raise RuntimeError("language role shutdown failed")
        self._reader.join(timeout=1)


def encode(vision_endpoint: str, item: dict) -> tuple[dict, dict]:
    remaining_ms = max(1, (item["_deadline_monotonic_ns"] - time.monotonic_ns()) // 1_000_000)
    request = {
        "op": "encode", "request_id": item["request_id"],
        "generation": item.get("generation", 1),
        "timeout_ms": remaining_ms,
        "parts": item["parts"],
    }
    return item, vision_call(vision_endpoint, request)


def run_round(language: LanguageProcess, vision_endpoint: str,
              items: list[dict], trace: list[dict]) -> dict:
    round_started = time.perf_counter()
    round_started_ns = time.monotonic_ns()
    if items and items[0].get("demote_prefix_before"):
        demotion = language.call({"op": "demote_prefix"})
        trace.append({"stage": "tiered_cache_demotion", "result": demotion})
        if not demotion.get("ok"):
            return {"ok": False, "error": "tiered_cache_demotion_failed"}
    items = [dict(item, _deadline_monotonic_ns=time.monotonic_ns() +
                  int(item.get("timeout_ms", 30000)) * 1_000_000)
             for item in items]
    stages = {item["request_id"]: "encode" for item in items}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as pool:
        encoded = list(pool.map(lambda item: encode(vision_endpoint, item), items))
    encode_elapsed_ms = (time.perf_counter() - round_started) * 1000.0
    joined = []
    for item, result in encoded:
        request_id = item["request_id"]
        trace.append({"request_id": request_id, "stage": "encode", "result": result})
        if not result.get("ok"):
            stages[request_id] = "failed"
            continue
        if time.monotonic_ns() >= item["_deadline_monotonic_ns"]:
            stages[request_id] = "failed"
            trace.append({"request_id": request_id, "stage": "join",
                          "result": {"ok": False, "error": "deadline_exceeded"}})
            continue
        stages[request_id] = "join"
        if item.get("cancel_stage") == "join":
            stages[request_id] = "cancelled"
            trace.append({"request_id": request_id, "stage": "join",
                          "result": {"ok": False, "error": "cancelled"}})
            continue
        joined.append({
            "request_id": request_id,
            "generation": item.get("generation", 1),
            "content": result["content"],
            "representation": result["representation"],
            "feature_content": result["feature_content"],
            "feature_representation": result["feature_representation"],
            "max_new_tokens": item.get("max_new_tokens", 16),
            "cancel_after_tokens": item.get("cancel_after_tokens", 0),
            "timeout_ms": item.get("language_timeout_ms", 30000),
            "deadline_monotonic_ns": item["_deadline_monotonic_ns"],
            "diagnostic_key": item.get("diagnostic_key", ""),
            "diagnostic_mode": item.get("diagnostic_mode", ""),
            "diagnostic_dump_logits_step": item.get("diagnostic_dump_logits_step", -1),
        })
        stages[request_id] = "prefill"
    if not joined:
        return {"ok": True, "outputs": [], "stages": stages,
                "coordinator_started_ns": round_started_ns,
                "coordinator_terminal_ns": time.monotonic_ns()}
    infer_future = language.send({"op": "infer", "requests": joined})
    cancel_results = []
    cancel_threads = []
    def external_cancel(request):
        time.sleep(request["external_cancel_after_ms"] / 1000.0)
        ack = language.call({"op": "cancel", "request_id": request["request_id"],
                             "generation": request.get("generation", 1)})
        cancel_results.append(ack)
    for item in items:
        if item.get("external_cancel_after_ms") is not None:
            thread = threading.Thread(target=external_cancel, args=(item,))
            thread.start()
            cancel_threads.append(thread)
        if item.get("external_checkpoint_after_ms") is not None:
            def external_checkpoint(request=item):
                time.sleep(request["external_checkpoint_after_ms"] / 1000.0)
                ack = language.call({"op": "checkpoint",
                                     "request_id": request["request_id"],
                                     "generation": request.get("generation", 1)})
                cancel_results.append(ack)
            thread = threading.Thread(target=external_checkpoint)
            thread.start()
            cancel_threads.append(thread)
    result = infer_future.result(timeout=120)
    for thread in cancel_threads:
        thread.join()
    if cancel_results:
        trace.append({"stage": "external_lifecycle_control", "results": cancel_results})
    result["encode_join_ms"] = encode_elapsed_ms
    result["round_latency_ms"] = (time.perf_counter() - round_started) * 1000.0
    for output in result.get("outputs", []):
        output["multimodal_ttft_ms"] = encode_elapsed_ms + output["ttft_ms"]
    trace.append({"stage": "language", "request_ids": [x["request_id"] for x in joined],
                  "result": result})
    for output in result.get("outputs", []):
        request_id = output.get("request_id")
        stages[request_id] = "cancelled" if output.get("failed") else "finished"
    for rejection in result.get("rejections", []):
        stages[rejection["request_id"]] = "rejected"
    result["stages"] = stages
    result["coordinator_started_ns"] = round_started_ns
    result["coordinator_terminal_ns"] = time.monotonic_ns()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-endpoint", required=True)
    parser.add_argument("--language-binary", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-endpoint", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--bundle-capacity", type=int, default=64 << 20)
    parser.add_argument("--staging-capacity", type=int, default=32 << 20)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--weight-mode", choices=("private", "shared"), default="private")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--weight-layout", default="")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    trace = [{"event": "coordinator_start", "monotonic_ns": time.monotonic_ns()}]
    language_command = [
        str(args.language_binary), str(args.model_bin), str(args.tokenizer),
        args.data_endpoint, str(args.device), str(args.bundle_capacity),
        str(args.staging_capacity),
        "0", "0" if args.disable_radix_cache else "1",
    ]
    if args.weight_mode == "shared":
        if len(args.model_sha256) != 64 or not args.weight_layout:
            parser.error("shared mode requires --model-sha256 and --weight-layout")
        language_command.extend(["1", "", "shared", args.model_sha256,
                                 args.weight_layout])
    language = LanguageProcess(language_command)
    trace.append(language.ready)
    results = []
    try:
        for round_items in spec["rounds"]:
            results.append(run_round(language, args.vision_endpoint, round_items, trace))
        status = language.call({"op": "status"})
        trace.append({"event": "language_status", "result": status})
    finally:
        language.close()
    output = {"ok": all(item.get("ok") for item in results),
              "rounds": results, "status": status}
    trace.append({"event": "coordinator_finish", "result": output,
                  "monotonic_ns": time.monotonic_ns()})
    args.trace.write_text("\n".join(json.dumps(item, sort_keys=True) for item in trace) + "\n")
    print(json.dumps(output, sort_keys=True))
    if not output["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
