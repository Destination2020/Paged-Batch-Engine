#!/usr/bin/env python3
"""Real two-worker VLM coordinator with fixed/RR/data-aware policies."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode
from pbe_roles.placement import WorkerSnapshot, choose


class Pool:
    def __init__(self, args):
        self.args = args
        self.workers = [self._start() for _ in range(2)]
        self.queued = [0, 0]
        self.inflight = [0, 0]
        self.rr = 0

    def _start(self):
        return LanguageProcess([
            str(self.args.language_binary), str(self.args.model_bin),
            str(self.args.tokenizer), self.args.data_endpoint, str(self.args.device),
            str(64 << 20), str(32 << 20), str(self.args.kv_slots_per_worker)])

    def restart(self, index):
        old_pid = self.workers[index].ready["worker_pid"]
        self.workers[index].close()
        self.workers[index] = self._start()
        self.queued[index] = self.inflight[index] = 0
        return {"worker": index, "old_pid": old_pid,
                "new_pid": self.workers[index].ready["worker_pid"]}

    def probe(self, item, encoded):
        request = {"content": encoded["content"],
                   "representation": encoded["representation"],
                   "feature_content": encoded["feature_content"],
                   "feature_representation": encoded["feature_representation"],
                   "max_new_tokens": item.get("max_new_tokens", 16)}
        return [worker.call({"op": "probe", "request": request})
                for worker in self.workers]

    def snapshots(self, probes, stale_worker=None, unavailable_worker=None):
        now = time.monotonic_ns()
        snapshots = []
        for index, worker in enumerate(self.workers):
            observed = now - 5_000_000_000 if index == stale_worker else now
            probe = probes[index]
            capacity = probe["capacity"]
            required = probe["required"]
            prefix = probe["prefix"]
            admissible = bool(probe["admissible"]) and index != unavailable_worker
            snapshots.append(WorkerSnapshot(
                index, str(worker.ready["worker_pid"]), "qwen25-vl-3b-bf16-v1",
                "qwen25-vl-request-bundle-v3", observed, self.queued[index],
                self.inflight[index], admissible,
                capacity["kv_free_bytes"], capacity["bundle_free_bytes"],
                capacity["staging_free_bytes"], prefix["missing_bytes"],
                self.args.bandwidth_bytes_per_ms,
                self.args.prefill_ms_per_token, self.args.decode_ms_per_token, 0.1))
        return snapshots

    def select(self, policy, item, encoded, probes=None, excluded=None):
        if "force_worker" in item:
            return item["force_worker"], {"policy": policy, "forced": True}
        if policy == "fixed":
            return 0, {"policy": policy, "worker": 0}
        if policy == "round_robin":
            selected = self.rr % 2
            self.rr += 1
            return selected, {"policy": policy, "worker": selected}
        decision = choose(
            self.snapshots(probes, item.get("stale_worker"), item.get("unavailable_worker")),
            model_revision="qwen25-vl-3b-bf16-v1",
            representation="qwen25-vl-request-bundle-v3",
            required_bytes=encoded["bundle_bytes"], prefix_key=encoded["content"],
            prefix_bytes=encoded["bundle_bytes"],
            prefill_tokens=encoded["feature_rows"] + 32,
            decode_tokens=item.get("max_new_tokens", 16), excluded=excluded)
        if not decision["ok"]:
            raise RuntimeError("placement failed: " + json.dumps(decision))
        return decision["worker"], {"policy": policy, **decision}

    def submit(self, worker_index, item, encoded):
        deadline = item["_deadline_monotonic_ns"]
        request = {"request_id": item["request_id"],
                   "generation": item.get("generation", 1),
                   "content": encoded["content"],
                   "representation": encoded["representation"],
                   "feature_content": encoded["feature_content"],
                   "feature_representation": encoded["feature_representation"],
                   "max_new_tokens": item.get("max_new_tokens", 16),
                   "deadline_monotonic_ns": deadline,
                   "timeout_ms": max(1, (deadline - time.monotonic_ns()) // 1_000_000)}
        tokens = encoded["feature_rows"] + 32 + request["max_new_tokens"]
        queued_at = time.perf_counter()
        future = self.workers[worker_index].send({"op": "infer", "requests": [request]})
        completion = {}
        future.add_done_callback(lambda _: completion.setdefault("at", time.perf_counter()))
        self.queued[worker_index] += tokens
        self.inflight[worker_index] += 1
        return future, tokens, queued_at, completion

    def complete(self, worker, tokens, future, queued_at, completion):
        result = future.result(timeout=180)
        self.queued[worker] -= tokens
        self.inflight[worker] -= 1
        completed_at = completion.get("at", time.perf_counter())
        return result, (completed_at - queued_at) * 1000, completed_at

    def close(self):
        statuses = [worker.call({"op": "status"}) for worker in self.workers]
        for worker in self.workers:
            worker.close()
        return statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-endpoint", required=True)
    parser.add_argument("--language-binary", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-endpoint", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kv-slots-per-worker", type=int, default=32)
    parser.add_argument("--policy", choices=("fixed", "round_robin", "data_aware"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bandwidth-bytes-per-ms", type=float, default=4096.0)
    parser.add_argument("--prefill-ms-per-token", type=float, default=1.5)
    parser.add_argument("--decode-ms-per-token", type=float, default=45.0)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())["placement_run_configuration"]
        args.bandwidth_bytes_per_ms = calibration["bandwidth_bytes_per_ms"]
        args.prefill_ms_per_token = calibration["prefill_ms_per_token"]
        args.decode_ms_per_token = calibration["decode_ms_per_token"]
    random.seed(args.seed)
    spec = json.loads(args.spec.read_text())
    pool = Pool(args)
    trace, latencies, outputs, restarts = [], [], [], []
    observation_start = None
    observation_end = None
    try:
        for round_items in spec["rounds"]:
            for item in round_items:
                if "restart_worker_before" in item:
                    restarts.append(pool.restart(item["restart_worker_before"]))
            stamped = [dict(item, _deadline_monotonic_ns=time.monotonic_ns() +
                            int(item.get("timeout_ms", 120000)) * 1_000_000)
                       for item in round_items]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(stamped)) as executor:
                pairs = list(executor.map(lambda value: encode(args.vision_endpoint, value), stamped))
            if any(not result.get("ok") for _, result in pairs):
                raise RuntimeError("vision encode failed")
            if round_items and round_items[0].get("shuffle_dispatch"):
                random.shuffle(pairs)
            probes = [pool.probe(item, encoded) if args.policy == "data_aware" else None
                      for item, encoded in pairs]
            pending = []
            for (item, encoded), worker_probes in zip(pairs, probes):
                started = time.perf_counter()
                worker, decision = pool.select(args.policy, item, encoded, worker_probes)
                decision["decision_us"] = (time.perf_counter() - started) * 1e6
                decision["request_id"] = item["request_id"]
                future, tokens, queued_at, completion = pool.submit(worker, item, encoded)
                observation_start = queued_at if observation_start is None else min(
                    observation_start, queued_at)
                pending.append((worker, tokens, future, queued_at, completion, item, decision))
            for worker, tokens, future, queued_at, completion, item, decision in pending:
                result, client_ms, completed_at = pool.complete(
                    worker, tokens, future, queued_at, completion)
                rejection_reasons = {x.get("reason") for x in result.get("rejections", [])}
                if (args.policy == "data_aware" and not result.get("outputs") and
                        rejection_reasons & {"unified_budget_exhausted",
                                             "device_memory_admission_exhausted"} and
                        time.monotonic_ns() < item["_deadline_monotonic_ns"]):
                    retry_probes = pool.probe(item, next(
                        encoded for candidate, encoded in pairs
                        if candidate["request_id"] == item["request_id"]))
                    retry_worker, retry_decision = pool.select(
                        args.policy, item, next(encoded for candidate, encoded in pairs
                            if candidate["request_id"] == item["request_id"]),
                        retry_probes, {worker})
                    encoded_retry = next(encoded for candidate, encoded in pairs
                        if candidate["request_id"] == item["request_id"])
                    retry_future, retry_tokens, retry_at, retry_completion = pool.submit(
                        retry_worker, item, encoded_retry)
                    result, retry_ms, completed_at = pool.complete(
                        retry_worker, retry_tokens, retry_future, retry_at, retry_completion)
                    decision["bounded_retry"] = {"from_worker": worker,
                        "to_worker": retry_worker, "decision": retry_decision,
                        "first_rejections": sorted(rejection_reasons)}
                    worker = retry_worker
                    client_ms += retry_ms
                observation_end = completed_at if observation_end is None else max(
                    observation_end, completed_at)
                output = result["outputs"][0] if result.get("outputs") else {}
                actual_compute = result.get("batch_latency_ms", 0.0)
                request_ok = bool(result.get("ok") and output and
                                  not output.get("failed", False))
                record = {"request_id": item["request_id"], "worker": worker,
                          "worker_pid": result.get("worker_pid"),
                          "decision": decision, "client_language_ms": client_ms,
                          "actual_compute_ms": actual_compute,
                          "actual_queue_wait_ms": max(0.0, client_ms - actual_compute),
                          "prediction_error_ms": client_ms - decision.get("predicted_ms", client_ms),
                          "output": output, "result_ok": request_ok}
                trace.append(record); latencies.append(client_ms); outputs.append(output)
    finally:
        statuses = pool.close()
    result = {"ok": all(x["result_ok"] for x in trace), "policy": args.policy,
              "seed": args.seed, "trace": trace, "restarts": restarts,
              "worker_status": statuses, "requests": len(trace),
              "mean_language_ms": statistics.mean(latencies),
              "p95_language_ms": sorted(latencies)[max(0, int(.95 * len(latencies)) - 1)],
              "throughput_requests_per_s": len(latencies) / max(
                  1e-9, observation_end - observation_start),
              "throughput_definition": "completed_requests / dispatch-to-last-completion wall window",
              "observation_window_ms": (observation_end - observation_start) * 1000,
              "rejected": sum(not x["result_ok"] for x in trace)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
