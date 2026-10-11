#!/usr/bin/env python3
"""Compare each model's private KV baseline with concurrent shared-arena serving.

This validates PBE output preservation, routing, streaming and cancellation.
It does not establish Transformers parity or MPS throughput improvements.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from deploy_multi_model import ROOT, Domain, rpc, validate_config


def request(port, prompt, model=None, stream=False, tokens=24):
    payload = dict(prompt=prompt, max_tokens=tokens, temperature=0, stream=stream)
    if model:
        payload["model"] = model
    return urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})


def completion(port, prompt, model=None):
    with urllib.request.urlopen(request(port, prompt, model), timeout=120) as response:
        result = json.load(response)
    if model and result["model"] != model:
        raise AssertionError("response model identity mismatch")
    return result["choices"][0]["text"]


def ready_http(port, process, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("HTTP process exited before readiness")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("HTTP readiness timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mps")
    parser.add_argument("--backend", choices=("process", "mps_process"), default="mps_process")
    parser.add_argument("--start-mps", action="store_true")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--serial", action="store_true", help="diagnostic serial request submission")
    parser.add_argument("--test-worker-failure", action="store_true",
                        help="kill one worker after correctness checks and verify entire GPU domain exits")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="pbe-validate-"))
    directory.mkdir(exist_ok=True, mode=0o700)
    env = os.environ.copy(); env["KUIPER_USE_FP8_KV_CACHE"] = "0"
    # Sequential private baselines are outside the measured MPS deployment.
    domain = Domain(directory, env)
    prompts = ["请用一句话介绍中国的首都。", "Write a short Python function to add two numbers.",
               "请总结以下内容：" + "今天的会议讨论了共享显存和并发推理。" * 24]
    # Keep numerical batch shapes identical for exact output preservation.
    # Concurrent requests still queue, and different model workers run together.
    for model in config["models"]:
        model["max_active_requests"] = 1
    expected = {}
    supervisor = None
    try:
        for model in config["models"]:
            command = [args.build_dir / "demo/serving_qwen", model["weights"], model["tokenizer"],
                       "--online-server=1", f"--listen-port={args.port - 1}", "--kv-cache-budget-mib=64",
                       "--radix-cache=off", "--warmup-rounds=1", "--final-summary=0", "--max-active-requests=1",
                       f"--max-batched-tokens={model.get('max_batched_tokens', 128)}",
                       f"--prefill-chunk-cap={model.get('prefill_chunk_cap', 128)}"]
            baseline = domain.spawn("baseline-" + model["id"], command)
            ready_http(args.port - 1, baseline)
            expected[model["id"]] = [completion(args.port - 1, prompt) for prompt in prompts]
            os.killpg(baseline.pid, signal.SIGTERM); baseline.wait(timeout=15)
            domain.children.remove(("baseline-" + model["id"], baseline))
        config.update(backend=args.backend, listen_port=args.port, listen_host="127.0.0.1")
        resolved_config = directory / "deployment.json"
        resolved_config.write_text(json.dumps(config, indent=2))
        command = [os.sys.executable, ROOT / "tools/serving/deploy_multi_model.py",
                   "--config", resolved_config, "--build-dir", args.build_dir,
                   "--runtime-dir", directory / "domain"]
        if args.start_mps:
            command.append("--start-mps")
        supervisor = domain.spawn("supervisor", command)
        ready_http(args.port, supervisor)
        actual = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1 if args.serial else 6) as executor:
            tasks = {(model["id"], index): executor.submit(completion, args.port, prompt, model["id"])
                     for model in config["models"] for index, prompt in enumerate(prompts)}
            for model in config["models"]:
                actual[model["id"]] = [tasks[(model["id"], i)].result() for i in range(len(prompts))]
        if actual != expected:
            raise AssertionError(json.dumps(dict(expected=expected, actual=actual), ensure_ascii=False))
        # Stream only complete UTF-8 JSON chunks and verify the model tag.
        first = config["models"][0]["id"]
        pieces = []
        with urllib.request.urlopen(request(args.port, prompts[0], first, stream=True), timeout=120) as response:
            for line in response:
                if line.startswith(b"data: {"):
                    event = json.loads(line[6:])
                    assert event["model"] == first
                    pieces.append(event["choices"][0]["text"])
        if "".join(pieces).rstrip() != expected[first][0].rstrip():
            raise AssertionError("streamed content differs from baseline")
        # A disconnected A stream must leave B routable and free its KV.
        with urllib.request.urlopen(request(args.port, prompts[1], first, stream=True, tokens=128), timeout=120) as response:
            response.readline()
        second = config["models"][1]["id"]
        if completion(args.port, prompts[0], second) != expected[second][0]:
            raise AssertionError("cancellation was routed to the wrong model")
        try:
            completion(args.port, "hello", "unknown-model")
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
        else:
            raise AssertionError("unknown model accepted")
        endpoint = f"ipc://{directory / 'domain/kv.sock'}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = rpc(endpoint, {"type": "metrics"})
            if state["lease_count"] == 0 and state["free_bytes"] == state["bytes"]:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("KV arena not fully reclaimed after generation/cancellation")
        report = dict(backend=args.backend, model_ids=list(expected),
                      different_weight_paths=len({model["weights"] for model in config["models"]}) > 1,
                      max_active_requests_per_model=1, baseline_matches=True, utf8_streaming=True, cancellation_route=True,
                      unknown_model_rejected=True, arena_fully_reclaimed=True, outputs=actual)
        if args.test_worker_failure:
            registry = json.loads((directory / "domain/registry.json").read_text())
            os.killpg(registry["models"][0]["worker_pid"], signal.SIGKILL)
            if supervisor.wait(timeout=30) == 0:
                raise AssertionError("worker failure did not fail the supervisor")
            pids = [item["worker_pid"] for item in registry["models"]] + [registry["arena_owner_pid"]]
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise AssertionError(f"GPU domain process survived worker failure: {pid}")
            report["worker_failure_stops_gpu_domain"] = True
        (directory / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        domain.stop()
        print(f"Validation artifacts: {directory}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
