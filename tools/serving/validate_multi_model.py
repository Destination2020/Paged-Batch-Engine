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


def stream_timing(port, prompt, model=None):
    """Client wall time to first nonempty SSE text and response completion."""
    start = time.perf_counter()
    arrivals, pieces = [], []
    with urllib.request.urlopen(request(port, prompt, model, stream=True), timeout=120) as response:
        for line in response:
            if line.startswith(b'data: {'):
                event = json.loads(line[6:])
                if model and event['model'] != model:
                    raise AssertionError('timed stream model mismatch')
                piece = event['choices'][0]['text']
                if piece:
                    arrivals.append(time.perf_counter())
                    pieces.append(piece)
    if not arrivals:
        raise AssertionError('empty timed stream')
    return dict(first_text_ms=(arrivals[0] - start) * 1000,
                total_ms=(time.perf_counter() - start) * 1000,
                text_events=len(arrivals), text=''.join(pieces))


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
    parser.add_argument("--reuse-rounds", type=int, default=0,
                        help="alternate real model requests, waiting for full KV return after each")
    parser.add_argument("--timing-rounds", type=int, default=0,
                        help="serial SSE latency samples with private, shared and 93.75%% occupied shared KV")
    args = parser.parse_args()
    if args.reuse_rounds < 0 or args.timing_rounds < 0:
        parser.error('reuse-rounds and timing-rounds must be nonnegative')
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
    timings = []
    def timed_samples(port, model, route, mode):
        for index in range(-1, args.timing_rounds):
            sample = stream_timing(port, prompts[1], route)
            if sample.pop('text').rstrip() != expected[model][1].rstrip():
                raise AssertionError('timed output differs from private baseline')
            if index >= 0:
                timings.append(dict(model=model, mode=mode, **sample))
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
            if args.timing_rounds:
                timed_samples(args.port - 1, model['id'], None, 'private')
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
        registry = json.loads((directory / "domain/registry.json").read_text())
        # Malformed JSON must receive an error while preserving the REP state machine.
        import zmq
        def rejects_malformed(endpoint):
            with zmq.Context() as context:
                with context.socket(zmq.REQ) as channel:
                    channel.setsockopt(zmq.LINGER, 0)
                    channel.setsockopt(zmq.RCVTIMEO, 2000)
                    channel.connect(endpoint)
                    channel.send(b'{')
                    assert not channel.recv_json()['ok']
            assert rpc(endpoint, {'type': 'health'})['ok']
        rejects_malformed(f"ipc://{directory / 'domain/kv.sock'}")
        for worker in registry['models']:
            rejects_malformed(worker['endpoint'])
        # Cancellation must reclaim server-side request records without polling final.
        for worker in registry["models"]:
            for _ in range(10):
                submitted = rpc(worker["endpoint"], dict(type="generate", request=dict(
                    model_id=worker["id"], prompt="Hello", timeout_ms=5000,
                    generation_config=dict(max_new_tokens=32))))
                rpc(worker["endpoint"], dict(type="cancel", request_id=submitted["request_id"],
                                            reason="cancellation_retention_regression"))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                remaining = rpc(worker["endpoint"], {"type": "metrics"})["metrics"]["remote_requests"]
                if remaining == 0:
                    break
                time.sleep(.05)
            else:
                raise AssertionError(f"worker retains cancelled requests: {worker['id']}: {remaining}")
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
                      max_active_requests_per_model=1, baseline_matches=True, utf8_streaming=True, cancellation_route=True, cancelled_request_records_reclaimed=True, malformed_rpc_rejected=True,
                      unknown_model_rejected=True, arena_fully_reclaimed=True, outputs=actual)
        reuse = []
        for round_index in range(args.reuse_rounds):
            for model in config["models"]:
                if completion(args.port, prompts[0], model["id"]) != expected[model["id"]][0]:
                    raise AssertionError("output changed after repeated cross-model pool reuse")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = rpc(endpoint, {"type": "metrics"})
                    if state["lease_count"] == 0 and state["free_bytes"] == state["bytes"]:
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError("KV not returned between alternating models")
                reuse.append(dict(round=round_index, model=model["id"],
                                  free_bytes=state["free_bytes"], lease_count=state["lease_count"]))
        report["alternating_reuse_requests"] = reuse
        if args.timing_rounds:
            for model in config['models']:
                timed_samples(args.port, model['id'], model['id'], 'shared_empty')
            health = rpc(endpoint, {'type': 'health'})
            identity = dict(epoch=health['epoch'], model='timing-capacity-holder')
            rpc(endpoint, dict(type='register', hard_limit=health['bytes'], **identity))
            pressure_bytes = (health['bytes'] * 15 // 16) // 256 * 256
            lease = rpc(endpoint, dict(type='acquire', layout='capacity-only',
                        bytes=pressure_bytes, alignment=256, **identity))['lease']
            try:
                for model in config['models']:
                    timed_samples(args.port, model['id'], model['id'], 'shared_93.75_percent_held')
            finally:
                # The holder never submits GPU work or maps/writes these bytes.
                rpc(endpoint, dict(type='release', lease=lease, gpu_complete=True, **identity))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = rpc(endpoint, {'type': 'metrics'})
                if state['lease_count'] == 0 and state['free_bytes'] == state['bytes']:
                    break
                time.sleep(.05)
            else:
                raise AssertionError('KV not returned after timing')
        report['timings'] = timings
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
