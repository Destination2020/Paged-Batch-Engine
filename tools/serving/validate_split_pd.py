#!/usr/bin/env python3
"""Real-weight 2P1D correctness, mixed long-output load, cancellation and reclaim."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from deploy_multi_model import ROOT, rpc
from validate_multi_model import ready_http


def generate(port, prompt, tokens=24, ignore_eos=False):
    start = time.monotonic()
    payload = dict(
        prompt=prompt, max_tokens=tokens, temperature=0, stream=True, ignore_eos=ignore_eos
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    pieces, first = [], None
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if line.startswith(b"data: {"):
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(event)
                text = event["choices"][0].get("text", "")
                if text:
                    if first is None:
                        first = time.monotonic()
                    pieces.append(text)
    end = time.monotonic()
    if first is None:
        raise AssertionError("No output text")
    return dict(
        text="".join(pieces),
        start_s=start,
        end_s=end,
        first_text_ms=(first - start) * 1000,
        total_ms=(end - start) * 1000,
        requested_output_tokens=tokens,
    )


def wait_reclaimed(endpoint, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = rpc(endpoint, {"type": "metrics"})
        if state["free_bytes"] == state["bytes"] and state["lease_count"] == 0:
            return state
        time.sleep(0.05)
    raise AssertionError(f"KV not reclaimed: {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["process", "mps_process"], default="mps_process")
    parser.add_argument("--extreme-output-tokens", type=int, default=4096)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 256 <= args.extreme_output_tokens <= config["max_output_tokens"]:
        parser.error("Extreme output must be between 256 and the deployment output limit")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, mode=0o700)
    port = config.get("listen_port", 8080)
    env = os.environ.copy()
    env.update(KUIPER_USE_FP8_KV_CACHE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    baseline = supervisor = None
    logs = []
    prompts = [
        "请用一句话介绍中国的首都。",
        "Write a short Python function to add two numbers.",
        "会议讨论了共享显存和并发推理。" * 192 + "请用一句话总结。",
    ]
    report = dict(backend=args.backend, model=config["model_id"], topology="2P1D")
    try:
        log = (output / "baseline.log").open("w")
        logs.append(log)
        baseline = subprocess.Popen(
            [
                str(args.build_dir / "demo/serving_qwen"),
                config["weights"],
                config["tokenizer"],
                "--online-server=1",
                f"--listen-port={port-1}",
                "--kv-cache-budget-mib=512",
                "--radix-cache=off",
                "--max-batched-tokens=128",
                "--prefill-chunk-cap=128",
                "--max-active-requests=1",
                "--final-summary=0",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready_http(port - 1, baseline)
        expected = [generate(port - 1, prompt)["text"] for prompt in prompts]
        os.killpg(baseline.pid, signal.SIGTERM)
        baseline.wait(timeout=30)
        baseline = None
        log = (output / "supervisor.log").open("w")
        logs.append(log)
        supervisor = subprocess.Popen(
            [
                "python3",
                str(ROOT / "tools/serving/deploy_split_pd.py"),
                "--config",
                str(args.config.resolve()),
                "--build-dir",
                str(args.build_dir.resolve()),
                "--runtime-dir",
                str(output / "domain"),
                "--backend",
                args.backend,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready_http(port, supervisor)
        registry = json.loads((output / "domain/registry.json").read_text())
        endpoint = next(w["endpoint"] for w in registry["workers"] if w["role"] == "decode")
        owner = f"ipc://{output}/domain/kv.sock"
        actual = [generate(port, prompt)["text"] for prompt in prompts]
        if actual != expected:
            raise AssertionError(dict(expected=expected, actual=actual))
        report.update(private_baseline_matches=True, outputs=actual, registry=registry)
        report["after_correctness"] = wait_reclaimed(owner)
        print("Private baseline and split P/D outputs match", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            big = [
                pool.submit(
                    generate,
                    port,
                    "请继续逐条编号，详细解释这个主题。"
                    + "会议讨论了共享显存和并发推理。" * repeat,
                    args.extreme_output_tokens,
                    True,
                )
                for repeat in (700, 500)
            ]
            big.append(
                pool.submit(
                    generate,
                    port,
                    "从1开始依次编号，持续列出计算机科学知识。",
                    args.extreme_output_tokens,
                    True,
                )
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                metrics = rpc(endpoint, {"type": "metrics"})["metrics"]
                split = metrics.get("split_pd", {})
                if (
                    split.get("long", {}).get("active", 0) >= 2
                    and split.get("decode_steps", 0) >= 32
                ):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("Mixed long load was not admitted")
            short = [pool.submit(generate, port, prompts[i % 2], 24) for i in range(3)]
            short_results = [future.result() for future in short]
            big_results = [future.result() for future in big]
        if max(r["end_s"] for r in short_results) >= min(r["end_s"] for r in big_results):
            raise AssertionError(
                "Short requests did not finish while extreme outputs remained active"
            )
        report["mixed_load"] = dict(
            short=short_results,
            extreme=[{k: v for k, v in r.items() if k != "text"} for r in big_results],
            short_completed_before_extreme=True,
        )
        report["after_mixed_load"] = wait_reclaimed(owner)
        report["mixed_metrics"] = rpc(endpoint, {"type": "metrics"})["metrics"]
        completed = report["mixed_metrics"]["split_pd"]["recent_requests"]
        if sum(r["generated_tokens"] == args.extreme_output_tokens for r in completed) < 3:
            raise AssertionError("Extreme outputs did not reach their configured token count")
        if report["mixed_metrics"]["split_pd"]["max_decode_batch"] < 2:
            raise AssertionError("Decode did not batch multiple requests")
        print("Mixed load: short requests finished before all three extreme outputs", flush=True)

        # Fill the long lane while keeping the short queue independently usable.
        ids = []
        rejected = 0
        generation = dict(
            max_new_tokens=args.extreme_output_tokens, ignore_eos=True, sampling=dict(temperature=0)
        )
        for _ in range(config["max_queue_size"] + 6):
            try:
                result = rpc(
                    endpoint,
                    dict(
                        type="generate",
                        request=dict(
                            prompt=prompts[2],
                            stream=True,
                            generation_config=generation,
                            timeout_ms=300000,
                        ),
                    ),
                )
                ids.append(result["request_id"])
            except RuntimeError as error:
                if "queue_full" not in str(error):
                    raise
                rejected += 1
        short_result = generate(port, prompts[0], 24)
        if rejected == 0:
            raise AssertionError("Long queue limit was not enforced")
        for request_id in ids:
            rpc(endpoint, dict(type="cancel", request_id=request_id, reason="validation_cancel"))
        report["after_cancel"] = wait_reclaimed(owner)
        report["queue_and_cancel"] = dict(
            long_accepted=len(ids),
            long_rejected=rejected,
            short_completed_while_long_queue_full=True,
            short_first_text_ms=short_result["first_text_ms"],
        )
        # Cancel after Decode has actually generated tokens, separately from
        # cancelling queued requests or a Prefill still computing its prompt.
        before = rpc(endpoint, {"type": "metrics"})["metrics"]["split_pd"]
        before_steps = before["decode_steps"]
        before_ids = {r["request_id"] for r in before["recent_requests"]}
        active = rpc(
            endpoint,
            dict(
                type="generate",
                request=dict(
                    prompt=prompts[2],
                    stream=True,
                    generation_config=generation,
                    timeout_ms=config["request_timeout_ms"],
                ),
            ),
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = rpc(endpoint, {"type": "metrics"})["metrics"]["split_pd"]
            if state["decode_steps"] >= before_steps + 16:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("Cancellation target did not enter Decode")
        rpc(
            endpoint,
            dict(type="cancel", request_id=active["request_id"], reason="validation_cancel"),
        )
        report["after_decode_cancel"] = wait_reclaimed(owner)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            report["final_metrics"] = rpc(endpoint, {"type": "metrics"})["metrics"]
            # Transport IDs also count rejected submissions; engine IDs do not.
            cancelled = next(
                (
                    r
                    for r in report["final_metrics"]["split_pd"]["recent_requests"]
                    if r["request_id"] not in before_ids
                ),
                None,
            )
            if cancelled is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Decode cancellation was not recorded")
        if (
            cancelled["failed"]
            or not 0 < cancelled["generated_tokens"] < args.extreme_output_tokens
        ):
            raise AssertionError(f"Active Decode cancellation failed: {cancelled}")
        report["decode_cancel"] = cancelled
        (output / "validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print("Queue isolation, cancellation and full arena reclaim passed", flush=True)
    finally:
        for process in (baseline, supervisor):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=330)
        for log in logs:
            log.close()
    shutdown = json.loads((output / "domain/shutdown.json").read_text())
    if not shutdown["graceful"]:
        raise AssertionError("Deployment shutdown was not graceful")
    print(f"Validation artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
