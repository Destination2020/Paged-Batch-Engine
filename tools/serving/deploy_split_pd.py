#!/usr/bin/env python3
"""Deploy two length-isolated Prefill processes and one batched Decode on one GPU.

All three workers load the same model. The owner is stopped only after every
importer exits; any worker failure discards the entire IPC domain.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.request

from deploy_multi_model import (
    Domain,
    MIB,
    ROOT,
    control,
    deployment_environment,
    gpu_info,
    mps_control_path,
    registered_server,
)


def validate(config):
    positive = (
        "arena_mib",
        "private_memory_mib",
        "short_prefill_kv_mib",
        "long_prefill_kv_mib",
        "decode_kv_mib",
        "short_prompt_tokens",
        "short_decode_slots",
        "long_decode_slots",
        "short_kv_tokens",
        "long_kv_tokens",
        "max_prompt_tokens",
        "max_output_tokens",
        "max_queue_size",
        "prefill_chunk_tokens",
        "request_timeout_ms",
    )
    for key in positive:
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("weights", "tokenizer"):
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config[key]}")
    with Path(config["weights"]).open("rb") as source:
        dim, _, layers, heads, kv_heads, _, context, magic, dtype = struct.unpack(
            "<9i", source.read(36)
        )
    if (
        magic != 0x44545950
        or dtype != 4
        or min(dim, layers, heads, kv_heads, context) <= 0
        or dim % heads
    ):
        raise ValueError("Expected valid PBE BF16 Qwen weights")
    token_bytes = 2 * layers * kv_heads * (dim // heads) * 2
    if config["max_prompt_tokens"] + config["max_output_tokens"] > context:
        raise ValueError("Input plus output exceeds model context")
    if config["short_prompt_tokens"] >= config["max_prompt_tokens"]:
        raise ValueError("Long lane must have a nonempty input range")
    if config["short_decode_slots"] + config["long_decode_slots"] > 16:
        raise ValueError("Decode batch limit is 16")
    # Disjoint hard quotas plus alignment slack guarantee the per-lane future
    # token promises without permitting decode recomputation under pressure.
    total = sum(
        config[key] for key in ("short_prefill_kv_mib", "long_prefill_kv_mib", "decode_kv_mib")
    )
    if total * MIB + 3 * 16 * token_bytes > config["arena_mib"] * MIB:
        raise ValueError("Arena must cover role quotas plus page-alignment slack")
    if (config["short_kv_tokens"] + config["long_kv_tokens"]) * token_bytes + 8 * MIB > config[
        "decode_kv_mib"
    ] * MIB:
        raise ValueError("Decode quota cannot cover lane token promises")
    for lane, max_input in [
        ("short", config["short_prompt_tokens"]),
        ("long", config["max_prompt_tokens"]),
    ]:
        charge = ((max_input + config["max_output_tokens"] + 15) // 16) * 16
        if charge > config[f"{lane}_kv_tokens"]:
            raise ValueError(f"{lane} lane cannot admit the maximum configured request")
        if ((max_input + 15) // 16) * 16 * token_bytes + 4 * MIB > config[
            f"{lane}_prefill_kv_mib"
        ] * MIB:
            raise ValueError(f"{lane} Prefill KV quota too small")
    if Path(config["weights"]).stat().st_size > config["private_memory_mib"] * MIB:
        raise ValueError("Weights exceed private worker budget")
    return dict(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=dim // heads,
        kv_bytes_per_token=token_bytes,
        context=context,
    )


def deploy(domain, config, build, backend, control_binary):
    owner_endpoint = f"ipc://{domain.directory}/kv.sock"
    domain.manage_service(
        dict(backend=backend, shutdown_timeout_s=300), control_binary, owner_endpoint
    )
    owner = domain.spawn(
        "kv-owner", [build / "demo/shared_kv_owner", owner_endpoint, config["arena_mib"]]
    )
    owner_health = domain.ready(owner_endpoint)
    if owner_health["gpu_free_bytes"] < (3 * config["private_memory_mib"] + 512) * MIB:
        raise RuntimeError("Not enough physical GPU memory for three private model budgets")
    endpoints = {
        role: f"ipc://{domain.directory}/{role}.sock"
        for role in ("prefill-short", "prefill-long", "decode")
    }
    workers = []
    for role in endpoints:
        decode = role == "decode"
        lane = role.removeprefix("prefill-")
        quota = config["decode_kv_mib"] if decode else config[f"{lane}_prefill_kv_mib"]
        model_id = config["model_id"] if decode else role
        env = domain.env.copy()
        env.update(KUIPER_USE_FP8_KV_CACHE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        percent = 50 if decode else 25
        if backend == "mps_process":
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percent)
            env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = (
                f"0={config['private_memory_mib']+config['arena_mib']}M"
            )
        command = [
            build / "demo/serving_qwen",
            config["weights"],
            config["tokenizer"],
            "--online-server=1",
            "--pd-mode=split-prefill-ipc",
            "--device-id=0",
            "--prefill-device-id=0",
            "--decode-device-id=0",
            "--radix-cache=off",
            f"--model-id={model_id}",
            f"--shared-kv-endpoint={owner_endpoint}",
            f'--private-memory-budget-mib={config["private_memory_mib"]}',
            f"--shared-kv-hard-limit-mib={quota}",
            "--shared-kv-chunk-mib=2",
            f'--max-queue-size={config["max_queue_size"]}',
            f'--max-prompt-tokens={config["max_prompt_tokens"]}',
            f'--max-output-tokens={config["max_output_tokens"]}',
            f'--request-timeout-ms={config["request_timeout_ms"]}',
            "--engine-zmq-timeout-ms=5000",
            "--warmup-rounds=0",
            "--final-summary=0",
        ]
        if decode:
            command += [
                "--online-process-role=zmq-decode-engine-core",
                f"--engine-zmq-endpoint={endpoints[role]}",
                "--max-batched-tokens=128",
                f'--pd-short-prefill-endpoint={endpoints["prefill-short"]}',
                f'--pd-long-prefill-endpoint={endpoints["prefill-long"]}',
            ]
            for key in (
                "short_prompt_tokens",
                "short_decode_slots",
                "long_decode_slots",
                "short_kv_tokens",
                "long_kv_tokens",
            ):
                command.append(f'--pd-{key.replace("_", "-")}={config[key]}')
        else:
            command += [
                "--online-process-role=zmq-prefill-engine-core",
                f"--prefill-zmq-endpoint={endpoints[role]}",
                f'--max-batched-tokens={config["prefill_chunk_tokens"]}',
                f'--prefill-chunk-cap={config["prefill_chunk_tokens"]}',
            ]
        process = domain.spawn(role, command, env)
        entry = (role, process, endpoints[role])
        if decode:
            domain.service["workers"].insert(0, entry)  # D releases P leases before P exits.
        else:
            domain.service["workers"].append(entry)
        health = domain.ready(endpoints[role])
        if health["model_id"] != model_id:
            raise RuntimeError("Worker identity mismatch")
        workers.append(
            dict(
                role=role,
                pid=process.pid,
                endpoint=endpoints[role],
                mps_active_thread_percentage=percent if backend == "mps_process" else None,
                health=health,
            )
        )
    pids = [owner.pid] + [worker["pid"] for worker in workers]
    server = (
        registered_server(control_binary, pids, domain.env) if backend == "mps_process" else None
    )
    if backend == "mps_process" and server is None:
        raise RuntimeError("All workers must register with one MPS server")
    registry = dict(
        topology="2P1D",
        backend=backend,
        workers=workers,
        mps_server=server,
        arena_owner_pid=owner.pid,
        arena_epoch=owner_health["epoch"],
        arena_bytes=owner_health["bytes"],
        model_id=config["model_id"],
    )
    (domain.directory / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    domain.service["api"] = domain.spawn(
        "api",
        [
            build / "demo/serving_qwen",
            "unused",
            "unused",
            "--online-server=1",
            "--online-process-role=zmq-http-api",
            f'--engine-zmq-endpoint={endpoints["decode"]}',
            f'--model-id={config["model_id"]}',
            f'--listen-host={config.get("listen_host", "127.0.0.1")}',
            f'--listen-port={config.get("listen_port",8080)}',
            f'--request-timeout-ms={config["request_timeout_ms"]}',
        ],
    )
    url = f'http://{config.get("listen_host", "127.0.0.1")}:{config.get("listen_port",8080)}/health'
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        domain.alive()
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        raise TimeoutError("HTTP API did not become ready")
    print(json.dumps(dict(status="READY", **registry)), flush=True)
    while True:
        domain.alive()
        if server and registered_server(control_binary, pids, domain.env) != server:
            raise RuntimeError("MPS registration changed; stop the complete domain")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mps")
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["mps_process", "process"], default="mps_process")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    geometry = validate(config)
    gpu = gpu_info(config.get("gpu_uuid"))
    if 3 * config["private_memory_mib"] + config["arena_mib"] + 512 > gpu["free_mib"]:
        raise ValueError("Insufficient GPU memory budget")
    binary = mps_control_path()
    if args.backend == "mps_process" and not binary:
        raise RuntimeError("MPS control program unavailable")
    if args.check:
        print(json.dumps(dict(gpu=gpu, geometry=geometry, config=config), indent=2))
        return
    directory = args.runtime_dir.resolve()
    directory.mkdir(parents=True, mode=0o700)
    env = deployment_environment(os.environ, gpu["uuid"], directory, args.backend)
    domain = Domain(directory, env)
    daemon_owned = False
    try:
        if args.backend == "mps_process":
            for kind in ("pipe", "log"):
                path = directory / ("mps-" + kind)
                path.mkdir(mode=0o700)
                env["CUDA_MPS_" + kind.upper() + "_DIRECTORY"] = str(path)
            subprocess.run([binary, "-d"], env=env, check=True, timeout=10)
            daemon_owned = True
        deploy(domain, config, args.build_dir.resolve(), args.backend, binary)
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        if domain.service is not None:
            domain.service["failure"] = str(error)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        shutdown = domain.stop()
        if daemon_owned and not (shutdown or {}).get("owner_quarantined"):
            control(binary, "quit", env)
        if shutdown is not None and not shutdown["graceful"]:
            raise RuntimeError("PD domain did not shut down gracefully; inspect shutdown.json")


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    main()
