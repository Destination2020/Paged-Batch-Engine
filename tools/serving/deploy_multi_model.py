#!/usr/bin/env python3
"""Exec-spawned PBE workers with a unique CUDA IPC owner and strict MPS checks.

Requires pyzmq. The supervisor never creates a CUDA context or imports torch.
Any owner/worker failure stops the entire GPU domain; leases are not reclaimed
from dead clients based on a timer. MPS kernel overlap is a separate probe result.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import struct
import tempfile
import time
import urllib.request

MIB = 1 << 20
ROOT = Path(__file__).resolve().parents[2]


def rpc(endpoint, body, timeout_ms=1000):
    import zmq
    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.connect(endpoint)
            socket.send_json(body)
            result = socket.recv_json()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "RPC failed"))
    return result


def gpu_info(uuid=None):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.free,memory.total,name,driver_version",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    devices = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        devices.append(dict(uuid=fields[0], free_mib=int(fields[1]), total_mib=int(fields[2]),
                            name=fields[3], driver=fields[4]))
    if uuid:
        devices = [device for device in devices if device["uuid"] == uuid]
    if len(devices) != 1:
        raise ValueError("Specify gpu_uuid selecting exactly one GPU")
    return devices[0]


def mps_control_path():
    binary = shutil.which("nvidia-cuda-mps-control")
    if not binary:
        for path in ("/usr/local/cuda/bin/nvidia-cuda-mps-control",):
            if Path(path).is_file():
                binary = path
                break
    return binary


def control(binary, command, env):
    result = subprocess.run([binary], input=command + "\n", capture_output=True,
                            text=True, env=env, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "MPS control failed")
    return result.stdout.strip()


def registered_server(binary, pids, env):
    for server in re.findall(r"(?m)^\s*(\d+)\s*$", control(binary, "get_server_list", env)):
        clients = set(map(int, re.findall(r"\b\d+\b", control(binary, f"get_client_list {server}", env))))
        if set(pids).issubset(clients):
            return int(server)
    return None


def terminate_mps_client(binary, pid, env):
    # This launcher targets the legacy MPS control interface used by 565/595.
    # Never interpret a successful shell exit alone as CUDA termination success.
    server = registered_server(binary, [pid], env)
    if server is None:
        raise RuntimeError(f'MPS client {pid} not registered; GPU termination unconfirmed')
    reply = control(binary, f'terminate_client {server} {pid}', env)
    if reply.strip() != '0':
        raise RuntimeError(f'MPS terminate_client failed: {reply}')
    return server


def validate_config(config, require_files=True):
    timeout = config.get('shutdown_timeout_s', 30)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 300:
        raise ValueError('shutdown_timeout_s must be between 1 and 300 seconds')
    if config.get("backend", "mps_process") not in ("mps_process", "process"):
        raise ValueError("backend must be mps_process or process")
    models = config.get("models", [])
    if len(models) < 2:
        raise ValueError("At least two model workers required")
    ids = [model.get("id", "") for model in models]
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", id) for id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Model IDs must be nonempty, unique safe identifiers")
    arena = config.get("arena_mib", 512)
    if not isinstance(arena, int) or isinstance(arena, bool) or arena <= 0:
        raise ValueError("arena_mib must be a positive integer")
    for key, default in (("chunk_target_mib", 2), ("reserve_mib", 768), ("listen_port", 8080)):
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    worst_kv_bytes = 0
    for model in models:
        for key in ("private_memory_mib", "kv_hard_limit_mib"):
            if not isinstance(model.get(key), int) or model[key] <= 0:
                raise ValueError(f"{model['id']}: {key} must be a positive integer")
        if model["kv_hard_limit_mib"] > arena:
            raise ValueError("Model KV hard limit cannot exceed entire arena")
        percentage = model.get("mps_active_thread_percentage", 100)
        if not isinstance(percentage, int) or not 1 <= percentage <= 100:
            raise ValueError("MPS active thread percentage must be in [1,100]")
        for key, default in (("max_batched_tokens", 128), ("prefill_chunk_cap", 128),
                             ("max_prompt_tokens", 1024), ("max_output_tokens", 256), ("max_active_requests", 2)):
            if not isinstance(model.get(key, default), int) or model.get(key, default) <= 0:
                raise ValueError(f"{model['id']}: {key} must be positive")
        for key in ("weights", "tokenizer"):
            if not model.get(key):
                raise ValueError(f"{model['id']}: missing {key}")
            path = Path(model[key])
            model[key] = str(path if path.is_absolute() else ROOT / path)
            if require_files and not Path(model[key]).is_file():
                raise FileNotFoundError(f"{model['id']}: missing {key}: {model[key]}")
        weights = Path(model["weights"])
        if weights.is_file():
            with weights.open("rb") as file:
                header = file.read(36)
            if len(header) != 36:
                raise ValueError("Truncated PBE weight header")
            dim, _, layers, heads, kv_heads, _, context, magic, dtype = struct.unpack("<9i", header)
            if magic != 0x44545950 or dtype != 4:
                raise ValueError("Shared KV deployment requires exported PBE BF16 weights")
            if min(dim, layers, heads, kv_heads, context) <= 0 or dim % heads or heads % kv_heads:
                raise ValueError("Invalid Qwen GQA geometry")
            if weights.stat().st_size > model["private_memory_mib"] * MIB:
                raise ValueError(f"{model['id']}: weights alone exceed private allocation budget")
            tokens = model.get("max_prompt_tokens", 1024) + model.get("max_output_tokens", 256)
            if tokens > context:
                raise ValueError(f"{model['id']}: prompt plus output exceeds model context")
            active = model.get("max_active_requests", 2)
            if active > 16:
                raise ValueError("max_active_requests exceeds Qwen worker batch limit 16")
            page_bytes = 16 * layers * kv_heads * (dim // heads) * 2 * 2
            chunk = ((max(config.get("chunk_target_mib", 2) * MIB, page_bytes) + page_bytes - 1)
                     // page_bytes) * page_bytes
            pages = ((tokens + 15) // 16) * active
            needed = ((pages * page_bytes + chunk - 1) // chunk) * chunk
            if needed > model["kv_hard_limit_mib"] * MIB:
                raise ValueError(f"{model['id']}: maximum request shape exceeds KV chunk quota")
            # Include alignment gaps between heterogeneous model groups. This is
            # an initial shape check; actual leasing still validates intervals.
            worst_kv_bytes += needed + page_bytes
    if worst_kv_bytes > arena * MIB:
        raise ValueError("Combined maximum request KV shapes exceed shared arena")
    if config.get("default_model", ids[0]) not in ids:
        raise ValueError("Unknown default_model")
    return arena + sum(model["private_memory_mib"] for model in models)


class Domain:
    def __init__(self, directory, env):
        self.directory = Path(directory)
        self.env = env
        self.children = []
        self.logs = []
        self.service = None
        self.stopped = False

    def manage_service(self, config, binary, owner_endpoint):
        self.service = dict(backend=config.get('backend', 'process'), binary=binary,
                            timeout=config.get('shutdown_timeout_s', 30),
                            owner_endpoint=owner_endpoint, workers=[], api=None)

    def spawn(self, name, command, env=None):
        log = open(self.directory / f"{name}.log", "w")
        self.logs.append(log)
        process = subprocess.Popen(list(map(str, command)), env=env or self.env,
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                   start_new_session=True)
        self.children.append((name, process))
        return process

    def alive(self):
        for name, child in self.children:
            if child.poll() is not None:
                raise RuntimeError(f"{name} exited ({child.returncode}); see {self.directory}/{name}.log")

    def ready(self, endpoint, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.alive()
            try:
                return rpc(endpoint, {"type": "health"})
            except Exception:
                time.sleep(0.1)
        raise TimeoutError(f"not ready: {endpoint}")

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.service is not None:
            return self.stop_service()
        # Stop API, then clients, wait for all client CUDA contexts to disappear,
        # and only then stop the exporter. Never free the owner before importers.
        clients = [(name, p) for name, p in self.children if name not in ("kv-owner", "ipc-event-owner")]
        owners = [(name, p) for name, p in self.children if name in ("kv-owner", "ipc-event-owner")]
        for group in (clients, owners):
            for _, child in reversed(group):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for _, child in reversed(group):
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for log in self.logs:
            log.close()

    def stop_service(self):
        if __package__:
            from .deployment_lifecycle import ShutdownCoordinator
        else:
            from deployment_lifecycle import ShutdownCoordinator
        return ShutdownCoordinator(self, rpc, terminate_mps_client).run()


def run_probe(domain, build, backend, binary):
    domain.manage_service(dict(backend=backend), binary, f'ipc://{domain.directory}/kv.sock')
    event_endpoint = f"ipc://{domain.directory}/event.sock"
    event_owner = domain.spawn("ipc-event-owner", [build / "demo/cuda_ipc_event_probe", "owner", event_endpoint])
    event_client = domain.spawn("ipc-event-client", [build / "demo/cuda_ipc_event_probe", "client", event_endpoint])
    for name, process in (("ipc-event-client", event_client), ("ipc-event-owner", event_owner)):
        if process.wait(timeout=30):
            raise RuntimeError(f"IPC memory/event probe failed: {domain.directory}/{name}.log")
        domain.children.remove((name, process))
    event_result = json.loads((domain.directory / "ipc-event-client.log").read_text().splitlines()[-1])
    endpoint = f"ipc://{domain.directory}/kv.sock"
    owner_process = domain.spawn("kv-owner", [build / "demo/shared_kv_owner", endpoint, 32])
    before = domain.ready(endpoint)
    barrier = domain.directory / "go"
    clients = [domain.spawn(name, [build / "demo/mps_kv_probe", endpoint, name, barrier])
               for name in ("probe-a", "probe-b")]
    deadline = time.monotonic() + 60
    while not all(Path(f"{barrier}.{name}").exists() for name in ("probe-a", "probe-b")):
        domain.alive()
        if time.monotonic() > deadline:
            raise TimeoutError("probe clients not ready")
        time.sleep(0.05)
    server = registered_server(binary, [owner_process.pid] + [client.pid for client in clients], domain.env) if backend == "mps_process" else None
    if backend == "mps_process" and server is None:
        raise RuntimeError("Both clients are not registered with the same MPS server")
    barrier.touch()
    results = []
    for name, process in zip(("probe-a", "probe-b"), clients):
        if process.wait(timeout=30):
            raise RuntimeError(f"{name} failed: {domain.directory}/{name}.log")
        domain.children.remove((name, process))
        lines = (domain.directory / f"{name}.log").read_text().splitlines()
        results.append(next(json.loads(line) for line in reversed(lines) if line.startswith("{")))
    after = rpc(endpoint, {"type": "metrics"})
    overlap = max(0, min(r["gpu_end_ns"] for r in results) - max(r["gpu_start_ns"] for r in results))
    report = dict(backend=backend, mps_clients_registered=server is not None,
                  mps_server=server, cuda_ipc_memory=all(r["cuda_ipc_memory"] for r in results),
                  cuda_ipc_memory_event=event_result["cuda_ipc_memory_event"],
                  ipc_event_pending_on_import=event_result["event_pending_on_import"],
                  deferred_retirement_verified=all(r["deferred_retirement_verified"] for r in results),
                  kv_payload_verified=all(r["kv_bytes_verified"] for r in results),
                  gpu_interval_overlap_ns=overlap, kernel_lifetime_overlap=overlap > 0,
                  mps_kernel_overlap_verified=False,
                  overlap_evidence="globaltimer intervals; confirm concurrent execution with Nsight Systems",
                  arena_fully_reclaimed=after["free_bytes"] == before["bytes"] and after["lease_count"] == 0,
                  clients=results)
    (domain.directory / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["arena_fully_reclaimed"]:
        raise RuntimeError("Probe leaked chunks")
    if backend == "mps_process" and not report["kernel_lifetime_overlap"]:
        raise RuntimeError("MPS clients registered but probe lifetimes did not overlap")


def run_service(domain, build, config, binary):
    endpoint = f"ipc://{domain.directory}/kv.sock"
    domain.manage_service(config, binary, endpoint)
    owner_process = domain.spawn("kv-owner", [build / "demo/shared_kv_owner", endpoint, config.get("arena_mib", 512)])
    owner = domain.ready(endpoint)
    if config.get('backend') == 'mps_process':
        if registered_server(binary, [owner_process.pid], domain.env) is None:
            raise RuntimeError('KV owner did not register with MPS; refusing to load models')
    private_and_reserve = sum(model["private_memory_mib"] for model in config["models"]) + config.get("reserve_mib", 768)
    if private_and_reserve * MIB > owner["gpu_free_bytes"]:
        raise RuntimeError("CUDA free bytes after arena allocation cannot fit private budgets and reserve")
    models = []
    workers = []
    for model in config["models"]:
        worker_endpoint = f"ipc://{domain.directory}/{model['id']}.sock"
        env = domain.env.copy()
        env["KUIPER_USE_FP8_KV_CACHE"] = "0"
        if config.get('backend') == 'mps_process':
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(model.get("mps_active_thread_percentage", 100))
        # Shared imported allocations may contribute to driver accounting; this
        # is a conservative client cap, separate from the global physical budget.
        if config.get('backend') == 'mps_process':
            env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={model['private_memory_mib'] + config.get('arena_mib', 512)}M"
        command = [build / "demo/serving_qwen", model["weights"], model["tokenizer"],
                   "--online-server=1", "--online-process-role=zmq-engine-core",
                   f"--engine-zmq-endpoint={worker_endpoint}", f"--model-id={model['id']}",
                   f"--shared-kv-endpoint={endpoint}",
                   f"--private-memory-budget-mib={model['private_memory_mib']}",
                   f"--shared-kv-hard-limit-mib={model['kv_hard_limit_mib']}",
                   f"--shared-kv-chunk-mib={config.get('chunk_target_mib', 2)}", "--radix-cache=off",
                   f"--max-batched-tokens={model.get('max_batched_tokens', 128)}",
                   f"--prefill-chunk-cap={model.get('prefill_chunk_cap', 128)}",
                   f"--max-prompt-tokens={model.get('max_prompt_tokens', 1024)}",
                   f"--max-output-tokens={model.get('max_output_tokens', 256)}",
                   f"--max-active-requests={model.get('max_active_requests', 2)}", "--warmup-rounds=1",
                   "--device-id=0", "--final-summary=0"]
        before = rpc(endpoint, {"type": "metrics"})["gpu_free_bytes"]
        worker = domain.spawn(model["id"], command, env)
        domain.service['workers'].append((model['id'], worker, worker_endpoint))
        health = domain.ready(worker_endpoint)
        if health.get("model_id") != model["id"]:
            raise RuntimeError("Worker model identity mismatch")
        after = rpc(endpoint, {"type": "metrics"})["gpu_free_bytes"]
        private_delta = max(0, before - after)
        tracked_private = health.get("private_allocator_reserved_bytes", 0)
        if tracked_private <= 0:
            raise RuntimeError("Worker did not report private CUDA allocation accounting")
        if max(private_delta, tracked_private) > model["private_memory_mib"] * MIB:
            raise RuntimeError(f"{model['id']} warmup GPU delta {private_delta / MIB:.1f} MiB exceeds private budget")
        models.append(dict(id=model["id"], endpoint=worker_endpoint, worker_pid=worker.pid,
                           visible_sm_count=health.get('visible_sm_count'),
                           max_batched_tokens_effective=health.get('max_batched_tokens_effective'),
                           mps_active_thread_percentage=model.get('mps_active_thread_percentage', 100)
                               if config.get('backend') == 'mps_process' else None,
                           mps_device_limit_mib=model['private_memory_mib'] + config.get('arena_mib', 512)
                               if config.get('backend') == 'mps_process' else None,
                           private_allocator_reserved_bytes=tracked_private,
                           private_gpu_delta_bytes=private_delta, private_gpu_delta_is_estimate=True))
        workers.append(worker)
    server = None
    if config.get("backend", "mps_process") == "mps_process":
        server = registered_server(binary, [owner_process.pid] + [worker.pid for worker in workers], domain.env)
        if server is None:
            raise RuntimeError("Owner and workers did not register with the same MPS server")
    registry = domain.directory / "registry.json"
    registry.write_text(json.dumps(dict(models=models, arena_owner_pid=owner_process.pid,
                                       backend=config.get('backend'), mps_server=server,
                                       default_model=config.get("default_model", models[0]["id"])), indent=2))
    host = config.get("listen_host", "127.0.0.1")
    port = config.get("listen_port", 8080)
    domain.service['api'] = domain.spawn("api", [build / "demo/serving_qwen", "unused", "unused",
                         "--online-server=1", "--online-process-role=zmq-http-api",
                         f"--model-registry={registry}", f"--listen-host={host}", f"--listen-port={port}",
                         "--request-timeout-ms=120000"])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        domain.alive()
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=1) as response:
                listed = json.load(response)
            if {item["id"] for item in listed["data"]} == {model["id"] for model in models}:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("API readiness failed")
    print(json.dumps(dict(status="READY", backend=config.get("backend", "mps_process"),
                          mps_server=server, models=models, arena_epoch=owner["epoch"],
                          arena_bytes=owner["bytes"], runtime_directory=str(domain.directory))), flush=True)
    next_mps_check = time.monotonic()
    while True:
        domain.alive()
        if server is not None and time.monotonic() >= next_mps_check:
            current = registered_server(binary, [owner_process.pid] + [w.pid for w in workers], domain.env)
            if current != server:
                raise RuntimeError('MPS domain registration lost or server changed; discard arena epoch')
            next_mps_check = time.monotonic() + 5
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-mps")
    parser.add_argument("--backend", choices=("mps_process", "process"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--start-mps", action="store_true", help="start an isolated daemon owned by this run")
    parser.add_argument("--runtime-dir", type=Path, help="new directory for logs/report; must not already exist")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    backend = args.backend or config.get("backend", "mps_process")
    config["backend"] = backend
    gpu = gpu_info(config.get("gpu_uuid"))
    binary = mps_control_path()
    issues = []
    if backend == "mps_process" and not binary:
        issues.append("nvidia-cuda-mps-control missing; MPS backend cannot run (no fallback)")
    if not args.probe:
        if not args.config:
            parser.error("--config required unless --probe is used")
        try:
            service_mib = validate_config(config, require_files=not args.check)
            reserve = config.get("reserve_mib", 768)
            if service_mib + reserve > gpu["free_mib"]:
                issues.append(f"GPU budget {service_mib}+reserve {reserve} MiB exceeds free {gpu['free_mib']} MiB")
            for model in config["models"]:
                for key in ("weights", "tokenizer"):
                    if not Path(model[key]).is_file():
                        issues.append(f"pending {model['id']} {key}: {model[key]}")
        except (ValueError, FileNotFoundError) as error:
            issues.append(str(error))
    for executable in (("shared_kv_owner", "mps_kv_probe", "cuda_ipc_event_probe") if args.probe else ("shared_kv_owner", "serving_qwen")):
        if not (args.build_dir / "demo" / executable).is_file():
            issues.append(f"missing executable: {args.build_dir}/demo/{executable}")
    if args.check or issues:
        print(json.dumps(dict(gpu=gpu, backend=backend, mps_control=binary,
                              wsl="microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
                              ready=not issues, issues=issues), indent=2))
        return 2 if issues else 0
    directory = args.runtime_dir or Path(tempfile.mkdtemp(prefix="pbe-domain-"))
    if args.runtime_dir:
        directory.mkdir(parents=True, mode=0o700)
    os.chmod(directory, 0o700)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    if backend == 'process':
        env = {key: value for key, value in env.items() if not key.startswith('CUDA_MPS_')}
        env['CUDA_MPS_PIPE_DIRECTORY'] = str(directory / 'mps-disabled')
    daemon_owned = False
    domain = Domain(directory, env)
    try:
        if backend == "mps_process":
            if args.start_mps:
                for kind in ("pipe", "log"):
                    path = directory / f"mps-{kind}"
                    path.mkdir(mode=0o700)
                    env[f"CUDA_MPS_{kind.upper()}_DIRECTORY"] = str(path)
                subprocess.run([binary, "-d"], env=env, check=True, timeout=10)
                daemon_owned = True
            control(binary, "get_server_list", env) # empty list is valid before first client
        elif args.start_mps:
            raise ValueError("--start-mps requires mps_process backend")
        if args.probe:
            run_probe(domain, args.build_dir.resolve(), backend, binary)
        else:
            run_service(domain, args.build_dir.resolve(), config, binary)
        return 0
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        if domain.service is not None:
            domain.service['failure'] = str(error)
        raise
    finally:
        # Repeated Ctrl-C must not interrupt CUDA/IPC cleanup halfway through.
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        shutdown = None
        try:
            shutdown = domain.stop()
        finally:
            try:
                if daemon_owned and not (shutdown or {}).get('owner_quarantined'):
                    control(binary, "quit", env)
                    if shutdown is not None:
                        shutdown['owned_mps_daemon_stopped'] = True
                        (directory / 'shutdown.json').write_text(json.dumps(shutdown, indent=2) + '\n')
            finally:
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGTERM, previous_term)
        print(f"Runtime logs: {directory}", flush=True)
        if shutdown is not None and not shutdown['graceful']:
            reason = next((event['reason'] for event in shutdown['events'] if event['event'] == 'fault'),
                          'shutdown incomplete')
            raise RuntimeError(f'GPU domain stopped after failure: {reason}; inspect shutdown.json')


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"DEPLOYMENT_ERROR: {error}", flush=True)
        raise SystemExit(1)
