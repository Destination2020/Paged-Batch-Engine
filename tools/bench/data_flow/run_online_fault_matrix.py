#!/usr/bin/env python3
"""Exercise malformed input and Data-service restart through the persistent language role."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from pbe_data_client import DataClient, DataKind
from pbe_data_client.client import checksum256
from pbe_data_client.tensor_bundle import encode
from pbe_roles.coordinator import LanguageProcess


def wait_socket(path: Path, process: subprocess.Popen):
    for _ in range(500):
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"data service exited with {process.returncode}")
        time.sleep(.01)
    raise RuntimeError("data service socket timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    layout = [text["num_hidden_layers"], 64, 16,
              text["num_key_value_heads"],
              text["hidden_size"] // text["num_attention_heads"], 4, 512]
    endpoint = Path(f"/tmp/pbe-online-fault-{os.getpid()}.sock")
    service_command = [str(args.build / "demo/pbe_data_service"), "serve-gpu",
                       str(endpoint), str(2 << 30), str(args.device),
                       *map(str, layout)]
    service_log = open(args.run_dir / "service.log", "w")
    service = subprocess.Popen(service_command, stdout=service_log, stderr=subprocess.STDOUT)
    wait_socket(endpoint, service)
    client = DataClient(str(endpoint))

    malformed = encode([("image_features", "bfloat16", (1, 2048), bytes(4096))])
    content = checksum256(malformed)
    representation = checksum256(b"qwen25-vl-request-bundle-v3")
    handle = client.reserve(DataKind.TENSOR_BUNDLE, content, representation, len(malformed))
    client.seal(handle, malformed)
    client.release_producer(handle)

    language = LanguageProcess([
        str(args.build / "demo/pbe_vlm_language_role"), str(args.model_bin),
        str(args.tokenizer), str(endpoint), str(args.device), str(64 << 20),
        str(32 << 20)])
    typed = {"request_id": "fault", "generation": 1, "content": content.hex(),
             "representation": representation.hex(), "max_new_tokens": 4,
             "timeout_ms": 30000}
    malformed_result = language.call({"op": "infer", "requests": [typed]})
    status_after_malformed = language.call({"op": "status"})
    assert not malformed_result["ok"]
    assert "bundle_shape_or_coverage_mismatch" in malformed_result["error"]
    assert status_after_malformed["budget"]["kv"]["used"] == 0

    old_incarnation = client.ping()
    os.kill(service.pid, signal.SIGKILL)
    service.wait(timeout=30)
    endpoint.unlink(missing_ok=True)
    service_log.close()
    restarted_log = open(args.run_dir / "service-restarted.log", "w")
    service = subprocess.Popen(service_command, stdout=restarted_log,
                               stderr=subprocess.STDOUT)
    wait_socket(endpoint, service)
    new_client = DataClient(str(endpoint))
    new_incarnation = new_client.ping()
    assert new_incarnation != old_incarnation
    stale_result = language.call({"op": "infer", "requests": [typed]})
    assert not stale_result["ok"] and "bundle_acquire_failed" in stale_result["error"]
    shutdown = language.call({"op": "shutdown"})
    language.process.wait(timeout=30)
    assert shutdown["ok"] and language.process.returncode == 0
    ipc = subprocess.check_output(
        [str(args.build / "demo/pbe_data_service"), "ipc-stats", str(endpoint)], text=True)
    stats = subprocess.check_output(
        [str(args.build / "demo/pbe_data_service"), "stats", str(endpoint)], text=True)
    new_client.shutdown()
    service.wait(timeout=30)
    restarted_log.close()
    result = {
        "malformed_rejected_before_compute": True,
        "same_language_pid_after_malformed":
            language.ready["worker_pid"] == status_after_malformed["worker_pid"],
        "budget_used_after_malformed": status_after_malformed["budget"]["kv"]["used"],
        "old_incarnation": old_incarnation, "new_incarnation": new_incarnation,
        "stale_object_rejected_after_restart": True,
        "language_shutdown_after_owner_loss": True,
        "ipc_after_restart": ipc.strip(), "data_after_restart": stats.strip(),
    }
    (args.run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
