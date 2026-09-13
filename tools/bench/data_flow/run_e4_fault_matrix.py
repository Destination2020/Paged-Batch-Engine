#!/usr/bin/env python3
"""Exercise E4 identity, idempotency, mapping, crash, and restart contracts."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess

LAYOUT = "pbe-qwen2-bf16-v1"


def wait_socket(path, process, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).is_socket(): return
        if process.poll() is not None: raise RuntimeError(f"owner exited {process.returncode}")
        time.sleep(.05)
    raise TimeoutError(path)


def parse_fields(text):
    result = {}
    for field in text.split():
        if "=" in field:
            key, value = field.split("=", 1)
            try: result[key] = int(value)
            except ValueError: result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    service = args.build / "demo/pbe_data_service"
    endpoint = f"/tmp/pbe-e4-fault-{os.getpid()}.sock"
    model_bytes = args.model_bin.stat().st_size
    events = []

    def service_command(sha=args.model_sha256, blocks=128):
        return [str(service), "serve-gpu-weights", endpoint, str(2 << 30),
                str(args.device), "36", str(blocks), "16", "2", "128", "4",
                str(args.model_bin), sha, LAYOUT, str(64 << 20), "2048"]

    def start_owner(log_name):
        log = (args.output.parent / log_name).open("w")
        process = subprocess.Popen(service_command(), stdout=log, stderr=subprocess.STDOUT,
                                   text=True)
        wait_socket(endpoint, process)
        return process, log

    def cli(*values):
        return subprocess.run([str(service), *map(str, values)], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Content is recomputed while uploading. A supplied digest for another file
    # never reaches ready and the partial CUDA allocation is freed on exit.
    bad_log = (args.output.parent / "bad-content-load.log").open("w")
    bad = subprocess.Popen(service_command("0" * 64), stdout=bad_log,
                           stderr=subprocess.STDOUT, text=True)
    bad_rc = bad.wait(timeout=180); bad_log.close()
    events.append({"event": "same_path_changed_content_rejected", "exit_code": bad_rc,
                   "socket_visible": Path(endpoint).exists()})
    Path(endpoint).unlink(missing_ok=True)

    # Cancellation before publication leaves no socket/ready object. Recovery
    # starts a fresh owner generation rather than accepting a late seal.
    cancel_log = (args.output.parent / "load-cancel.log").open("w")
    cancelled = subprocess.Popen(service_command(), stdout=cancel_log,
                                 stderr=subprocess.STDOUT, text=True)
    time.sleep(.1); cancelled.kill(); cancelled.wait(); cancel_log.close()
    events.append({"event": "loading_cancelled_before_ready", "exit_code": cancelled.returncode,
                   "socket_visible": Path(endpoint).exists()})
    Path(endpoint).unlink(missing_ok=True)

    owner, owner_log = start_owner("fault-owner-1.log")
    try:
        wrong_content = cli("acquire-weight", endpoint, "1" * 64, LAYOUT, 4,
                            model_bytes, 7001, 1)
        wrong_dtype = cli("acquire-weight", endpoint, args.model_sha256, LAYOUT, 1,
                          model_bytes, 7001, 2)
        wrong_bytes = cli("acquire-weight", endpoint, args.model_sha256, LAYOUT, 4,
                          model_bytes - 2, 7001, 3)
        events.extend([
            {"event": "wrong_content_rejected", "exit_code": wrong_content.returncode,
             "stderr": wrong_content.stderr.strip()},
            {"event": "wrong_dtype_rejected", "exit_code": wrong_dtype.returncode,
             "stderr": wrong_dtype.stderr.strip()},
            {"event": "wrong_bytes_rejected", "exit_code": wrong_bytes.returncode,
             "stderr": wrong_bytes.stderr.strip()}])

        first = cli("acquire-weight", endpoint, args.model_sha256, LAYOUT, 4,
                    model_bytes, 8001, 9)
        replay = cli("acquire-weight", endpoint, args.model_sha256, LAYOUT, 4,
                     model_bytes, 8001, 9)
        first_fields, replay_fields = parse_fields(first.stdout), parse_fields(replay.stdout)
        stats_one = parse_fields(cli("weight-stats", endpoint).stdout)
        release = cli("release-weight", endpoint, first_fields["service_incarnation"],
                      first_fields["consumer_incarnation"], first_fields["lease_id"])
        release_again = cli("release-weight", endpoint, first_fields["service_incarnation"],
                            first_fields["consumer_incarnation"], first_fields["lease_id"])
        stats_zero = parse_fields(cli("weight-stats", endpoint).stdout)
        events.append({"event": "reply_lost_replay_and_duplicate_release",
            "first": first_fields, "replay": replay_fields,
            "same_lease": first_fields == replay_fields,
            "active_after_replay": stats_one.get("active_leases"),
            "first_release_exit": release.returncode,
            "duplicate_release_exit": release_again.returncode,
            "active_after_release": stats_zero.get("active_leases")})

        wrong_gpu = cli("map-weight", endpoint, args.model_sha256, LAYOUT, 4,
                        model_bytes, 1 if args.device == 0 else 0)
        after_map = parse_fields(cli("weight-stats", endpoint).stdout)
        events.append({"event": "physical_gpu_mapping_mismatch",
                       "exit_code": wrong_gpu.returncode,
                       "stderr": wrong_gpu.stderr.strip(),
                       "active_leases_after": after_map.get("active_leases")})

        worker_command = [str(args.build / "demo/pbe_vlm_language_role"),
            str(args.model_bin), str(args.tokenizer), endpoint, str(args.device),
            str(64 << 20), str(32 << 20), "32", "0", "1", "", "shared",
            args.model_sha256, LAYOUT]
        bad_layout = subprocess.run(worker_command[:-1] + ["unknown-layout"], text=True,
                                    input="", stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=60)
        layout_ipc = parse_fields(cli("ipc-stats", endpoint).stdout)
        layout_weight = parse_fields(cli("weight-stats", endpoint).stdout)
        events.append({"event": "unknown_layout_rejected_before_compute",
                       "exit_code": bad_layout.returncode,
                       "stderr": bad_layout.stderr.strip(),
                       "free_slots_after": layout_ipc.get("free_slots"),
                       "total_slots": layout_ipc.get("total_slots"),
                       "active_weight_leases_after": layout_weight.get("active_leases")})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            followers = list(pool.map(lambda _: LanguageProcess(worker_command), range(2)))
        follower_status = [value.call({"op": "status"}) for value in followers]
        concurrent_stats = parse_fields(cli("weight-stats", endpoint).stdout)
        for value in followers: value.close()
        events.append({"event": "concurrent_followers_share_ready_object",
            "worker_pids": [value["worker_pid"] for value in follower_status],
            "lease_ids": [value["weights"]["import_lease_id"] for value in follower_status],
            "allocation_ids": [value["weights"]["allocation_id"] for value in follower_status],
            "upload_count": concurrent_stats.get("upload_count"),
            "allocation_count": concurrent_stats.get("allocation_count"),
            "active_leases": concurrent_stats.get("active_leases")})
        worker = LanguageProcess(worker_command)
        status = worker.call({"op": "status"})
        old_token = status["weights"]
        worker.process.kill(); worker.process.wait()
        leaked = parse_fields(cli("weight-stats", endpoint).stdout)
        events.append({"event": "consumer_sigkill_requires_owner_isolation",
                       "consumer_pid": status["worker_pid"],
                       "active_leases_before_isolation": leaked.get("active_leases")})
    finally:
        owner.kill(); owner.wait(); owner_log.close()
        Path(endpoint).unlink(missing_ok=True)

    owner2, owner2_log = start_owner("fault-owner-2.log")
    try:
        after_restart = parse_fields(cli("weight-stats", endpoint).stdout)
        stale = cli("release-weight", endpoint, old_token["owner_incarnation"],
                    old_token["consumer_incarnation"], old_token["import_lease_id"])
        events.append({"event": "owner_restart_rejects_old_generation",
            "old_incarnation": old_token["owner_incarnation"],
            "new_incarnation": after_restart.get("owner_incarnation"),
            "stale_release_exit": stale.returncode,
            "new_active_leases": after_restart.get("active_leases")})

        live = LanguageProcess(worker_command)
        live_status = live.call({"op": "status"})
        owner2.kill(); owner2.wait(); owner2_log.close()
        fail_stop = live.call({"op": "infer", "requests": []}, timeout=30)
        events.append({"event": "owner_crash_blocks_new_compute",
                       "worker_pid": live_status["worker_pid"],
                       "response": fail_stop})
        live.process.kill(); live.process.wait()
        Path(endpoint).unlink(missing_ok=True)
        owner3, owner3_log = start_owner("fault-owner-3.log")
        recovered = LanguageProcess(worker_command)
        recovered_status = recovered.call({"op": "status"})
        recovered.close()
        recovered_leases = parse_fields(cli("weight-stats", endpoint).stdout)
        events.append({"event": "fresh_generation_recovers_service",
            "owner_incarnation": recovered_status["weights"]["owner_incarnation"],
            "active_leases_after_worker_exit": recovered_leases.get("active_leases")})
        cli("shutdown", endpoint); owner3.wait(timeout=30); owner3_log.close()
    finally:
        if owner2.poll() is None:
            owner2.kill(); owner2.wait(); owner2_log.close()
        Path(endpoint).unlink(missing_ok=True)

    capacity_log = (args.output.parent / "capacity-exhausted.log").open("w")
    capacity = subprocess.Popen(service_command(blocks=300000), stdout=capacity_log,
                                stderr=subprocess.STDOUT, text=True)
    capacity_rc = capacity.wait(timeout=60); capacity_log.close()
    events.append({"event": "physical_capacity_exhausted_before_publication",
                   "exit_code": capacity_rc,
                   "socket_visible": Path(endpoint).exists()})
    Path(endpoint).unlink(missing_ok=True)

    by_event = {item["event"]: item for item in events}
    checks = {
        "changed_content_rejected": by_event["same_path_changed_content_rejected"]["exit_code"] != 0 and
            not by_event["same_path_changed_content_rejected"]["socket_visible"],
        "load_cancel_not_published": by_event["loading_cancelled_before_ready"]["exit_code"] != 0 and
            not by_event["loading_cancelled_before_ready"]["socket_visible"],
        "identity_mismatches_rejected": all(by_event[name]["exit_code"] != 0 for name in
            ("wrong_content_rejected", "wrong_dtype_rejected", "wrong_bytes_rejected")),
        "idempotent_acquire_release": by_event["reply_lost_replay_and_duplicate_release"]["same_lease"] and
            by_event["reply_lost_replay_and_duplicate_release"]["active_after_replay"] == 1 and
            by_event["reply_lost_replay_and_duplicate_release"]["active_after_release"] == 0 and
            by_event["reply_lost_replay_and_duplicate_release"]["duplicate_release_exit"] == 0,
        "mapping_failure_releases_lease": by_event["physical_gpu_mapping_mismatch"]["exit_code"] != 0 and
            by_event["physical_gpu_mapping_mismatch"]["active_leases_after"] == 0,
        "unknown_layout_rejected_and_rolled_back":
            by_event["unknown_layout_rejected_before_compute"]["exit_code"] != 0 and
            by_event["unknown_layout_rejected_before_compute"]["free_slots_after"] ==
            by_event["unknown_layout_rejected_before_compute"]["total_slots"] and
            by_event["unknown_layout_rejected_before_compute"]["active_weight_leases_after"] == 0,
        "concurrent_followers_single_upload":
            len(set(by_event["concurrent_followers_share_ready_object"]["worker_pids"])) == 2 and
            len(set(by_event["concurrent_followers_share_ready_object"]["lease_ids"])) == 2 and
            len(set(by_event["concurrent_followers_share_ready_object"]["allocation_ids"])) == 1 and
            by_event["concurrent_followers_share_ready_object"]["upload_count"] == 1 and
            by_event["concurrent_followers_share_ready_object"]["allocation_count"] == 1 and
            by_event["concurrent_followers_share_ready_object"]["active_leases"] == 2,
        "consumer_crash_detected": by_event["consumer_sigkill_requires_owner_isolation"][
            "active_leases_before_isolation"] == 1,
        "old_generation_rejected": by_event["owner_restart_rejects_old_generation"]["old_incarnation"] !=
            by_event["owner_restart_rejects_old_generation"]["new_incarnation"] and
            by_event["owner_restart_rejects_old_generation"]["stale_release_exit"] != 0,
        "owner_crash_fail_stop": not by_event["owner_crash_blocks_new_compute"]["response"].get("ok") and
            "fail_stop" in by_event["owner_crash_blocks_new_compute"]["response"].get("error", ""),
        "restart_recovers_clean": by_event["fresh_generation_recovers_service"][
            "active_leases_after_worker_exit"] == 0,
        "capacity_exhaustion_not_published": by_event[
            "physical_capacity_exhausted_before_publication"]["exit_code"] != 0 and
            not by_event["physical_capacity_exhausted_before_publication"]["socket_visible"],
    }
    result = {"schema": "pbe-e4-fault-matrix-v1", "ok": all(checks.values()),
              "checks": checks, "events": events}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (args.output.parent / "fault_trace.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True)+"\n" for item in events))
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
