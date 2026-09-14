#!/usr/bin/env python3
"""Real-model N3-N5 production coordinator and multi-P/D lifecycle acceptance."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode
from pbe_roles.pd_runtime import ProductionPDCoordinator, RoleRegistry
from run_e4_shared_weight_ab import LAYOUT, file_sha256, service_stats, wait_socket


def worker_command(args, endpoint: str, slots: int) -> list[str]:
    command = [str(args.build / "demo/pbe_vlm_language_role"), str(args.model_bin),
               str(args.tokenizer), endpoint, str(args.device), str(64 << 20),
               str(32 << 20), str(slots), "0", "1", ""]
    if args.weight_mode == "shared":
        command += ["shared", args.model_sha256, LAYOUT]
    return command


def item(args, request_id: str, **extra) -> dict:
    result = {"request_id": request_id, "generation": 1,
              "parts": [{"type": "image", "path": str(args.image), "size": 224},
                        {"type": "text", "text":
                         "Describe the image briefly and name its main subject."}],
              "max_new_tokens": 8, "timeout_ms": 120000}
    result.update(extra)
    return result


def wire(encoded: dict, request_id: str) -> dict:
    return {"request_id": request_id, "generation": 1,
            "content": encoded["content"], "representation": encoded["representation"],
            "feature_content": encoded["feature_content"],
            "feature_representation": encoded["feature_representation"],
            "max_new_tokens": 8, "timeout_ms": 120000}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight-mode", choices=("private", "shared"), default="shared")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.model_sha256 = file_sha256(args.model_bin)
    config = json.loads((args.model_dir / "config.json").read_text())
    config = config.get("text_config") or config
    endpoint = f"/tmp/pbe-n3n5-data-{os.getpid()}.sock"
    vision_endpoint = f"/tmp/pbe-n3n5-vision-{os.getpid()}.sock"
    service_cmd = [str(args.build / "demo/pbe_data_service"),
        "serve-gpu-weights" if args.weight_mode == "shared" else "serve-gpu",
        endpoint, str(2 << 30), str(args.device), str(config["num_hidden_layers"]),
        "256", "16", str(config["num_key_value_heads"]),
        str(config["hidden_size"] // config["num_attention_heads"]), "4"]
    if args.weight_mode == "shared":
        service_cmd += [str(args.model_bin), args.model_sha256, LAYOUT, str(64 << 20)]
    service_cmd += ["4096"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_log = (args.output.parent / "data.log").open("w")
    vision_log = (args.output.parent / "vision.log").open("w")
    service = subprocess.Popen(service_cmd, stdout=data_log, stderr=subprocess.STDOUT,
                               text=True)
    vision = None
    workers: dict[str, LanguageProcess] = {}
    closed: set[str] = set()
    evidence: dict = {"schema": "pbe-v4-production-multi-pd-v1",
                      "service_command": service_cmd, "paths": [], "faults": []}
    try:
        wait_socket(endpoint, service, 180)
        env = dict(os.environ, PYTHONPATH="python")
        vision_cmd = [str(Path(".venv/bin/python")),
            "python/pbe_roles/vision/persistent_worker.py", "--listen", vision_endpoint,
            "--endpoint", endpoint, "--model", str(args.model_dir),
            "--device", f"cuda:{args.device}", "--batch-window-ms", "20",
            "--max-batch", "8"]
        vision = subprocess.Popen(vision_cmd, stdout=vision_log, stderr=subprocess.STDOUT,
                                  env=env, text=True)
        wait_socket(vision_endpoint, vision, 180)
        for name in ("p0", "p1", "d0", "d1"):
            workers[name] = LanguageProcess(worker_command(args, endpoint, 32))
        registry = RoleRegistry(max_staleness_ms=120000)
        for name in ("p0", "p1"):
            registry.register(name, workers[name], {"prefill"}, endpoint)
        for name in ("d0", "d1"):
            registry.register(name, workers[name], {"decode"}, endpoint)
        coordinator = ProductionPDCoordinator(registry, vision_endpoint)

        # Real worker-owned capacity contention: after one half-pool anchor,
        # two requests race for the remaining half. Exactly one atomic grant
        # may succeed, and all unconsumed grants are explicitly rolled back.
        reserve_deadline = time.monotonic_ns() + 120_000_000_000
        anchor = workers["d0"].call({"op": "pd_reserve", "role": "decode",
            "request_id": "reserve-anchor", "generation": 1,
            "max_new_tokens": 256, "deadline_monotonic_ns": reserve_deadline})
        def reserve_contender(name: str):
            return workers["d0"].call({"op": "pd_reserve", "role": "decode",
                "request_id": name, "generation": 1, "max_new_tokens": 256,
                "deadline_monotonic_ns": reserve_deadline})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            contenders = list(pool.map(reserve_contender,
                                       ("reserve-contender-a", "reserve-contender-b")))
        reserve_releases = []
        for name, grant in [("reserve-anchor", anchor),
                            ("reserve-contender-a", contenders[0]),
                            ("reserve-contender-b", contenders[1])]:
            if grant.get("ok"):
                reserve_releases.append(workers["d0"].call({"op": "pd_unreserve",
                    "request_id": name, "generation": 1,
                    "reservation_id": grant["reservation_id"]}))
        evidence["faults"].append({"case": "atomic_reservation_contention",
            "anchor": anchor, "contenders": contenders, "releases": reserve_releases,
            "status_after": workers["d0"].call({"op": "status"})})

        # Every P→D edge is exercised through the production entry.
        for p in ("p0", "p1"):
            for d in ("d0", "d1"):
                result = coordinator.submit(item(args, f"edge-{p}-{d}",
                                                   prefill_worker=p, decode_worker=d))
                evidence["paths"].append({"topology": "2P2D", "edge": f"{p}->{d}",
                                           "result": result})

        # Sub-topology assertions use the same registered production path.
        for topology, edges in {"1P1D": [("p0", "d0")],
                                "1P2D": [("p0", "d0"), ("p0", "d1")],
                                "2P1D": [("p0", "d0"), ("p1", "d0")]}.items():
            for index, (p, d) in enumerate(edges):
                result = coordinator.submit(item(args, f"{topology}-{index}",
                    prefill_worker=p, decode_worker=d))
                evidence["paths"].append({"topology": topology,
                                           "edge": f"{p}->{d}", "result": result})

        cancelled = coordinator.submit(item(args, "cancel-before-d", prefill_worker="p0",
                    decode_worker="d0", cancel_before_decode=True))
        timed_out = coordinator.submit(item(args, "deadline-expired", timeout_ms=-1))
        first = coordinator.submit(item(args, "duplicate", prefill_worker="p0",
                                        decode_worker="d0"))
        duplicate = coordinator.submit(item(args, "duplicate", prefill_worker="p0",
                                            decode_worker="d0"))
        evidence["faults"] += [{"case": "cancel_before_decode", "result": cancelled},
                               {"case": "deadline_before_prepare", "result": timed_out},
                               {"case": "duplicate_generation", "first": first,
                                "result": duplicate}]

        def wait_for_stage(request_id: str, stage: str, timeout_s: float = 30) -> dict:
            until = time.monotonic() + timeout_s
            while time.monotonic() < until:
                snapshot = coordinator.active_state(request_id, 1)
                if snapshot and snapshot["stage"] == stage:
                    return snapshot
                time.sleep(.01)
            raise RuntimeError(f"request {request_id} did not reach {stage}")

        # External cancellation is issued after Decode starts, rather than
        # represented by a submission-time boolean.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            cancel_future = pool.submit(coordinator.submit, item(args,
                "external-cancel-in-decode", prefill_worker="p0", decode_worker="d0",
                max_new_tokens=64, hold_after_attach_ms=3000))
            cancel_snapshot = wait_for_stage("external-cancel-in-decode", "decode")
            cancel_ack = coordinator.cancel("external-cancel-in-decode", 1)
            cancel_result = cancel_future.result(timeout=30)
        evidence["faults"].append({"case": "external_cancel_during_decode",
            "active_snapshot": cancel_snapshot, "ack": cancel_ack,
            "result": cancel_result})

        # The same absolute deadline covers Encode, reservation, Prefill,
        # handoff and Decode. A held attach makes expiration deterministic.
        deadline_result = coordinator.submit(item(args, "deadline-in-decode",
            prefill_worker="p0", decode_worker="d0", timeout_ms=2500,
            max_new_tokens=64, hold_after_attach_ms=5000))
        evidence["faults"].append({"case": "absolute_deadline_during_decode",
                                    "result": deadline_result})

        encoded_item = item(args, "mechanism-encode")
        encoded_item["_deadline_monotonic_ns"] = time.monotonic_ns()+120_000_000_000
        _, encoded = encode(vision_endpoint, encoded_item)
        if not encoded.get("ok"):
            raise RuntimeError(f"mechanism vision encode failed: {encoded}")
        publication = workers["p0"].call({"op": "pd_prefill",
                                           "request": wire(encoded, "provider-check"),
                                           "oracle_steps": 0}, timeout=180)
        denied = workers["d0"].call({"op": "pd_decode", "request_id": "provider-check",
            "generation": 1, "kv_content": publication["kv_content"],
            "kv_representation": publication["kv_representation"],
            "expected_provider_incarnation": publication["provider_incarnation"]+1,
            "max_new_tokens": 8}, timeout=180)
        accepted = workers["d0"].call({"op": "pd_decode", "request_id": "provider-check",
            "generation": 1, "kv_content": publication["kv_content"],
            "kv_representation": publication["kv_representation"],
            "expected_provider_incarnation": publication["provider_incarnation"],
            "max_new_tokens": 8}, timeout=180)
        released = workers["p0"].call({"op": "pd_release", "request_id": "provider-check",
                                       "generation": 1})
        evidence["faults"].append({"case": "provider_generation_authorization",
                                    "denied": denied, "accepted": accepted,
                                    "released": released})

        # One published prefix fans out to independent D importers. One branch
        # is cancelled before attach; the other two complete with private tails.
        fan = workers["p0"].call({"op": "pd_prefill",
                                  "request": wire(encoded, "fanout"),
                                  "oracle_steps": 0}, timeout=180)
        def consume(name: str):
            return workers[name].call({"op": "pd_decode", "request_id": f"fanout-{name}",
                "generation": 1, "kv_content": fan["kv_content"],
                "kv_representation": fan["kv_representation"],
                "expected_provider_incarnation": fan["provider_incarnation"],
                "max_new_tokens": 8}, timeout=180)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fan_results = list(pool.map(consume, ("d0", "d1")))
        fan_release = workers["p0"].call({"op": "pd_release", "request_id": "fanout",
                                          "generation": 1})
        evidence["faults"].append({"case": "fanout_cow_and_branch_isolation",
                                    "cancelled_branch": cancelled,
                                    "consumers": fan_results, "release": fan_release})

        # D acquires its own attach lease, then P exits normally while D is
        # deliberately held after authorization but before attention.
        survive = workers["p1"].call({"op": "pd_prefill",
                                      "request": wire(encoded, "p-exit"),
                                      "oracle_steps": 0}, timeout=180)
        future = workers["d0"].send({"op": "pd_decode", "request_id": "p-exit",
            "generation": 1, "kv_content": survive["kv_content"],
            "kv_representation": survive["kv_representation"],
            "expected_provider_incarnation": survive["provider_incarnation"],
            "hold_after_attach_ms": 1500, "max_new_tokens": 8})
        time.sleep(.5)
        p_exit_begin = time.monotonic_ns(); workers["p1"].close()
        p_exit_end = time.monotonic_ns(); closed.add("p1")
        survived = future.result(timeout=180)
        stale = workers["d0"].call({"op": "pd_decode", "request_id": "p-exit-stale",
            "generation": 1, "kv_content": survive["kv_content"],
            "kv_representation": survive["kv_representation"],
            "expected_provider_incarnation": survive["provider_incarnation"],
            "max_new_tokens": 8}, timeout=180)
        evidence["faults"].append({"case": "provider_normal_exit_after_attach",
            "p_exit_begin_ns": p_exit_begin, "p_exit_end_ns": p_exit_end,
            "decode": survived, "stale_reuse": stale})

        # Static endpoint rejoins with a new incarnation; the registry rejects
        # identity reuse but accepts the fresh generation.
        workers["d1"].close(); closed.add("d1")
        workers["d1-new"] = LanguageProcess(worker_command(args, endpoint, 32))
        new_record = registry.register("d1", workers["d1-new"], {"decode"}, endpoint)
        rejoin = coordinator.submit(item(args, "new-d-join", prefill_worker="p0",
                                         decode_worker="d1"))
        evidence["faults"].append({"case": "new_decode_incarnation",
                                    "incarnation": new_record.incarnation,
                                    "result": rejoin})
        evidence["registry"] = registry.snapshot()
        evidence["pre_shutdown_ipc"] = service_stats(
            args.build / "demo/pbe_data_service", endpoint, "ipc-stats")
    finally:
        for name, process in workers.items():
            if name not in closed:
                try: process.close()
                except Exception as error: evidence.setdefault("cleanup_errors", []).append(str(error))
        if vision is not None:
            vision.terminate(); vision.wait(timeout=30)
        evidence["post_worker_ipc"] = service_stats(
            args.build / "demo/pbe_data_service", endpoint, "ipc-stats")
        subprocess.run([str(args.build / "demo/pbe_data_service"), "shutdown", endpoint],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        service.wait(timeout=30); data_log.close(); vision_log.close()

    paths_ok = all(x["result"].get("ok") for x in evidence["paths"])
    faults = {x["case"]: x for x in evidence["faults"]}
    fanout = faults["fanout_cow_and_branch_isolation"]["consumers"]
    checks = {
        "all_topologies_and_edges": paths_ok and len(evidence["paths"]) == 9,
        "dynamic_authorization": not faults["provider_generation_authorization"]["denied"].get("ok")
            and faults["provider_generation_authorization"]["accepted"].get("ok"),
        "decode_never_recomputes_prefix": all(x["result"].get("decode", {}).get(
            "actual_computed_prompt_tokens") == 0 for x in evidence["paths"]),
        "ordered_worker_reservations_consumed": all(
            path["result"].get("prefill", {}).get("consumed_reservation_id") ==
                next(span["p_reservation"]["reservation_id"]
                     for span in path["result"]["state"]["trace"]
                     if span["stage"] == "reserve") and
            path["result"].get("decode", {}).get("consumed_reservation_id") ==
                next(span["d_reservation"]["reservation_id"]
                     for span in path["result"]["state"]["trace"]
                     if span["stage"] == "reserve")
            for path in evidence["paths"]),
        "atomic_reservation_contention_and_recovery":
            faults["atomic_reservation_contention"]["anchor"].get("ok") and
            sum(bool(x.get("ok")) for x in faults[
                "atomic_reservation_contention"]["contenders"]) == 1 and
            faults["atomic_reservation_contention"]["status_after"]["pd"][
                "active_reservations"] == 0,
        "fanout_two_real_decoders": len({x.get("worker_pid") for x in fanout}) == 2 and
            all(x.get("ok") and x.get("attach_grant") and
                x.get("private_grant") != x.get("prefix_grant") for x in fanout),
        "cancel_timeout_duplicate_terminal": not faults["cancel_before_decode"]["result"].get("ok")
            and not faults["deadline_before_prepare"]["result"].get("ok")
            and not faults["duplicate_generation"]["result"].get("ok"),
        "external_cancel_and_deadline_reach_active_decode":
            faults["external_cancel_during_decode"]["ack"].get("accepted") and
            faults["external_cancel_during_decode"]["active_snapshot"]["stage"] == "decode" and
            faults["external_cancel_during_decode"]["result"]["state"]["terminal"] == "cancelled" and
            faults["absolute_deadline_during_decode"]["result"]["state"]["terminal"] == "timed_out",
        "provider_exit_survives_and_stale_rejected": faults[
            "provider_normal_exit_after_attach"]["decode"].get("ok") and
            faults["provider_normal_exit_after_attach"]["decode"].get(
                "attach_completed_ns", 2**63) < faults[
                    "provider_normal_exit_after_attach"]["p_exit_end_ns"] and
            not faults["provider_normal_exit_after_attach"]["stale_reuse"].get("ok"),
        "new_decode_joined": faults["new_decode_incarnation"]["result"].get("ok"),
        "all_ipc_grants_reclaimed": evidence["post_worker_ipc"].get("active_grants") == 0,
    }
    evidence["checks"] = checks; evidence["ok"] = all(checks.values())
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": evidence["ok"], "checks": checks}, sort_keys=True))
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
