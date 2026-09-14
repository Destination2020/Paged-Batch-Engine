#!/usr/bin/env python3
"""Fair fixed-KV/open-loop topology matrix using production coordinator paths."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

from pbe_roles.coordinator import LanguageProcess, encode
from pbe_roles.pd_runtime import ProductionPDCoordinator, RoleRegistry
from pbe_roles.vision.client import call as vision_call
from run_e4_shared_weight_ab import LAYOUT, file_sha256, gpu_used_mib, wait_socket

TOPOLOGIES = {"unified": (0, 0), "1P1D": (1, 1), "1P2D": (1, 2),
              "2P1D": (2, 1), "2P2D": (2, 2)}
WORKLOADS = {
    "long_input_short_output": ("Explain only visible evidence. " * 24, 4),
    "short_input_long_output": ("Describe the image.", 12),
    "mixed": ("Describe the visible flow and its main subject. " * 5, 8),
}
# Frozen after the original pilot observed roughly 0.37--0.45 req/s.  These
# represent 0.25, 0.5 and 5.0 offered req/s; near_saturation is a finite
# sustained-overload arrival train, never a zero-gap burst.
LOADS_MS = {"low": 4000, "medium": 2000, "near_saturation": 200}


def percentile(values, p):
    values = sorted(values); pos = (len(values)-1)*p/100
    lo = int(pos); hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def command(args, endpoint, slots):
    value = [str(args.build/"demo/pbe_vlm_language_role"), str(args.model_bin),
             str(args.tokenizer), endpoint, str(args.device), str(64<<20),
             str(32<<20), str(slots), "0", "1", ""]
    if args.mode == "shared": value += ["shared", args.model_sha256, LAYOUT]
    return value


def request(args, request_id, workload, pids, dids, index):
    text, output = WORKLOADS[workload]
    item = {"request_id": request_id, "generation": 1,
            "parts": [{"type": "image", "path": str(args.image), "size": 224},
                      {"type": "text", "text": text}],
            "max_new_tokens": output, "timeout_ms": 120000}
    if pids: item["prefill_worker"] = pids[index % len(pids)]
    if dids: item["decode_worker"] = dids[(index // max(1, len(pids))) % len(dids)]
    return item


def run_session(args, topology):
    p_count, d_count = TOPOLOGIES[topology]
    endpoint = f"/tmp/pbe-n6-data-{os.getpid()}-{args.mode}-{topology}.sock"
    vision_endpoint = f"/tmp/pbe-n6-vision-{os.getpid()}-{args.mode}-{topology}.sock"
    config = json.loads((args.model_dir/"config.json").read_text())
    config = config.get("text_config") or config
    service_cmd = [str(args.build/"demo/pbe_data_service"),
        "serve-gpu-weights" if args.mode == "shared" else "serve-gpu", endpoint,
        str(2<<30), str(args.device), str(config["num_hidden_layers"]), "256", "16",
        str(config["num_key_value_heads"]),
        str(config["hidden_size"]//config["num_attention_heads"]), "4"]
    if args.mode == "shared":
        service_cmd += [str(args.model_bin), args.model_sha256, LAYOUT, str(64<<20)]
    service_cmd += ["4096"]
    trial_dir = args.output_dir/"raw_v2"/args.mode/topology
    trial_dir.mkdir(parents=True, exist_ok=True)
    data_log = (trial_dir/"data.log").open("w")
    vision_log = (trial_dir/"vision.log").open("w")
    service = subprocess.Popen(service_cmd, stdout=data_log, stderr=subprocess.STDOUT,
                               text=True); vision = None; workers = []
    result = {"mode": args.mode, "topology": topology, "service_command": service_cmd,
              "records": [], "cells": [], "oracle_steps_in_timing": 0,
              "diagnostics_in_timing": False, "client_ttft": "N/A (non-streaming)"}
    baseline_mib = gpu_used_mib(args.device)
    try:
        wait_socket(endpoint, service, 180)
        env = dict(os.environ, PYTHONPATH="python")
        vision_cmd = [".venv/bin/python", "python/pbe_roles/vision/persistent_worker.py",
            "--listen", vision_endpoint, "--endpoint", endpoint, "--model",
            str(args.model_dir), "--device", f"cuda:{args.device}",
            "--batch-window-ms", "20", "--max-batch", "8"]
        vision = subprocess.Popen(vision_cmd, stdout=vision_log, stderr=subprocess.STDOUT,
                                  env=env, text=True); wait_socket(vision_endpoint, vision, 180)
        registry = RoleRegistry(max_staleness_ms=120000); coordinator = None
        pids, dids = [], []
        if topology == "unified":
            workers.append(LanguageProcess(command(args, endpoint, 128)))
        else:
            role_slots = 128//(p_count+d_count)
            # Give any remainder to Decode; total ownership remains exactly 128.
            slot_counts = [role_slots]*p_count + [role_slots]*d_count
            slot_counts[-1] += 128-sum(slot_counts)
            for index in range(p_count):
                process = LanguageProcess(command(args, endpoint, slot_counts[index]))
                workers.append(process); name=f"p{index}"; pids.append(name)
                registry.register(name, process, {"prefill"}, endpoint)
            for index in range(d_count):
                process = LanguageProcess(command(
                    args, endpoint, slot_counts[p_count+index]))
                workers.append(process); name=f"d{index}"; dids.append(name)
                registry.register(name, process, {"decode"}, endpoint)
            coordinator = ProductionPDCoordinator(registry, vision_endpoint)
        result["resident_mib"] = gpu_used_mib(args.device)-baseline_mib
        result["worker_ready"] = [x.ready for x in workers]

        def submit_one(value, scheduled):
            actual = time.perf_counter()
            if topology == "unified":
                stamped = dict(value,
                    _deadline_monotonic_ns=time.monotonic_ns()+120_000_000_000)
                _, encoded = encode(vision_endpoint, stamped)
                wire = {"request_id": value["request_id"], "generation": 1,
                    "content": encoded["content"],
                    "representation": encoded["representation"],
                    "feature_content": encoded["feature_content"],
                    "feature_representation": encoded["feature_representation"],
                    "max_new_tokens": value["max_new_tokens"],
                    "deadline_monotonic_ns": stamped["_deadline_monotonic_ns"],
                    "timeout_ms": 120000}
                response = workers[0].call({"op": "infer", "requests": [wire]},
                                           timeout=180)
                output = next((item for item in response.get("outputs", [])
                               if item.get("request_id") == value["request_id"]), {})
                return {"ok": response.get("ok") and bool(output),
                    "response": response, "tokens": output.get("tokens", []),
                    "request_id": value["request_id"], "scheduled": scheduled,
                    "started": actual, "completed": time.perf_counter(),
                    "unified_batch_size": response.get("coalesced_rpc_count", 1)}
            pd = coordinator.submit(value)
            return {"ok": pd.get("ok", False), "response": pd,
                    "tokens": pd.get("tokens", []),
                    "request_id": value["request_id"], "scheduled": scheduled,
                    "started": actual, "completed": time.perf_counter()}

        # One explicit untimed request establishes the declared warm-cache and
        # initialized-kernel state before any formal trial.
        warm_value = request(args, f"{args.mode}-{topology}-warmup", "mixed",
                             pids, dids, 0)
        warmup = submit_one(warm_value, time.perf_counter())
        if not warmup["ok"]:
            raise RuntimeError("n6_warmup_failed")
        result["warmup"] = warmup
        if topology == "unified":
            probe_values = [request(args, f"{args.mode}-{topology}-batch-probe-{index}",
                                    "mixed", pids, dids, index)
                            for index in range(5)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                probe_encoded = list(pool.map(lambda value: encode(
                    vision_endpoint, dict(value, _deadline_monotonic_ns=
                        time.monotonic_ns()+120_000_000_000))[1], probe_values))
            probe_wires = [{"request_id": value["request_id"], "generation": 1,
                "content": encoded["content"], "representation": encoded["representation"],
                "feature_content": encoded["feature_content"],
                "feature_representation": encoded["feature_representation"],
                "max_new_tokens": value["max_new_tokens"], "timeout_ms": 120000}
                for value, encoded in zip(probe_values, probe_encoded)]
            result["unified_batch_capability_probe"] = workers[0].call(
                {"op": "infer", "requests": probe_wires}, timeout=300)

        selected_workloads = (WORKLOADS if args.workload == "all" else (args.workload,))
        selected_loads = (LOADS_MS if args.load == "all" else (args.load,))
        cells = [(workload, load, trial) for workload in selected_workloads
                 for load in selected_loads for trial in range(args.repeats)]
        random.Random(20260914 + sum(map(ord, topology+args.mode))).shuffle(cells)
        for cell_order, (workload, load, trial) in enumerate(cells):
            interval = LOADS_MS[load]/1000
            trial_epoch = time.perf_counter()
            cache_initial = {"vision": vision_call(vision_endpoint,
                {"op": "feature_status"}),
                "workers": [worker.call({"op": "status"}) for worker in workers]}
            request_count = (max(args.requests_per_trial, 11)
                             if load == "near_saturation"
                             else args.requests_per_trial)
            futures=[]
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=request_count) as pool:
                for request_index in range(request_count):
                    scheduled = trial_epoch + request_index * interval
                    delay = scheduled - time.perf_counter()
                    if delay > 0: time.sleep(delay)
                    rid=(f"{args.mode}-{topology}-{workload}-{load}-"
                         f"trial{trial}-request{request_index}")
                    value=request(args,rid,workload,pids,dids,request_index)
                    value.update({"trial": trial, "arm": f"{args.mode}-{topology}"})
                    futures.append(pool.submit(submit_one, value, scheduled))
                rows=[future.result(timeout=300) for future in futures]
            first_send=min(row["started"] for row in rows)
            cell_end=max(row["completed"] for row in rows)
            for request_index,row in enumerate(rows):
                row.update({"workload":workload,"load":load,"trial":trial,
                            "request_index":request_index,
                            "scheduled_arrival_ns":int(row.pop("scheduled")*1e9),
                            "actual_send_ns":int(row["started"]*1e9),
                            "client_ms":(row["completed"]-row["started"])*1000})
                result["records"].append(row)
            window=cell_end-first_send
            result["cells"].append({"workload":workload,"load":load,
                "trial":trial,"trial_id":f"{args.mode}-{topology}-{workload}-{load}-{trial}",
                "order":cell_order,"offered_interval_ms":LOADS_MS[load],
                "requests":request_count,
                "arrival_train_ms":(request_count-1)*LOADS_MS[load],
                "cache_initial":cache_initial,
                "completed":sum(bool(x["ok"]) for x in rows),
                "throughput_requests_per_s":sum(bool(x["ok"]) for x in rows)/window,
                "output_tokens_per_s":sum(len(x["tokens"]) for x in rows)/window,
                "p50_e2e_ms":statistics.median(x["client_ms"] for x in rows),
                "p95_e2e_ms":percentile([x["client_ms"] for x in rows],95),
                "observation_window_ms":window*1000})
        result["status"]=[x.call({"op":"status"}) for x in workers]
    finally:
        for worker in reversed(workers):
            try: worker.close()
            except Exception as error: result.setdefault("cleanup_errors",[]).append(str(error))
        if vision is not None: vision.terminate(); vision.wait(timeout=30)
        subprocess.run([str(args.build/"demo/pbe_data_service"),"shutdown",endpoint],
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        service.wait(timeout=30); data_log.close(); vision_log.close()
    result["all_outputs_complete"] = all(x["ok"] and x["tokens"] for x in result["records"])
    result["unified_batch_capability_ok"] = (topology != "unified" or
        result.get("unified_batch_capability_probe", {}).get("admitted") == 5 and
        len(result["unified_batch_capability_probe"].get("outputs", [])) == 5)
    result["five_independent_windows_per_cell"] = all(
        sum(cell["workload"] == workload and cell["load"] == load
            for cell in result["cells"]) == args.repeats
        for workload in selected_workloads for load in selected_loads)
    result["requests_per_trial"] = args.requests_per_trial
    result["finite_sustained_overload"] = all(
        cell["offered_interval_ms"] > 0 and cell["arrival_train_ms"] >= 2000
        for cell in result["cells"] if cell["load"] == "near_saturation")
    result["all_workers_executed"] = (topology=="unified" or all(
        any(path in {r["response"]["state"][key] for r in result["records"]}
            for key in ("prefill_worker","decode_worker"))
        for path in pids+dids))
    result["ok"] = (result["all_outputs_complete"] and
                    result["five_independent_windows_per_cell"] and
                    result["finite_sustained_overload"] and
                    result["unified_batch_capability_ok"] and
                    result["all_workers_executed"])
    (trial_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--model-bin",type=Path,required=True); parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True); parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--device",type=int,default=0); parser.add_argument("--mode",choices=("private","shared"),required=True)
    parser.add_argument("--repeats",type=int,default=5,
                        help="independent observation windows per workload/load cell")
    parser.add_argument("--requests-per-trial",type=int,default=5)
    parser.add_argument("--workload", choices=tuple(WORKLOADS) + ("all",),
                        default="all")
    parser.add_argument("--load", choices=tuple(LOADS_MS) + ("all",), default="all")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--topology",choices=tuple(TOPOLOGIES)+("all",),default="all")
    args=parser.parse_args(); args.model_sha256=file_sha256(args.model_bin); args.output_dir.mkdir(parents=True,exist_ok=True)
    selected=TOPOLOGIES if args.topology=="all" else (args.topology,)
    results=[]
    for topology in selected:
        print(json.dumps({"event":"topology_start","mode":args.mode,"topology":topology}),flush=True)
        results.append(run_session(args,topology))
    output={"schema":"pbe-v4-n6-topology-matrix-v2","mode":args.mode,
            "independent_trials_per_cell":args.repeats,
            "requests_per_trial":{"low":args.requests_per_trial,
                "medium":args.requests_per_trial,
                "near_saturation":max(args.requests_per_trial,11)},
            "fixed_total_kv_slots":128,"arrival_seed":20260914,
            "timing":"client submit through full non-streaming response; cleanup resource cost retained",
            "results":results,"ok":all(x["ok"] for x in results)}
    (args.output_dir/f"results-v2-{args.mode}.json").write_text(
        json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"complete","mode":args.mode,"ok":output["ok"]}),flush=True)
    return 0 if output["ok"] else 1

if __name__=="__main__": raise SystemExit(main())
