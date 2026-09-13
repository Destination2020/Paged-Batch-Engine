#!/usr/bin/env python3
"""Freeze the B1 evidence inventory and measurement protocol."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def entry(root: Path, path: str, classification: str, use: str,
          samples: str, missing: list[str], boundary: str) -> dict:
    source = root / path
    return {
        "path": path,
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
        "classification": classification,
        "use": use,
        "sample_scope": samples,
        "missing_fields": missing,
        "conclusion_boundary": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    items = [
        entry(root, "docs/data_flow_evidence/v4/M9/final_results.json",
              "direct_reference", "warm unified versus persistent 1P1D mechanics",
              "5 paired repetitions per arm, 10 requests total",
              ["true client streaming timestamps", "100-request trial"],
              "Valid only for the frozen persistent topology; not a scale claim."),
        entry(root, "docs/data_flow_evidence/v4/M9/kv_cache_ab/results.json",
              "mechanism_only", "semantic KV saved-token and overhead prior",
              "5 repetitions per on/off arm",
              ["feature-cache factorial", "request-level scheduled/send timestamps"],
              "Cannot support B2 factorial or client TTFT by itself."),
        entry(root, "docs/data_flow_evidence/v4/E1/checks.json",
              "mechanism_only", "semantic identity and demand-page correctness",
              "real model correctness/fault gate",
              ["formal B2 performance matrix"],
              "Correctness and saved-compute evidence only."),
        entry(root, "docs/data_flow_evidence/v4/E2_final/results.json",
              "direct_reference", "B4 fixed/round-robin/data-aware comparison",
              "5 trials and 55 requests per policy",
              ["true client streaming timestamps", "open-loop scheduled arrivals"],
              "Use request-completion window throughput and server TTFT/ITL labels."),
        entry(root, "docs/data_flow_evidence/v4/M9/pressure_policy_ab/results.json",
              "direct_reference", "B4 recovery policy boundary",
              "5 repetitions per policy",
              ["100-request trial", "client streaming timestamps"],
              "Shows recovery/recompute cost; no universal tail-latency claim."),
        entry(root, "docs/data_flow_evidence/v4/E3/checks.json",
              "mechanism_only", "dependency lifecycle and exact recovery correctness",
              "10,000 metadata lifecycles plus real GPU recovery",
              ["request-level performance trace for lifecycle loop"],
              "Lifecycle count is not serving throughput."),
        entry(root, "docs/data_flow_evidence/v4/E4/results.json",
              "direct_reference_with_gap", "B3 fixed-KV memory/overhead and capacity mechanism",
              "5 repetitions per private/shared cell",
              ["common hard-budget capacity scan", "last-success/first-failure ladder"],
              "Fixed-KV comparison is quotable; old 64/8192-block capacity result is not a maximum."),
        entry(root, "docs/data_flow_evidence/v4/E4/numerics/checks.json",
              "direct_reference", "B3 BF16 correctness boundary",
              "20 cases, 40 mode outputs, 4 same-history logits audits",
              [], "Numeric acceptance only, not a performance result."),
        entry(root, "docs/data_flow_evidence/v4/E4/completion_manifest.json",
              "direct_reference", "E4 source/binary/model provenance",
              "389 source, 4 binary, 3 model and 10 acceptance identities",
              [], "Historical E4 identity retained; B1 has a new current manifest."),
        entry(root, "docs/data_flow_evidence/v4/M9/persistent_pd_deployment_final_gpu0_v4/results.json",
              "direct_reference", "single-GPU unified versus real 1P1D baseline",
              "5 repetitions per arm",
              ["1P2D", "2P1D", "2P2D"],
              "Does not establish general multi-P/D topology support."),
    ]
    inventory = {
        "schema": "pbe-v4-b1-evidence-inventory-v1",
        "date": "2026-09-13",
        "rules": {
            "historical_files_immutable": True,
            "historical_identity_may_not_enter_current_paired_comparison": True,
            "mechanism_is_not_end_to_end_speedup": True,
            "missing_streaming_timestamps_forbid_client_ttft_claim": True,
        },
        "items": items,
        "gaps_requiring_measurement": [
            "B2 feature-cache x semantic-KV factorial with 0/50/90 percent image reuse",
            "B3 common hard-budget capacity ladder with identical probing algorithm",
        ],
        "unsupported_requires_explicit_status": [
            "B5 production multi-P/D registration, routing and cross-provider page authorization",
        ],
    }
    (output / "evidence_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    gpu_lines = command("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,mig.mode.current",
                        "--format=csv,noheader").splitlines()
    binaries = {}
    for relative in ("demo/pbe_data_service", "demo/pbe_vlm_language_role",
                     "demo/pbe_multi_role_text", "test/test_llm"):
        path = args.build / relative
        binaries[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    model_files = {}
    for path in (args.model, args.model_dir / "config.json",
                 args.model_dir / "preprocessor_config.json",
                 args.model_dir / "tokenizer.json", args.image):
        model_files[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    protocol = {
        "schema": "pbe-v4-performance-protocol-v1",
        "date": "2026-09-13",
        "frozen_before_supplemental_runs": True,
        "source_head": command("git", "rev-parse", "HEAD"),
        "dirty_workspace_preserved": True,
        "hardware": {"gpus": gpu_lines, "mig": "disabled", "mps": "disabled",
                     "exclusive_gpu_measurement": True},
        "software": {"python": platform.python_version(), "kernel": platform.release(),
                     "cuda": command("nvcc", "--version").splitlines()[-1],
                     "build_type": "RelWithDebInfo", "nccl": "off",
                     "cuda_graph": "not used", "cpu_threads": os.cpu_count()},
        "binaries": binaries,
        "inputs": model_files,
        "random_seed": 20260913,
        "metrics": {
            "throughput": "successful requests / (last terminal - first send), including drain",
            "latency_delta_percent": "(baseline-candidate)/baseline*100",
            "throughput_delta_percent": "(candidate-baseline)/baseline*100",
            "memory_units": "MiB=2^20 bytes; raw physical bytes retained",
            "failure_denominator": "offered requests",
            "trial_summary": "median and full min/max across independent trials",
            "confidence_intervals": "not reported: five trials are retained individually",
            "client_ttft": "N/A: coordinator is non-streaming",
            "server_ttft": "model-side first-token timing, never labelled client TTFT",
            "slo_goodput": "N/A without true client streaming and a justified predeclared SLO",
        },
        "b2": {
            "representative": {"image_reuse_percent": 50, "resolution": 224,
                               "context": "short", "arms": ["00", "10", "01", "11"]},
            "boundaries": [
                {"image_reuse_percent": 0, "resolution": 224, "context": "short",
                 "arms": ["00", "11"]},
                {"image_reuse_percent": 90, "resolution": 280, "context": "long",
                 "arms": ["00", "11"]},
            ],
            "arm_bits": "feature_cache,semantic_kv_cache",
            "independent_trials_per_cell": 5,
            "warmup_requests_per_trial": 1,
            "measured_requests_per_trial": 10,
            "sampling": {"max_new_tokens": 8, "repetition_penalty": 1.05},
            "sample_limit_reason": "3B Vision+language topology cold start cost; no reliable p99 claim",
            "order": "fixed seed randomized across all 40 trials",
            "load_generation": "closed-loop, fixed concurrency 1; no open-loop queueing claim",
            "identity_contract": "feature-cache lookup policy may not alter canonical media content or semantic KV identity",
            "numerical_comparison": "pair arm00 and candidate by cell/repetition/request; exact output sequences pass, divergences require first-step same-history M5/E4 logits evidence",
            "deterministic_vision": {
                "PBE_DETERMINISTIC_VISION": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "reason": "keep cache-hit versus BF16 Vision recompute comparisons inside the frozen M5/E4 numerical contract",
            },
        },
        "b3": {
            "fixed_kv_source": "E4 final 2x5 cells",
            "capacity_ladder": "same block ladder and hard physical budget for private/shared",
            "hard_budget_mib": 18000,
            "safety_margin_mib": 256,
            "probe_blocks": [64, 1024, 2048, 4096, 8192, 12288, 16384],
            "repetitions": 5,
            "capacity_limit_claim": "only bracketed by last success and first failure",
        },
        "b4": {"reuse_existing": True, "placement_repeats": 5,
               "placement_requests_per_policy": 55, "recovery_repeats": 5},
        "b5": {"audit_before_measurement": True,
               "manual_process_orchestration_is_not_production_support": True},
        "global": {"warmup_required": True, "cold_start_reported_separately": True,
                   "no_parallel_gpu_experiments": True, "external_gpu_load_policy":
                   "invalidate and rerun the complete paired trial"},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "inventory_items": len(items),
                      "protocol": str(output / "protocol.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
