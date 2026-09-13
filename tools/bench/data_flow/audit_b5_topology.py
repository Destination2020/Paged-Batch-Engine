#!/usr/bin/env python3
"""Audit production multi-P/D support without treating manual workers as serving support."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines_with(path, needles):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if any(needle in line for needle in needles):
            rows.append({"line": number, "text": line.strip()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    coordinator = root / "python/pbe_roles/coordinator.py"
    role = root / "demo/pbe_vlm_language_role.cpp"
    runner = root / "tools/bench/data_flow/run_persistent_pd_deployment_ab.py"
    validation_path = root / "docs/data_flow_evidence/v4/M9/persistent_pd_deployment_final_gpu0_v4/validation.json"
    validation = json.loads(validation_path.read_text())
    result = {
        "schema": "pbe-v4-b5-topology-support-audit-v1",
        "accepted": False,
        "status": "blocked_by_missing_production_coordination",
        "validated_existing": {
            "unified_1p1d_process": {"status": "validated_baseline", "source": str(validation_path.relative_to(root))},
            "split_1p1d": {"status": "validated", "source": str(validation_path.relative_to(root)),
                            "gate_ok": validation.get("ok") is True},
        },
        "matrix": {
            "unified": "validated historical baseline",
            "1P1D": "validated real persistent handoff",
            "1P2D": "unsupported in production Coordinator",
            "2P1D": "unsupported in production Coordinator",
            "2P2D": "unsupported in production Coordinator",
        },
        "blocking_contracts": [
            "Coordinator constructs exactly one language process and has no Prefill/Decode role registry",
            "No production request router selects among multiple Prefill providers or Decode consumers",
            "Decode page authorization is a static initial-slot vector fixed at process construction",
            "A Decode worker cannot dynamically add a later Prefill provider generation/slot grant",
            "No validator proves all P-to-D paths, four-process participation, cancellation and recovery",
        ],
        "why_manual_probe_is_insufficient":
            "Starting extra LanguageProcess instances in a benchmark could exercise IPC, but would not prove the production launcher, registry, routing, authorization, or lifecycle contract required by B5.",
        "code_evidence": {
            "coordinator": {"path": str(coordinator.relative_to(root)), "sha256": sha(coordinator),
                            "matches": lines_with(coordinator, ["language = LanguageProcess", "run_round(language"])},
            "language_role": {"path": str(role.relative_to(root)), "sha256": sha(role),
                              "matches": lines_with(role, ["initial_slots_", "binary_search(initial_slots_"])},
            "persistent_runner": {"path": str(runner.relative_to(root)), "sha256": sha(runner),
                                  "matches": lines_with(runner, ["prefill = worker", "decode = worker"])},
        },
        "next_implementation_entry": [
            "typed role registration with incarnation and capability",
            "dynamic provider-generation page authorization independent of process startup",
            "request routing across P/D pairs with idempotent lifecycle rollback",
            "then run private/shared 1P2D, 2P1D and 2P2D five-trial cells",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": False, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
