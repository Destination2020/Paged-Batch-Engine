#!/usr/bin/env python3
"""Freeze E4 source, binary, model and acceptance evidence identities."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path, root):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    try: name = str(path.relative_to(root))
    except ValueError: name = str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": value.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others",
        "--exclude-standard", "CMakeLists.txt", "demo", "infMain", "python", "test",
        "tools/bench/data_flow", "tools/launch"], cwd=root, text=True).splitlines()
    sources = [root / name for name in sorted(set(names)) if (root / name).is_file() and
               not name.startswith("build-")]
    evidence_names = [
        "docs/PBE_MULTI_ROLE_MULTIMODAL_EXECUTION_PLAN_V4_20260912.md",
        "docs/V4_E4_SHARED_WEIGHT_COMPLETION_20260913.md",
        "docs/data_flow_evidence/v4/E4/baseline_manifest.json",
        "docs/data_flow_evidence/v4/E4/weight_manifest.json",
        "docs/data_flow_evidence/v4/E4/results.json",
        "docs/data_flow_evidence/v4/E4/checks.json",
        "docs/data_flow_evidence/v4/E4/faults/results.json",
        "docs/data_flow_evidence/v4/E4/numerics/checks.json",
        "docs/data_flow_evidence/v4/E4/regression/checks.json",
        "docs/data_flow_evidence/v4/E4/sanitizer/results.json",
    ]
    evidence = [root / name for name in evidence_names]
    missing = [str(path) for path in evidence if not path.is_file()]
    if missing: raise SystemExit("missing acceptance files: " + ", ".join(missing))
    result = {"schema": "pbe-e4-completion-manifest-v1", "date": "2026-09-13",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
        "dirty_workspace_preserved": True,
        "source_files": [digest(path, root) for path in sources],
        "build_artifacts": [digest(args.build / name, root) for name in
                            ("lib/libllama.so", "demo/pbe_data_service",
                             "demo/pbe_vlm_language_role", "test/test_llm")],
        "model_files": [digest(path, root) for path in
                        (args.model, args.config, args.tokenizer)],
        "acceptance_files": [digest(path, root) for path in evidence],
        "replay_commands": [
            "PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_shared_weight_ab.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-dir MODEL_DIR --image IMAGE --device 0 --repeats 5 --output docs/data_flow_evidence/v4/E4/results.json",
            "python3 tools/bench/data_flow/validate_e4_shared_weight.py --evidence docs/data_flow_evidence/v4/E4 --model MODEL --output docs/data_flow_evidence/v4/E4/checks.json --required-repeats 5",
            "PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_fault_matrix.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-sha256 SHA256 --device 0 --output docs/data_flow_evidence/v4/E4/faults/results.json",
            "PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_compute_sanitizer.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-dir MODEL_DIR --image IMAGE --device 0 --output docs/data_flow_evidence/v4/E4/sanitizer/results.json",
            "python3 tools/bench/data_flow/validate_e4_numerics.py --private PRIVATE_RESULT --shared SHARED_RESULT --private-validation PRIVATE_GATE --shared-validation SHARED_GATE --cross-validation CROSS_GATE --output docs/data_flow_evidence/v4/E4/numerics/checks.json",
            "build-v3/test/test_llm --gtest_color=no",
            "PBE_RUN_DIR=EVIDENCE_DIR tools/launch/multi_role_text.sh build-v3 TEXT_MODEL TEXT_TOKENIZER",
            "python3 tools/bench/data_flow/validate_e4_regression.py --directory docs/data_flow_evidence/v4/E4/regression --output docs/data_flow_evidence/v4/E4/regression/checks.json",
        ]}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"ok": True, "source_files": len(sources),
                      "acceptance_files": len(evidence)}, sort_keys=True))


if __name__ == "__main__": main()
