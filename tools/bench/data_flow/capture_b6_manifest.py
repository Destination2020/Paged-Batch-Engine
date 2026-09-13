#!/usr/bin/env python3
"""Capture current source, binary, input, environment and evidence identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def run(root: Path, *command: str) -> str:
    return subprocess.check_output(command, cwd=root, text=True,
                                   stderr=subprocess.STDOUT).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".sh"}
    source_files = []
    for base in ("CMakeLists.txt", "demo", "infMain", "python", "test", "tools"):
        path = root / base
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if (candidate.is_file() and
                    (candidate.name == "CMakeLists.txt" or candidate.suffix in suffixes) and
                    "__pycache__" not in candidate.parts):
                source_files.append(capture(candidate.relative_to(root)))
    source_files.sort(key=lambda item: item["path"])

    binaries = [capture(args.build / relative) for relative in (
        "demo/pbe_data_service", "demo/pbe_vlm_language_role",
        "demo/pbe_multi_role_text", "test/test_llm")]
    inputs = [capture(path) for path in (
        args.model, args.model_dir / "config.json",
        args.model_dir / "preprocessor_config.json",
        args.model_dir / "tokenizer.json", args.image)]
    evidence_files = []
    for path in sorted(out.rglob("*")):
        if (path.is_file() and path.name not in {"manifest.json", "workspace_state.patch"} and
                "trials" not in path.parts and not path.name.endswith(".log")):
            evidence_files.append(capture(path))

    diff = subprocess.run(["git", "diff", "--binary", "--", ":!build-v3"], cwd=root,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=True).stdout
    (out / "workspace_state.patch").write_text(diff, encoding="utf-8")
    status = run(root, "git", "status", "--short")
    untracked = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = root / line[3:]
        if path.is_file():
            untracked.append(capture(path.relative_to(root)))
    manifest = {
        "schema": "pbe-v4-b6-manifest-v1",
        "source_head": run(root, "git", "rev-parse", "HEAD"),
        "workspace_dirty": bool(status),
        "workspace_status": status.splitlines(),
        "workspace_patch": capture(out / "workspace_state.patch"),
        "untracked_regular_files_visible_in_status": untracked,
        "sources": source_files,
        "binaries": binaries,
        "inputs": inputs,
        "evidence": evidence_files,
        "environment": {
            "python": platform.python_version(),
            "kernel": platform.release(),
            "cuda": run(root, "nvcc", "--version").splitlines()[-1],
            "gpu": run(root, "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,mig.mode.current", "--format=csv,noheader").splitlines(),
        },
        "notes": [
            "Historical manifests are not overwritten.",
            "Trial directories are referenced by aggregate evidence and commands rather than duplicated in this manifest.",
            "The workspace patch covers tracked changes; current-source hashes and the status listing cover the dirty workspace identity.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "sources": len(source_files),
                      "evidence": len(evidence_files), "dirty": bool(status)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
