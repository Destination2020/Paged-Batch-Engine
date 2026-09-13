#!/usr/bin/env python3
"""Capture a reproducible local manifest for a pinned Qwen2.5-VL snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


PACKAGES = [
    "accelerate", "huggingface-hub", "hf-xet", "numpy", "pillow",
    "safetensors", "torch", "transformers",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        "model-00001-of-00002.safetensors": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
        "model-00002-of-00002.safetensors": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
    }
    files = []
    for path in sorted(p for p in args.model.iterdir() if p.is_file()):
        digest = sha256(path)
        if path.name in expected and digest != expected[path.name]:
            raise ValueError(f"LFS digest mismatch for {path.name}: {digest}")
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest})
    missing = sorted(set(expected) - {entry["path"] for entry in files})
    if missing:
        raise ValueError(f"snapshot is incomplete: {missing}")

    dependencies = {}
    for package in PACKAGES:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    result = {
        "schema": "pbe.model.snapshot.v1",
        "repository": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": args.revision,
        "local_path": str(args.model.resolve()),
        "license_file": "LICENSE",
        "license_name": "Qwen Research License Agreement",
        "files": files,
        "dependencies": dependencies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
