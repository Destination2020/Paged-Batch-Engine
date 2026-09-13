"""Stable cache identity for a local vision model and processor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pbe_data_client.client import checksum256

IDENTITY_SCHEMA = b"pbe-qwen25-vl-vision-input-v2"
BUNDLE_SCHEMA = b"pbe-qwen25-vl-vision-bf16-bundle-v2"


def model_manifest(model_path: Path) -> bytes:
    model_path = model_path.resolve()
    digest = hashlib.sha256(str(model_path).encode())
    for name in ("config.json", "preprocessor_config.json", "processor_config.json",
                 "tokenizer_config.json", "model.safetensors.index.json"):
        path = model_path / name
        digest.update(name.encode() + b"\0")
        if path.exists(): digest.update(path.read_bytes())
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for name in sorted(set(index.get("weight_map", {}).values())):
            stat = (model_path / name).stat()
            digest.update(name.encode() + b"\0" + f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.digest()


def cache_identity(image_bytes: bytes, size: int, model_path: Path) -> tuple[bytes, bytes, bytes]:
    manifest = model_manifest(model_path)
    content = hashlib.sha256(IDENTITY_SCHEMA+b"\0"+image_bytes+b"\0"+str(size).encode()+b"\0"+manifest).digest()
    representation = checksum256(BUNDLE_SCHEMA)
    return content, representation, manifest


def singleflight_lock_path(endpoint: str, service_incarnation: int, content: bytes) -> Path:
    identity = hashlib.sha256(endpoint.encode()+b"\0"+str(service_incarnation).encode()+b"\0"+content).hexdigest()
    return Path("/tmp") / f"pbe-vision-{identity}.lock"
