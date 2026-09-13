#!/usr/bin/env python3
"""Export the Qwen2.5-VL language core to PBE's compatible BF16 container.

The vision tower remains in its official safetensors files.  This exporter maps
only the language core and writes a versioned sidecar that records every source
key and the model constants that are otherwise represented by the legacy PBE
reader (RMS epsilon in its Qwen build and RoPE theta in the serialized tables).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from safetensors import safe_open


DTYPE_MAGIC = 0x44545950  # DTYP
PBE_BF16 = 4
SCHEMA = "pbe.qwen25_vl.language_core.bf16.v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


class TensorSource:
    def __init__(self, model_dir: Path) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        self.model_dir = model_dir
        self.index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = self.index["weight_map"]

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"checkpoint is missing required tensor {key}")
        path = self.model_dir / self.weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def write_bf16(stream: BinaryIO, tensor: torch.Tensor) -> int:
    values = tensor.detach().contiguous().to(torch.bfloat16).view(torch.uint16).numpy()
    stream.write(values.astype(np.uint16, copy=False).tobytes())
    return int(tensor.numel() * 2)


def language_keys(layers: int) -> list[str]:
    keys = ["model.embed_tokens.weight"]
    keys += [f"model.layers.{i}.input_layernorm.weight" for i in range(layers)]
    for projection in ("q_proj", "k_proj", "v_proj"):
        for i in range(layers):
            keys += [
                f"model.layers.{i}.self_attn.{projection}.weight",
                f"model.layers.{i}.self_attn.{projection}.bias",
            ]
    keys += [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(layers)]
    keys += [f"model.layers.{i}.post_attention_layernorm.weight" for i in range(layers)]
    keys += [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(layers)]
    keys += [f"model.layers.{i}.mlp.down_proj.weight" for i in range(layers)]
    keys += [f"model.layers.{i}.mlp.up_proj.weight" for i in range(layers)]
    keys += ["model.norm.weight"]
    return keys


def write_rope(stream: BinaryIO, head_dim: int, length: int, theta: float) -> int:
    inverse = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(length, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse)
    count = write_bf16(stream, frequencies.cos())
    count += write_bf16(stream, frequencies.sin())
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError(f"expected qwen2_5_vl, got {config.get('model_type')}")
    if config.get("torch_dtype") != "bfloat16":
        raise ValueError("M1 primary export requires the official BF16 checkpoint")
    if not config.get("tie_word_embeddings"):
        raise ValueError("untied lm_head is not supported by this export schema")

    dim = int(config["hidden_size"])
    hidden_dim = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    vocab = int(config["vocab_size"])
    max_length = min(int(config["max_position_embeddings"]), args.max_sequence_length)
    theta = float(config["rope_theta"])
    epsilon = float(config["rms_norm_eps"])
    if epsilon != 1e-6:
        raise ValueError(f"PBE Qwen RMSNorm is compiled for eps=1e-6, checkpoint uses {epsilon}")
    if config.get("rope_scaling", {}).get("type") != "mrope":
        raise ValueError("expected Qwen2.5-VL mrope configuration")

    source = TensorSource(args.model)
    keys = language_keys(layers)
    missing = sorted(set(keys) - set(source.weight_map))
    if missing:
        raise KeyError(f"missing {len(missing)} language tensors: {missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    written: list[dict[str, object]] = []
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<7i", dim, hidden_dim, layers, heads, kv_heads, vocab, max_length))
        stream.write(struct.pack("<2i", DTYPE_MAGIC, PBE_BF16))
        for key in keys:
            tensor = source.tensor(key)
            count = write_bf16(stream, tensor)
            written.append({"key": key, "shape": list(tensor.shape), "bytes": count})
        rope_bytes = write_rope(stream, dim // heads, max_length, theta)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.output)

    language_checkpoint_files = sorted({source.weight_map[key] for key in keys})
    sidecar = {
        "schema": SCHEMA,
        "container_reader": "legacy ModelConfig + DTYP BF16 header",
        "model_type": config["model_type"],
        "dimensions": {
            "hidden_size": dim,
            "intermediate_size": hidden_dim,
            "layers": layers,
            "attention_heads": heads,
            "kv_heads": kv_heads,
            "vocab_size": vocab,
            "max_sequence_length": max_length,
        },
        "numerics": {
            "weight_dtype": "bfloat16",
            "rms_norm_eps": epsilon,
            "rope_theta": theta,
            "rope_type": "mrope",
            "mrope_section": config["rope_scaling"]["mrope_section"],
            "qkv_bias": True,
            "tie_word_embeddings": True,
        },
        "vision_weights_included": False,
        "tensors": written,
        "serialized_rope_bytes": rope_bytes,
        "source_files": [
            {"path": name, "sha256": sha256(args.model / name)}
            for name in language_checkpoint_files
        ],
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
    }
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sidecar": str(sidecar_path)}, indent=2))


if __name__ == "__main__":
    main()
