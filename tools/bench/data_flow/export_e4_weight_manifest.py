#!/usr/bin/env python3
"""Export and validate the exact Qwen2 BF16 PBE weight-file layout used by E4."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

LAYOUT = "pbe-qwen2-bf16-v1"
DTYPE_MAGIC = 0x44545950
BF16 = 4
ELEMENT_BYTES = 2


def contiguous_strides(shape):
    strides = []
    value = ELEMENT_BYTES
    for size in reversed(shape):
        strides.append(value)
        value *= size
    return list(reversed(strides))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.model.read_bytes()
    if len(raw) < 36:
        raise SystemExit("model is shorter than the Qwen2 BF16 header")
    dim, hidden, layers, heads, kv_heads, signed_vocab, seq_len = struct.unpack_from(
        "<7i", raw, 0
    )
    magic, dtype_id = struct.unpack_from("<2i", raw, 28)
    if magic != DTYPE_MAGIC or dtype_id != BF16:
        raise SystemExit("E4 manifest accepts only the explicit BF16 PBE layout")
    if min(dim, hidden, layers, heads, kv_heads, abs(signed_vocab), seq_len) <= 0:
        raise SystemExit("invalid model dimensions")
    if dim % heads or heads % kv_heads:
        raise SystemExit("invalid Qwen attention topology")
    vocab = abs(signed_vocab)
    kv_dim = dim * kv_heads // heads
    head_size = dim // heads
    tied = signed_vocab > 0
    cursor = 36
    tensors = []

    def add(name, shape, checkpoint, *, shared=True, alias_of=None, offset=None):
        nonlocal cursor
        count = 1
        for value in shape:
            count *= value
        byte_count = count * ELEMENT_BYTES
        byte_offset = cursor if offset is None else offset
        item = {
            "name": name,
            "shape": shape,
            "dtype": "bf16",
            "stride_bytes": contiguous_strides(shape),
            "required_alignment_bytes": 2,
            "actual_offset_alignment_bytes": byte_offset & -byte_offset,
            "file_offset_bytes": byte_offset,
            "bytes": byte_count,
            "file_interval": [byte_offset, byte_offset + byte_count],
            "source_checkpoint_interval": {
                "tensor": checkpoint,
                "element_start": 0,
                "element_count": count,
            },
            "shared_immutable_parameter": shared,
            "alias_of": alias_of,
        }
        tensors.append(item)
        if offset is None:
            cursor += byte_count
        return byte_offset

    embedding_offset = add("token_embedding.weight", [vocab, dim],
                           "model.embed_tokens.weight")
    for i in range(layers):
        add(f"layers.{i}.attention_norm.weight", [dim],
            f"model.layers.{i}.input_layernorm.weight")
    for i in range(layers):
        add(f"layers.{i}.attention.q_proj.weight", [dim, dim],
            f"model.layers.{i}.self_attn.q_proj.weight")
        add(f"layers.{i}.attention.q_proj.bias", [dim],
            f"model.layers.{i}.self_attn.q_proj.bias")
    for projection in ("k", "v"):
        for i in range(layers):
            add(f"layers.{i}.attention.{projection}_proj.weight", [kv_dim, dim],
                f"model.layers.{i}.self_attn.{projection}_proj.weight")
            add(f"layers.{i}.attention.{projection}_proj.bias", [kv_dim],
                f"model.layers.{i}.self_attn.{projection}_proj.bias")
    for i in range(layers):
        add(f"layers.{i}.attention.o_proj.weight", [dim, dim],
            f"model.layers.{i}.self_attn.o_proj.weight")
    for i in range(layers):
        add(f"layers.{i}.ffn_norm.weight", [dim],
            f"model.layers.{i}.post_attention_layernorm.weight")
    for pbe_name, checkpoint_name, shape in (
        ("gate_proj", "gate_proj", [hidden, dim]),
        ("down_proj", "down_proj", [dim, hidden]),
        ("up_proj", "up_proj", [hidden, dim]),
    ):
        for i in range(layers):
            add(f"layers.{i}.mlp.{pbe_name}.weight", shape,
                f"model.layers.{i}.mlp.{checkpoint_name}.weight")
    add("final_norm.weight", [dim], "model.norm.weight")
    # These immutable export artifacts are covered by the owner slab and checksum,
    # but Qwen2 regenerates the runtime RoPE cache and does not bind either as a
    # shared operator parameter.
    add("export.freqs_cos", [seq_len, head_size // 2], "generated.freqs_cos",
        shared=False)
    add("export.freqs_sin", [seq_len, head_size // 2], "generated.freqs_sin",
        shared=False)
    if tied:
        add("lm_head.weight", [vocab, dim], "lm_head.weight", alias_of="token_embedding.weight",
            offset=embedding_offset)
    else:
        add("lm_head.weight", [vocab, dim], "lm_head.weight")

    physical = [t for t in tensors if t["alias_of"] is None]
    ordered = sorted(physical, key=lambda item: item["file_offset_bytes"])
    previous = 36
    for item in ordered:
        if item["file_offset_bytes"] != previous:
            raise SystemExit(f"gap or overlap before {item['name']}: {previous} != "
                             f"{item['file_offset_bytes']}")
        if item["file_interval"][1] > len(raw):
            raise SystemExit(f"out-of-bounds tensor: {item['name']}")
        previous = item["file_interval"][1]
    if previous != len(raw) or cursor != len(raw):
        raise SystemExit(f"layout coverage mismatch: computed={cursor}, file={len(raw)}")
    for alias in (t for t in tensors if t["alias_of"] is not None):
        target = next(t for t in tensors if t["name"] == alias["alias_of"])
        if alias["file_interval"] != target["file_interval"]:
            raise SystemExit(f"invalid alias interval: {alias['name']}")

    document = {
        "schema": "pbe-e4-weight-manifest-v1",
        "layout_identity": LAYOUT,
        "model_path": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "file_bytes": len(raw),
        "header_bytes": 36,
        "dtype": "bf16",
        "tp": 1,
        "config": {
            "dim": dim, "hidden_dim": hidden, "layers": layers,
            "heads": heads, "kv_heads": kv_heads, "kv_dim": kv_dim,
            "head_size": head_size, "vocab_size": vocab,
            "seq_len": seq_len, "tied_embedding_lm_head": tied,
        },
        "validation": {
            "file_exactly_covered": True,
            "bounds_valid": True,
            "overlap_count_excluding_declared_aliases": 0,
            "declared_alias_count": sum(t["alias_of"] is not None for t in tensors),
            "physical_tensor_ranges": len(physical),
            "logical_tensor_views": len(tensors),
            "immutable_operator_tensor_views": sum(t["shared_immutable_parameter"] for t in tensors),
            "physical_slab_allocations": 1,
            "physical_slab_bytes": len(raw),
        },
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
