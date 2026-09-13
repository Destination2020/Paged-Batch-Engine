#!/usr/bin/env python3
"""Freeze Qwen2.5-VL processor, vision, mRoPE, and language reference tensors.

This is an offline numerical probe.  It intentionally keeps Transformers out of
the serving path and writes portable JSON/NPZ fixtures for the C++ adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def resized(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def conversations(source: Image.Image) -> dict[str, list[list[dict[str, Any]]]]:
    text = "Describe the image briefly and name its main subject."
    image_224 = resized(source, (224, 224))
    image_wide = resized(source, (280, 168))
    return {
        "text": [[{"role": "user", "content": [{"type": "text", "text": text}]}]],
        "one_image_224": [[{"role": "user", "content": [
            {"type": "image", "image": image_224}, {"type": "text", "text": text}
        ]}]],
        "one_image_wide": [[{"role": "user", "content": [
            {"type": "image", "image": image_wide}, {"type": "text", "text": text}
        ]}]],
        "two_images": [[{"role": "user", "content": [
            {"type": "image", "image": image_224},
            {"type": "text", "text": "Compare this with the next image."},
            {"type": "image", "image": image_wide},
            {"type": "text", "text": text},
        ]}]],
        "mixed_batch": [
            [{"role": "user", "content": [
                {"type": "image", "image": image_224}, {"type": "text", "text": text}
            ]}],
            [{"role": "user", "content": [
                {"type": "image", "image": image_wide}, {"type": "text", "text": text}
            ]}],
        ],
    }


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    data = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "min": float(data.min()) if data.numel() else None,
        "max": float(data.max()) if data.numel() else None,
        "mean": float(data.mean()) if data.numel() else None,
    }


def make_inputs(processor: Any, messages: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
    value = processor.apply_chat_template(
        messages if len(messages) > 1 else messages[0],
        tokenize=True,
        add_generation_prompt=True,
        padding=len(messages) > 1,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(value)


def probe_case(
    name: str,
    messages: list[list[dict[str, Any]]],
    processor: Any,
    model: Qwen2_5_VLForConditionalGeneration,
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    cpu_inputs = make_inputs(processor, messages)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in cpu_inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    mm_types = inputs.get("mm_token_type_ids")
    image_grid = inputs.get("image_grid_thw")

    embeddings = model.model.get_input_embeddings()(input_ids)
    image_features = None
    if "pixel_values" in inputs:
        image_features = model.model.get_image_features(
            inputs["pixel_values"], image_grid, return_dict=True
        ).pooler_output
        image_features = torch.cat(image_features, dim=0).to(embeddings.device, embeddings.dtype)
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=embeddings, image_features=image_features
        )
        embeddings = embeddings.masked_scatter(image_mask, image_features)

    if mm_types is not None and image_grid is not None:
        position_ids, rope_deltas = model.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=image_grid,
            attention_mask=attention_mask,
        )
    else:
        base = attention_mask.long().cumsum(-1) - 1
        base.masked_fill_(attention_mask == 0, 0)
        position_ids = base.unsqueeze(0).expand(3, -1, -1)
        rope_deltas = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=device)

    with torch.inference_mode():
        language = model.model.language_model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = model.lm_head(language.last_hidden_state)
        generation_args = {
            key: value for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        }
        generated_output = model.generate(
            **generation_args,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        generated_a = generated_output.sequences
        generated_b = model.generate(**generation_args, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_lengths = attention_mask.sum(-1).tolist()
    generation_start = input_ids.shape[1]
    generated_tokens = [
        generated_a[row, generation_start:].detach().cpu().tolist()
        for row in range(generated_a.shape[0])
    ]
    repeated_tokens = [
        generated_b[row, generation_start:].detach().cpu().tolist()
        for row in range(generated_b.shape[0])
    ]
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    top_values, top_indices = torch.topk(logits[:, -1, :].float(), k=10, dim=-1)

    arrays: dict[str, np.ndarray] = {
        "input_ids": input_ids.detach().cpu().numpy(),
        "attention_mask": attention_mask.detach().cpu().numpy(),
        "position_ids": position_ids.detach().cpu().numpy(),
        "rope_deltas": rope_deltas.detach().cpu().numpy(),
        "inputs_embeds": to_numpy(embeddings),
        "last_logits": to_numpy(logits[:, -1, :]),
        "hidden_0": to_numpy(language.hidden_states[0]),
        "hidden_18": to_numpy(language.hidden_states[len(language.hidden_states) // 2]),
        "hidden_final": to_numpy(language.hidden_states[-1]),
        "generation_logits": to_numpy(torch.stack(generated_output.logits, dim=1)),
    }
    cache_layers = language.past_key_values.layers
    for layer_index in (0, len(cache_layers) // 2, len(cache_layers) - 1):
        arrays[f"kv_key_{layer_index}"] = to_numpy(cache_layers[layer_index].keys)
        arrays[f"kv_value_{layer_index}"] = to_numpy(cache_layers[layer_index].values)
    if mm_types is not None:
        arrays["mm_token_type_ids"] = mm_types.detach().cpu().numpy()
    if image_grid is not None:
        arrays["image_grid_thw"] = image_grid.detach().cpu().numpy()
    if image_features is not None:
        arrays["image_features"] = to_numpy(image_features)

    fixture_path = output_dir / f"{name}.npz"
    np.savez_compressed(fixture_path, **arrays)
    raw_paths = {
        "input_ids_i32": output_dir / f"{name}.input_ids.i32.bin",
        "position_ids_i32": output_dir / f"{name}.position_ids.i32.bin",
        "inputs_embeds_bf16": output_dir / f"{name}.inputs_embeds.bf16.bin",
        "last_logits_f32": output_dir / f"{name}.last_logits.f32.bin",
        "generation_logits_f32": output_dir / f"{name}.generation_logits.f32.bin",
    }
    for tensor_name in ("hidden_0", "hidden_18", "hidden_final",
                        "kv_key_0", "kv_value_0", "kv_key_18", "kv_value_18",
                        "kv_key_35", "kv_value_35"):
        raw_paths[f"{tensor_name}_f32"] = output_dir / f"{name}.{tensor_name}.f32.bin"
    raw_paths["input_ids_i32"].write_bytes(
        input_ids.detach().cpu().numpy().astype("<i4", copy=False).tobytes()
    )
    raw_paths["position_ids_i32"].write_bytes(
        position_ids.detach().cpu().numpy().astype("<i4", copy=False).tobytes()
    )
    raw_paths["inputs_embeds_bf16"].write_bytes(
        embeddings.detach().cpu().contiguous().view(torch.uint16).numpy()
        .astype("<u2", copy=False).tobytes()
    )
    raw_paths["last_logits_f32"].write_bytes(
        logits[:, -1, :].detach().float().cpu().numpy().astype("<f4", copy=False).tobytes()
    )
    raw_paths["generation_logits_f32"].write_bytes(
        torch.stack(generated_output.logits, dim=1).detach().float().cpu().numpy()
        .astype("<f4", copy=False).tobytes()
    )
    for tensor_name in ("hidden_0", "hidden_18", "hidden_final",
                        "kv_key_0", "kv_value_0", "kv_key_18", "kv_value_18",
                        "kv_key_35", "kv_value_35"):
        raw_paths[f"{tensor_name}_f32"].write_bytes(
            arrays[tensor_name].astype("<f4", copy=False).tobytes()
        )
    generation_top2 = torch.topk(torch.stack(generated_output.scores, dim=1).float(), k=2, dim=-1)
    return {
        "case": name,
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "prompt_lengths": prompt_lengths,
        "input_keys": sorted(cpu_inputs),
        "input_tensors": {key: tensor_summary(value) for key, value in cpu_inputs.items()},
        "image_feature": tensor_summary(image_features) if image_features is not None else None,
        "position_axis_equal_for_text": bool(torch.equal(position_ids[0], position_ids[1]) and torch.equal(position_ids[1], position_ids[2])),
        "rope_deltas": rope_deltas.detach().cpu().flatten().tolist(),
        "last_logits_top10_ids": top_indices.detach().cpu().tolist(),
        "last_logits_top10_values": top_values.detach().cpu().tolist(),
        "generated_token_ids": generated_tokens,
        "greedy_top2_margin": (generation_top2.values[..., 0] - generation_top2.values[..., 1])
        .detach().cpu().tolist(),
        "repeated_token_ids": repeated_tokens,
        "repeat_exact": generated_tokens == repeated_tokens,
        "decoded": decoded,
        "fixture": fixture_path.name,
        "fixture_sha256": sha256(fixture_path),
        "raw_fixtures": {
            key: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in raw_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[
        "text", "one_image_224", "one_image_wide", "two_images", "mixed_batch"
    ])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.case or ["text", "one_image_224", "one_image_wide", "two_images", "mixed_batch"]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    # Decoder-only batched generation requires left padding so the final row is
    # the prompt tail for every request.
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    source = Image.open(args.image).convert("RGB")
    cases = conversations(source)
    results = [
        probe_case(name, cases[name], processor, model, args.output_dir, args.max_new_tokens)
        for name in selected
    ]
    manifest = {
        "schema": "pbe.qwen25_vl.reference.v1",
        "model_path": str(args.model.resolve()),
        "image_path": str(args.image.resolve()),
        "image_sha256": sha256(args.image),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "max_new_tokens": args.max_new_tokens,
        "cases": results,
    }
    output = args.output_dir / "reference.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "cases": [r["case"] for r in results]}, indent=2))


if __name__ == "__main__":
    main()
