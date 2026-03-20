# Updated on March 19, 2026
"""
Export a local HuggingFace Qwen1.5-MoE style model into the fp32 binary
layout expected by HighPerInfFram's QwenMoeModel.
"""

import argparse
import json
import struct
from pathlib import Path

import torch


K_MOE_MAGIC = 0x4D4F4531


def serialize_fp32(file, tensor):
    """Write one tensor to file in flattened fp32 format."""
    tensor.detach().cpu().to(torch.float32).contiguous().view(-1).numpy().tofile(file)


def expect_shape(name, tensor, expected_shape):
    actual_shape = tuple(tensor.shape)
    expected_shape = tuple(expected_shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )


def resolve_local_model_dir(hf_arg):
    model_dir = Path(hf_arg).expanduser()
    if not model_dir.is_absolute():
        model_dir = model_dir.resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_dir}. "
            "--hf only accepts a local model directory."
        )
    return model_dir


def load_local_config_json(model_dir):
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under local model dir: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_moe_config(config_json, config):
    model_type = config_json.get("model_type", getattr(config, "model_type", None))
    num_experts = int(getattr(config, "num_experts", config_json.get("num_experts", 0)) or 0)
    top_k = int(
        getattr(config, "num_experts_per_tok", config_json.get("num_experts_per_tok", 0)) or 0
    )
    moe_hidden = int(
        getattr(config, "moe_intermediate_size", config_json.get("moe_intermediate_size", 0)) or 0
    )
    if model_type != "qwen2_moe" or num_experts <= 0 or top_k <= 0 or moe_hidden <= 0:
        raise ValueError(
            "This exporter only supports local Qwen MoE models such as "
            "Qwen1.5-MoE-A2.7B. Dense Qwen models are not supported."
        )


def get_required_attr(config, name):
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"Missing required config attribute: {name}")
    return value


def load_model(model_dir):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "transformers is required for export. Please run `pip install transformers`."
        ) from exc

    config_json = load_local_config_json(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    validate_moe_config(config_json, config)

    print(f"loading model from local dir: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    return model, config


def write_headers(file, config):
    dim = int(get_required_attr(config, "hidden_size"))
    hidden_dim = int(get_required_attr(config, "intermediate_size"))
    layer_num = int(get_required_attr(config, "num_hidden_layers"))
    head_num = int(get_required_attr(config, "num_attention_heads"))
    kv_head_num = int(get_required_attr(config, "num_key_value_heads"))
    vocab_size = int(get_required_attr(config, "vocab_size"))
    seq_len = int(get_required_attr(config, "max_position_embeddings"))
    tie_word_embeddings = bool(get_required_attr(config, "tie_word_embeddings"))

    model_header = struct.pack(
        "iiiiiii",
        dim,
        hidden_dim,
        layer_num,
        head_num,
        kv_head_num,
        vocab_size if tie_word_embeddings else -vocab_size,
        seq_len,
    )
    file.write(model_header)

    shared_hidden_dim = int(
        getattr(config, "shared_expert_intermediate_size", 0) or 0
    )
    moe_header = struct.pack(
        "iiiiiiii",
        K_MOE_MAGIC,
        int(get_required_attr(config, "num_experts")),
        int(get_required_attr(config, "num_experts_per_tok")),
        1 if shared_hidden_dim > 0 else 0,
        int(get_required_attr(config, "moe_intermediate_size")),
        shared_hidden_dim,
        int(getattr(config, "decoder_sparse_step", 1) or 1),
        int(bool(getattr(config, "norm_topk_prob", False))),
    )
    file.write(moe_header)


def export_qwen_moe(model, config, output_path):
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    vocab_size = int(config.vocab_size)
    num_experts = int(config.num_experts)
    moe_hidden_dim = int(config.moe_intermediate_size)
    shared_hidden_dim = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
    tie_word_embeddings = bool(config.tie_word_embeddings)
    kv_dim = hidden_size * num_kv_heads // num_heads

    core_model = model.model
    layers = core_model.layers
    if len(layers) != num_layers:
        raise ValueError(f"layer count mismatch: config={num_layers}, model={len(layers)}")

    output_path = Path(output_path)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        write_headers(file, config)

        expect_shape("model.embed_tokens.weight", core_model.embed_tokens.weight, (vocab_size, hidden_size))
        serialize_fp32(file, core_model.embed_tokens.weight)

        for layer_idx, layer in enumerate(layers):
            prefix = f"model.layers.{layer_idx}"

            expect_shape(f"{prefix}.input_layernorm.weight", layer.input_layernorm.weight, (hidden_size,))
            serialize_fp32(file, layer.input_layernorm.weight)

            expect_shape(f"{prefix}.self_attn.q_proj.weight", layer.self_attn.q_proj.weight, (hidden_size, hidden_size))
            expect_shape(f"{prefix}.self_attn.q_proj.bias", layer.self_attn.q_proj.bias, (hidden_size,))
            serialize_fp32(file, layer.self_attn.q_proj.weight)
            serialize_fp32(file, layer.self_attn.q_proj.bias)

            expect_shape(f"{prefix}.self_attn.k_proj.weight", layer.self_attn.k_proj.weight, (kv_dim, hidden_size))
            expect_shape(f"{prefix}.self_attn.k_proj.bias", layer.self_attn.k_proj.bias, (kv_dim,))
            serialize_fp32(file, layer.self_attn.k_proj.weight)
            serialize_fp32(file, layer.self_attn.k_proj.bias)

            expect_shape(f"{prefix}.self_attn.v_proj.weight", layer.self_attn.v_proj.weight, (kv_dim, hidden_size))
            expect_shape(f"{prefix}.self_attn.v_proj.bias", layer.self_attn.v_proj.bias, (kv_dim,))
            serialize_fp32(file, layer.self_attn.v_proj.weight)
            serialize_fp32(file, layer.self_attn.v_proj.bias)

            expect_shape(f"{prefix}.self_attn.o_proj.weight", layer.self_attn.o_proj.weight, (hidden_size, hidden_size))
            serialize_fp32(file, layer.self_attn.o_proj.weight)

            expect_shape(
                f"{prefix}.post_attention_layernorm.weight",
                layer.post_attention_layernorm.weight,
                (hidden_size,),
            )
            serialize_fp32(file, layer.post_attention_layernorm.weight)

            expect_shape(f"{prefix}.mlp.gate.weight", layer.mlp.gate.weight, (num_experts, hidden_size))
            serialize_fp32(file, layer.mlp.gate.weight)

            for expert_idx in range(num_experts):
                expert = layer.mlp.experts[expert_idx]
                expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"
                expect_shape(f"{expert_prefix}.gate_proj.weight", expert.gate_proj.weight, (moe_hidden_dim, hidden_size))
                expect_shape(f"{expert_prefix}.down_proj.weight", expert.down_proj.weight, (hidden_size, moe_hidden_dim))
                expect_shape(f"{expert_prefix}.up_proj.weight", expert.up_proj.weight, (moe_hidden_dim, hidden_size))
                serialize_fp32(file, expert.gate_proj.weight)
                serialize_fp32(file, expert.down_proj.weight)
                serialize_fp32(file, expert.up_proj.weight)

            if shared_hidden_dim > 0:
                expect_shape(
                    f"{prefix}.mlp.shared_expert.gate_proj.weight",
                    layer.mlp.shared_expert.gate_proj.weight,
                    (shared_hidden_dim, hidden_size),
                )
                expect_shape(
                    f"{prefix}.mlp.shared_expert.down_proj.weight",
                    layer.mlp.shared_expert.down_proj.weight,
                    (hidden_size, shared_hidden_dim),
                )
                expect_shape(
                    f"{prefix}.mlp.shared_expert.up_proj.weight",
                    layer.mlp.shared_expert.up_proj.weight,
                    (shared_hidden_dim, hidden_size),
                )
                expect_shape(
                    f"{prefix}.mlp.shared_expert_gate.weight",
                    layer.mlp.shared_expert_gate.weight,
                    (1, hidden_size),
                )
                serialize_fp32(file, layer.mlp.shared_expert.gate_proj.weight)
                serialize_fp32(file, layer.mlp.shared_expert.down_proj.weight)
                serialize_fp32(file, layer.mlp.shared_expert.up_proj.weight)
                serialize_fp32(file, layer.mlp.shared_expert_gate.weight)

        expect_shape("model.norm.weight", core_model.norm.weight, (hidden_size,))
        serialize_fp32(file, core_model.norm.weight)

        if not tie_word_embeddings:
            expect_shape("lm_head.weight", model.lm_head.weight, (vocab_size, hidden_size))
            serialize_fp32(file, model.lm_head.weight)

    print(f"wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a local Qwen1.5-MoE style HF model into HighPerInfFram fp32 bin format."
    )
    parser.add_argument("filepath", help="output .bin file path")
    parser.add_argument(
        "--hf",
        required=True,
        help="local HF model directory path; repo ids are not supported",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        model_dir = resolve_local_model_dir(args.hf)
        model, config = load_model(model_dir)
        export_qwen_moe(model, config, args.filepath)
    except Exception as exc:
        raise SystemExit(f"export failed: {exc}") from exc


if __name__ == "__main__":
    main()
