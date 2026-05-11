#!/usr/bin/env python3
import argparse
from time import perf_counter


def build_chatml_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def metric_value(metrics, *names: str) -> float:
    for name in names:
        value = float(getattr(metrics, name, 0.0) or 0.0)
        if value > 0.0:
            return value
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph capture/replay while keeping vLLM compile enabled.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-rounds", type=int, default=1)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": not args.disable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": False,
    }
    if args.disable_cuda_graph:
        from vllm.config import CompilationConfig
        from vllm.config.compilation import CUDAGraphMode, CompilationMode

        llm_kwargs["compilation_config"] = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        )
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        skip_reading_prefix_cache=args.disable_prefix_caching,
    )
    formatted_prompt = build_chatml_prompt(args.prompt)
    for _ in range(args.warmup_rounds):
        llm.generate([formatted_prompt], sampling, use_tqdm=False)

    start = perf_counter()
    outputs = llm.generate([formatted_prompt], sampling, use_tqdm=False)
    wall_s = perf_counter() - start

    output = outputs[0]
    completion = output.outputs[0]
    prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
    output_tokens = len(completion.token_ids)
    throughput = output_tokens / wall_s if wall_s > 0.0 else 0.0

    metrics = getattr(output, "metrics", None)
    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    ttft_ms = first_token_latency * 1000.0 if first_token_latency > 0.0 else 0.0
    first_token_ts = metric_value(metrics, "first_token_ts", "first_token_time")
    last_token_ts = metric_value(metrics, "last_token_ts", "last_token_time")
    if output_tokens > 1 and first_token_ts > 0.0 and last_token_ts > first_token_ts:
        itl_ms = (last_token_ts - first_token_ts) * 1000.0 / (output_tokens - 1)
    else:
        itl_ms = 0.0

    print("=== vLLM Single Request ===")
    print(f"prompt={args.prompt!r}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"output_tokens={output_tokens}")
    print(f"wall_seconds={wall_s:.6f}")
    print(f"throughput_tps={throughput:.3f}")
    print(f"ttft_ms={ttft_ms:.3f}")
    print(f"itl_ms={itl_ms:.3f}")
    print(f"e2e_latency_ms={wall_s * 1000.0:.3f}")
    print("=== Output ===")
    print(completion.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
