#!/usr/bin/env python3
import argparse
import statistics
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from report_utils import (
    build_report_context,
    dump_json,
    format_metric,
    load_workloads,
    parse_workload_names,
    summarize_runs,
)


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


def extract_request_latency_ms(metrics, token_count: int) -> tuple[float, float]:
    if metrics is None:
        return 0.0, 0.0

    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    ttft_ms = first_token_latency * 1000.0

    first_token_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
    last_token_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
    generation_tokens = int(getattr(metrics, "num_generation_tokens", 0) or 0)
    decode_token_count = generation_tokens if generation_tokens > 0 else token_count

    if decode_token_count <= 1 or first_token_ts <= 0.0 or last_token_ts <= 0.0:
        return ttft_ms, 0.0

    decode_window_ms = (last_token_ts - first_token_ts) * 1000.0
    return ttft_ms, decode_window_ms / (decode_token_count - 1)


def render_markdown(context, workload_reports: list[dict]) -> str:
    lines = [
        "# vLLM Baseline Report",
        "",
        "## Metadata",
        "",
        f"- Created at: `{context.created_at}`",
        f"- Git branch: `{context.git_branch}`",
        f"- Git commit: `{context.git_commit}`",
        f"- GPU: `{context.gpu_name}`",
        f"- GPU memory (MB): `{context.gpu_memory_mb}`",
        f"- Driver: `{context.driver_version}`",
        f"- Model: `{context.model_path}`",
        f"- Tokenizer: `{context.tokenizer_path}`",
        f"- Workloads: `{context.prompts_path}`",
        f"- Runs: `{context.runs}`",
        "- Prompt format: `ChatML wrapper matched to serving_qwen`",
        "- Prefix caching: `disabled for parity with current PagedBatchEngine stage`",
        "",
    ]

    for workload in workload_reports:
        aggregate = workload["aggregate"]
        lines.extend(
            [
                f"## Workload `{workload['name']}`",
                "",
                f"- Description: `{workload['description']}`",
                f"- Prompt count: `{len(workload['prompts'])}`",
                f"- Max new tokens: `{workload['config']['max_new_tokens']}`",
                f"- Throughput mean: `{format_metric(aggregate['throughput_tps']['mean'], 'tokens/s')}`",
                f"- TTFT mean: `{format_metric(aggregate['ttft_ms']['mean'], 'ms')}`",
                f"- ITL mean: `{format_metric(aggregate['itl_ms']['mean'], 'ms')}`",
                "",
                "| Run | Throughput (tokens/s) | TTFT mean (ms) | ITL mean (ms) | Requests | Decode Tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, payload in enumerate(workload["runs"], start=1):
            lines.append(
                f"| {idx} | {payload['throughput_tps']:.3f} | {payload['ttft_ms']:.3f} | "
                f"{payload['itl_ms']:.3f} | {payload['request_count']} | {payload['decode_tokens']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=False)
    parser.add_argument("--prompts", default="tools/bench/workloads.json")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = load_workloads(
        (repo_root / args.prompts).resolve(),
        selected_names=parse_workload_names(args.workloads),
        default_name="default",
        default_description="Legacy prompt-list workload.",
        max_new_tokens=args.max_new_tokens,
    )

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.max_new_tokens,
        skip_reading_prefix_cache=True,
    )

    context = build_report_context(
        repo_root=repo_root,
        output_dir=output_dir,
        model_path=args.model,
        tokenizer_path=args.tokenizer or args.model,
        prompts_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
        runs=args.runs,
    )

    workload_reports = []
    for workload in workloads:
        run_payloads = []
        throughput_values = []
        ttft_values = []
        itl_values = []
        formatted_prompts = [build_chatml_prompt(prompt) for prompt in workload["prompts"]]

        sampling_params.max_tokens = workload["max_new_tokens"]

        for run_idx in range(args.runs):
            start = perf_counter()
            outputs = llm.generate(formatted_prompts, sampling_params)
            wall_s = perf_counter() - start

            per_request_ttft_ms = []
            per_request_itl_ms = []
            total_decode_tokens = 0

            for output in outputs:
                metrics = getattr(output, "metrics", None)
                completion = output.outputs[0]
                token_count = len(completion.token_ids)
                total_decode_tokens += token_count
                ttft_ms, itl_ms = extract_request_latency_ms(metrics, token_count)
                if ttft_ms > 0.0:
                    per_request_ttft_ms.append(ttft_ms)
                if itl_ms > 0.0:
                    per_request_itl_ms.append(itl_ms)

            throughput = total_decode_tokens / wall_s if wall_s > 0 else 0.0
            ttft_ms = statistics.fmean(per_request_ttft_ms) if per_request_ttft_ms else 0.0
            itl_ms = statistics.fmean(per_request_itl_ms) if per_request_itl_ms else 0.0

            run_payload = {
                "run_index": run_idx + 1,
                "request_count": len(outputs),
                "decode_tokens": total_decode_tokens,
                "wall_s": wall_s,
                "throughput_tps": throughput,
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
            }
            run_payloads.append(run_payload)
            throughput_values.append(throughput)
            ttft_values.append(ttft_ms)
            itl_values.append(itl_ms)

        aggregate = {
            "throughput_tps": summarize_runs(throughput_values),
            "ttft_ms": summarize_runs(ttft_values),
            "itl_ms": summarize_runs(itl_values),
        }
        workload_reports.append(
            {
                "name": workload["name"],
                "description": workload["description"],
                "prompts": deepcopy(workload["prompts"]),
                "formatted_prompt_style": "chatml_serving_qwen",
                "config": {
                    "max_new_tokens": workload["max_new_tokens"],
                },
                "runs": run_payloads,
                "aggregate": aggregate,
            }
        )

    raw_path = output_dir / "vllm-baseline.json"
    md_path = output_dir / "vllm-baseline.md"
    dump_json(
        raw_path,
        {
            "context": context.__dict__,
            "workloads": workload_reports,
        },
    )
    md_path.write_text(render_markdown(context, workload_reports), encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
