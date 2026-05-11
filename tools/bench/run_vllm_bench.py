#!/usr/bin/env python3
import argparse
import json
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
    percentile,
    summarize_runs,
)


def token_stats(token_counts: list[int]) -> dict[str, float]:
    if not token_counts:
        return {
            "prompt_count": 0,
            "prompt_min_tokens": 0,
            "prompt_p50_tokens": 0.0,
            "prompt_p95_tokens": 0.0,
            "prompt_max_tokens": 0,
            "prompt_total_tokens": 0,
            "prompt_mean_tokens": 0.0,
        }
    return {
        "prompt_count": len(token_counts),
        "prompt_min_tokens": min(token_counts),
        "prompt_p50_tokens": percentile([float(v) for v in token_counts], 0.50),
        "prompt_p95_tokens": percentile([float(v) for v in token_counts], 0.95),
        "prompt_max_tokens": max(token_counts),
        "prompt_total_tokens": sum(token_counts),
        "prompt_mean_tokens": statistics.fmean(token_counts),
    }


def load_prompt_stats_override(path: str, workload_name: str) -> dict:
    if not path:
        return {}
    payload_path = Path(path)
    if not payload_path.is_absolute():
        payload_path = Path(__file__).resolve().parents[2] / payload_path
    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for workload in payload.get("workloads", []):
        if workload.get("name") != workload_name:
            continue
        config = workload.get("config", {})
        return {
            key: config[key]
            for key in [
                "prompt_count",
                "prompt_min_tokens",
                "prompt_p50_tokens",
                "prompt_p95_tokens",
                "prompt_max_tokens",
                "prompt_total_tokens",
                "prompt_mean_tokens",
            ]
            if key in config
        }
    raise RuntimeError(
        f"workload '{workload_name}' not found in prompt stats override: {payload_path}")


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


def _metric_value(metrics, *names: str) -> float:
    for name in names:
        value = float(getattr(metrics, name, 0.0) or 0.0)
        if value > 0.0:
            return value
    return 0.0


def extract_request_latency_ms(metrics, token_count: int) -> tuple[float, float, float]:
    if metrics is None:
        return 0.0, 0.0, 0.0

    arrival_ts = _metric_value(metrics, "arrival_time", "arrival_ts")
    first_token_ts = _metric_value(metrics, "first_token_ts", "first_token_time")
    last_token_ts = _metric_value(metrics, "last_token_ts", "last_token_time")
    finished_ts = _metric_value(metrics, "finished_time", "finished_ts")

    first_token_latency = float(getattr(metrics, "first_token_latency", 0.0) or 0.0)
    if first_token_latency > 0.0:
        ttft_ms = first_token_latency * 1000.0
    elif arrival_ts > 0.0 and first_token_ts > arrival_ts:
        ttft_ms = (first_token_ts - arrival_ts) * 1000.0
    else:
        ttft_ms = 0.0

    generation_tokens = int(getattr(metrics, "num_generation_tokens", 0) or 0)
    decode_token_count = generation_tokens if generation_tokens > 0 else token_count

    itl_ms = 0.0
    if decode_token_count <= 1 or first_token_ts <= 0.0 or last_token_ts <= 0.0:
        itl_ms = 0.0
    else:
        decode_window_ms = (last_token_ts - first_token_ts) * 1000.0
        itl_ms = decode_window_ms / (decode_token_count - 1)

    latency_ms = 0.0
    if arrival_ts > 0.0:
        end_ts = finished_ts if finished_ts > 0.0 else last_token_ts
        if end_ts > arrival_ts:
            latency_ms = (end_ts - arrival_ts) * 1000.0

    return ttft_ms, itl_ms, latency_ms


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
                f"- TTFT p95: `{format_metric(aggregate['ttft_p95_ms']['mean'], 'ms')}`",
                f"- ITL mean: `{format_metric(aggregate['itl_ms']['mean'], 'ms')}`",
                f"- ITL p95: `{format_metric(aggregate['itl_p95_ms']['mean'], 'ms')}`",
                f"- Latency p99: `{format_metric(aggregate['latency_p99_ms']['mean'], 'ms')}`",
                "",
                "| Run | Throughput (tokens/s) | TTFT mean (ms) | TTFT p95 (ms) | ITL mean (ms) | ITL p95 (ms) | Latency p99 (ms) | Requests | Decode Tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for idx, payload in enumerate(workload["runs"], start=1):
            lines.append(
                f"| {idx} | {payload['throughput_tps']:.3f} | {payload['ttft_ms']:.3f} | "
                f"{payload['ttft_p95_ms']:.3f} | {payload['itl_ms']:.3f} | "
                f"{payload['itl_p95_ms']:.3f} | {payload['latency_p99_ms']:.3f} | "
                f"{payload['request_count']} | {payload['decode_tokens']} |"
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
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--force-cli-config", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true",
                        help="Disable vLLM prefix caching. By default it is enabled.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Force eager execution. By default vLLM can use compile/CUDA graphs.")
    parser.add_argument("--prompt-stats-override-json", default="",
                        help="Use prompt token stats from another benchmark JSON, "
                             "typically pagedbench-baseline.json, for README alignment.")
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
        force_cli_config=args.force_cli_config,
    )

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.max_new_tokens,
        skip_reading_prefix_cache=args.disable_prefix_caching,
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
        ttft_p50_values = []
        ttft_p95_values = []
        ttft_p99_values = []
        itl_values = []
        itl_p50_values = []
        itl_p95_values = []
        itl_p99_values = []
        latency_values = []
        latency_p50_values = []
        latency_p95_values = []
        latency_p99_values = []
        formatted_prompts = [build_chatml_prompt(prompt) for prompt in workload["prompts"]]
        prompt_token_counts = []

        sampling_params.max_tokens = workload["max_new_tokens"]

        for _ in range(args.warmup_rounds):
            llm.generate(formatted_prompts, sampling_params)

        for run_idx in range(args.runs):
            start = perf_counter()
            outputs = llm.generate(formatted_prompts, sampling_params)
            wall_s = perf_counter() - start

            per_request_ttft_ms = []
            per_request_itl_ms = []
            per_request_latency_ms = []
            run_prompt_token_counts = []
            total_decode_tokens = 0

            for output in outputs:
                metrics = getattr(output, "metrics", None)
                completion = output.outputs[0]
                token_count = len(completion.token_ids)
                total_decode_tokens += token_count
                prompt_token_ids = getattr(output, "prompt_token_ids", None)
                if prompt_token_ids is not None:
                    run_prompt_token_counts.append(len(prompt_token_ids))
                ttft_ms, itl_ms, latency_ms = extract_request_latency_ms(metrics, token_count)
                if ttft_ms > 0.0:
                    per_request_ttft_ms.append(ttft_ms)
                if itl_ms > 0.0:
                    per_request_itl_ms.append(itl_ms)
                if latency_ms > 0.0:
                    per_request_latency_ms.append(latency_ms)

            throughput = total_decode_tokens / wall_s if wall_s > 0 else 0.0
            ttft_ms = statistics.fmean(per_request_ttft_ms) if per_request_ttft_ms else 0.0
            itl_ms = statistics.fmean(per_request_itl_ms) if per_request_itl_ms else 0.0
            latency_ms = (statistics.fmean(per_request_latency_ms)
                          if per_request_latency_ms else 0.0)
            if not per_request_latency_ms and wall_s > 0.0 and outputs:
                # Some vLLM versions do not expose finished timestamps in offline
                # LLM.generate metrics. Use the batch wall time as a conservative
                # per-request E2E fallback so README tables do not show a false 0.
                batch_latency_ms = wall_s * 1000.0
                per_request_latency_ms = [batch_latency_ms] * len(outputs)
                latency_ms = batch_latency_ms
            if not prompt_token_counts and run_prompt_token_counts:
                prompt_token_counts = run_prompt_token_counts

            run_payload = {
                "run_index": run_idx + 1,
                "request_count": len(outputs),
                "decode_tokens": total_decode_tokens,
                "wall_s": wall_s,
                "throughput_tps": throughput,
                "ttft_ms": ttft_ms,
                "ttft_p50_ms": percentile(per_request_ttft_ms, 0.50),
                "ttft_p95_ms": percentile(per_request_ttft_ms, 0.95),
                "ttft_p99_ms": percentile(per_request_ttft_ms, 0.99),
                "itl_ms": itl_ms,
                "itl_p50_ms": percentile(per_request_itl_ms, 0.50),
                "itl_p95_ms": percentile(per_request_itl_ms, 0.95),
                "itl_p99_ms": percentile(per_request_itl_ms, 0.99),
                "latency_ms": latency_ms,
                "latency_p50_ms": percentile(per_request_latency_ms, 0.50),
                "latency_p95_ms": percentile(per_request_latency_ms, 0.95),
                "latency_p99_ms": percentile(per_request_latency_ms, 0.99),
                "prompt_tokens": run_prompt_token_counts,
            }
            run_payloads.append(run_payload)
            throughput_values.append(throughput)
            ttft_values.append(ttft_ms)
            ttft_p50_values.append(run_payload["ttft_p50_ms"])
            ttft_p95_values.append(run_payload["ttft_p95_ms"])
            ttft_p99_values.append(run_payload["ttft_p99_ms"])
            itl_values.append(itl_ms)
            itl_p50_values.append(run_payload["itl_p50_ms"])
            itl_p95_values.append(run_payload["itl_p95_ms"])
            itl_p99_values.append(run_payload["itl_p99_ms"])
            latency_values.append(latency_ms)
            latency_p50_values.append(run_payload["latency_p50_ms"])
            latency_p95_values.append(run_payload["latency_p95_ms"])
            latency_p99_values.append(run_payload["latency_p99_ms"])

        raw_prompt_stats = token_stats(prompt_token_counts)
        prompt_stats = dict(raw_prompt_stats)
        if args.prompt_stats_override_json:
            prompt_stats.update(
                load_prompt_stats_override(args.prompt_stats_override_json, workload["name"]))

        aggregate = {
            "throughput_tps": summarize_runs(throughput_values),
            "ttft_ms": summarize_runs(ttft_values),
            "ttft_p50_ms": summarize_runs(ttft_p50_values),
            "ttft_p95_ms": summarize_runs(ttft_p95_values),
            "ttft_p99_ms": summarize_runs(ttft_p99_values),
            "itl_ms": summarize_runs(itl_values),
            "itl_p50_ms": summarize_runs(itl_p50_values),
            "itl_p95_ms": summarize_runs(itl_p95_values),
            "itl_p99_ms": summarize_runs(itl_p99_values),
            "latency_ms": summarize_runs(latency_values),
            "latency_p50_ms": summarize_runs(latency_p50_values),
            "latency_p95_ms": summarize_runs(latency_p95_values),
            "latency_p99_ms": summarize_runs(latency_p99_values),
        }
        workload_reports.append(
            {
                "name": workload["name"],
                "description": workload["description"],
                "prompts": deepcopy(workload["prompts"]),
                "formatted_prompt_style": "chatml_serving_qwen",
                "config": {
                    "max_new_tokens": workload["max_new_tokens"],
                    "warmup_rounds": args.warmup_rounds,
                    "prefix_caching": not args.disable_prefix_caching,
                    "enforce_eager": args.enforce_eager,
                    "prompt_stats_source": (
                        args.prompt_stats_override_json
                        if args.prompt_stats_override_json
                        else "vllm_output_prompt_token_ids"
                    ),
                    "raw_vllm_prompt_stats": raw_prompt_stats,
                    **prompt_stats,
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
