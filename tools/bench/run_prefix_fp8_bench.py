#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from report_utils import (
    build_report_context,
    dump_json,
    format_metric,
    load_workloads,
    parse_workload_names,
    summarize_runs,
)


SUMMARY_RE = re.compile(r"(\w+)=([^\s]+)")
CONFIG_VALUE_KEYS = [
    "max_batched_tokens_request",
    "max_batched_tokens_resolved",
    "prefill_chunk_cap_request",
    "prefill_chunk_cap_resolved",
    "gpu_memory_utilization",
    "warmup_rounds",
    "prompt_count",
    "prompt_min_tokens",
    "prompt_p50_tokens",
    "prompt_p95_tokens",
    "prompt_max_tokens",
    "prompt_total_tokens",
    "prompt_mean_tokens",
    "capacity_max_batch_size",
    "capacity_block_size",
    "capacity_total_kv_blocks",
    "capacity_free_kv_blocks",
    "capacity_layer_num",
    "capacity_model_dim",
    "capacity_head_num",
    "capacity_kv_head_num",
    "capacity_head_size",
    "capacity_kv_dim",
    "capacity_hidden_dim",
    "capacity_vocab_size",
    "capacity_gpu_free_memory_bytes",
    "capacity_gpu_total_memory_bytes",
    "capacity_kv_bytes_per_token",
    "capacity_workspace_bytes_per_token",
    "auto_usable_free_kv_blocks",
    "auto_kv_token_budget",
    "auto_gpu_workspace_bytes_budget",
    "auto_workspace_token_budget",
    "auto_capacity_token_budget",
    "auto_scheduler_seq_window",
    "auto_target_decode_concurrency",
    "auto_prefill_reserved_seqs",
    "auto_prompt_prefill_chunk_target",
    "auto_scheduler_token_budget",
    "auto_safety_token_cap",
    "auto_prefill_safety_token_cap",
]

VARIANTS = [
    {
        "name": "baseline_bf16",
        "description": "BF16 KV cache with Prefix Cache disabled.",
        "env": {
            "KUIPER_ENABLE_PREFIX_CACHE": "0",
        },
    },
    {
        "name": "prefix_cache_bf16",
        "description": "BF16 KV cache with warm Prefix Cache reuse enabled.",
        "env": {
            "KUIPER_ENABLE_PREFIX_CACHE": "1",
        },
    },
    {
        "name": "fp8_kv",
        "description": "FP8 KV cache with Prefix Cache disabled.",
        "env": {
            "KUIPER_ENABLE_PREFIX_CACHE": "0",
            "KUIPER_USE_FP8_KV_CACHE": "1",
        },
    },
    {
        "name": "prefix_cache_fp8_kv",
        "description": "FP8 KV cache with warm Prefix Cache reuse enabled.",
        "env": {
            "KUIPER_ENABLE_PREFIX_CACHE": "1",
            "KUIPER_USE_FP8_KV_CACHE": "1",
        },
    },
]


def parse_summary(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("FINAL_SUMMARY "):
            return dict(SUMMARY_RE.findall(line))
    raise RuntimeError("FINAL_SUMMARY line not found in serving_qwen output")


def parse_config_summary(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("CONFIG_SUMMARY "):
            return dict(SUMMARY_RE.findall(line))
    raise RuntimeError("CONFIG_SUMMARY line not found in serving_qwen output")


def build_command(
    binary: str,
    model: str,
    tokenizer: str,
    prompts: list[str],
    max_new_tokens: int,
    max_batched_tokens: int | str,
    prefill_chunk_cap: int | str,
    warmup_rounds: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        binary,
        model,
        tokenizer,
        *prompts,
        f"--max-new-tokens={max_new_tokens}",
        f"--max-batched-tokens={max_batched_tokens}",
        f"--prefill-chunk-cap={prefill_chunk_cap}",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--warmup-rounds={warmup_rounds}",
        "--quiet=1",
        "--step-profile=0",
        "--final-summary=1",
    ]


def to_float(payload: dict[str, str], key: str) -> float:
    value = payload.get(key, "0")
    try:
        return float(value)
    except ValueError:
        return 0.0


def percent_delta(current: float, baseline: float) -> str:
    if baseline == 0.0:
        return "n/a"
    return f"{((current - baseline) / baseline) * 100.0:+.1f}%"


def mib_from_tokens(tokens: float, kv_bytes_per_token: float) -> float:
    if tokens <= 0.0 or kv_bytes_per_token <= 0.0:
        return 0.0
    return tokens * kv_bytes_per_token / (1024.0 * 1024.0)


def render_markdown(context, variants: list[dict], workload_reports: list[dict]) -> str:
    lines = [
        "# Prefix Cache / FP8 KV Benchmark Report",
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
        f"- GPU memory utilization: `{context.gpu_memory_utilization}`",
        f"- Warmup rounds per run: `{context.warmup_rounds}`",
        "",
        "## Method",
        "",
        "- Every variant uses the same prompt set, scheduler caps, and warmup count within a workload.",
        "- Prefix Cache variants rely on the warmup round to seed cache entries before the measured round.",
        "- `prefix_cache_stats` are reset after warmup, so reported hits and reused tokens cover only measured runs.",
        "- FP8 KV memory benefit is reported from `capacity_kv_bytes_per_token` and `auto_kv_token_budget` in `CONFIG_SUMMARY`.",
        "",
        "## Variants",
        "",
    ]

    for variant in variants:
        env_text = ", ".join(f"{key}={value}" for key, value in variant["env"].items())
        lines.append(f"- `{variant['name']}`: `{variant['description']}` Env=`{env_text}`")

    lines.append("")

    for workload in workload_reports:
        reports = workload["variant_reports"]
        report_by_name = {report["name"]: report for report in reports}
        baseline = report_by_name["baseline_bf16"]
        lines.extend(
            [
                f"## Workload `{workload['name']}`",
                "",
                f"- Description: `{workload['description']}`",
                f"- Prompt count: `{len(workload['prompts'])}`",
                f"- Max new tokens: `{workload['config']['max_new_tokens']}`",
                f"- Max batched tokens: `{workload['config']['max_batched_tokens_request']}` -> `{workload['config']['max_batched_tokens_resolved']}`",
                f"- Prefill chunk cap: `{workload['config']['prefill_chunk_cap_request']}` -> `{workload['config']['prefill_chunk_cap_resolved']}`",
                f"- Prompt token p50/p95/max: `{workload['config'].get('prompt_p50_tokens', 'n/a')}` / `{workload['config'].get('prompt_p95_tokens', 'n/a')}` / `{workload['config'].get('prompt_max_tokens', 'n/a')}`",
                "",
                "### Performance",
                "",
                "| Variant | Throughput (tokens/s) | TTFT mean ms | ITL mean ms | Latency p95 ms | Avg forward ms | Avg sample ms | TTFT delta vs baseline | ITL delta vs baseline |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for report in reports:
            aggregate = report["aggregate"]
            lines.append(
                f"| {report['name']} | "
                f"{aggregate['throughput_tps']['mean']:.3f} | "
                f"{aggregate['ttft_ms']['mean']:.3f} | "
                f"{aggregate['itl_ms']['mean']:.3f} | "
                f"{aggregate['latency_p95_ms']['mean']:.3f} | "
                f"{aggregate['avg_forward_ms']['mean']:.3f} | "
                f"{aggregate['avg_sample_ms']['mean']:.3f} | "
                f"{percent_delta(aggregate['ttft_ms']['mean'], baseline['aggregate']['ttft_ms']['mean'])} | "
                f"{percent_delta(aggregate['itl_ms']['mean'], baseline['aggregate']['itl_ms']['mean'])} |"
            )
        lines.extend(
            [
                "",
                "### Memory / Reuse",
                "",
                "| Variant | KV bytes/token | Auto KV token budget | Total KV blocks | Prefix hits | Reused tokens | Reused KV MiB eq | KV bytes/token delta vs baseline | KV token budget delta vs baseline |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for report in reports:
            aggregate = report["aggregate"]
            lines.append(
                f"| {report['name']} | "
                f"{aggregate['capacity_kv_bytes_per_token']['mean']:.0f} | "
                f"{aggregate['auto_kv_token_budget']['mean']:.0f} | "
                f"{aggregate['capacity_total_kv_blocks']['mean']:.0f} | "
                f"{aggregate['prefix_cache_hits']['mean']:.1f} | "
                f"{aggregate['prefix_cache_tokens_reused']['mean']:.1f} | "
                f"{aggregate['reused_kv_mib_equivalent']['mean']:.3f} | "
                f"{percent_delta(aggregate['capacity_kv_bytes_per_token']['mean'], baseline['aggregate']['capacity_kv_bytes_per_token']['mean'])} | "
                f"{percent_delta(aggregate['auto_kv_token_budget']['mean'], baseline['aggregate']['auto_kv_token_budget']['mean'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    default_output_dir = (
        f"docs/benchmarks/compare-{datetime.now().date().isoformat()}-prefix-fp8"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/demo/serving_qwen")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", default="tools/bench/prefix_fp8_workloads.json")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output-dir", default=default_output_dir)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_workloads = parse_workload_names(args.workloads)
    workloads = load_workloads(
        (repo_root / args.prompts).resolve(),
        selected_names=selected_workloads,
        default_name="default",
        default_description="Legacy prompt-list workload.",
        max_new_tokens=64,
        max_batched_tokens="auto",
        prefill_chunk_cap="auto",
        force_cli_config=False,
    )

    selected_variants = parse_workload_names(args.variants)
    variants = [
        deepcopy(variant)
        for variant in VARIANTS
        if selected_variants is None or variant["name"] in selected_variants
    ]
    if not variants:
        raise RuntimeError("no benchmark variants selected")

    context = build_report_context(
        repo_root=repo_root,
        output_dir=output_dir,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        prompts_path=args.prompts,
        max_new_tokens=max(workload["max_new_tokens"] for workload in workloads),
        runs=args.runs,
    )
    context.warmup_rounds = args.warmup_rounds
    context.gpu_memory_utilization = args.gpu_memory_utilization

    workload_reports = []
    for workload in workloads:
        variant_reports = []
        for variant in variants:
            run_payloads = []
            throughput_values = []
            ttft_values = []
            itl_values = []
            latency_p95_values = []
            avg_forward_values = []
            avg_sample_values = []
            prefix_hit_values = []
            prefix_reused_token_values = []
            kv_bytes_per_token_values = []
            kv_token_budget_values = []
            total_kv_block_values = []
            reused_kv_mib_values = []

            for run_idx in range(args.runs):
                cmd = build_command(
                    args.binary,
                    args.model,
                    args.tokenizer,
                    workload["prompts"],
                    workload["max_new_tokens"],
                    workload["max_batched_tokens"],
                    workload["prefill_chunk_cap"],
                    args.warmup_rounds,
                    args.gpu_memory_utilization,
                )
                env = os.environ.copy()
                env.update(variant["env"])
                env["GLOG_logtostderr"] = "1"
                proc = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                    env=env,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"serving_qwen failed on workload {workload['name']} "
                        f"variant {variant['name']} run {run_idx + 1}\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )

                summary = parse_summary(proc.stdout)
                config_summary = parse_config_summary(proc.stdout)
                prefix_reused_tokens = to_float(summary, "prefix_cache_tokens_reused")
                kv_bytes_per_token = to_float(config_summary, "capacity_kv_bytes_per_token")
                reused_kv_mib = mib_from_tokens(prefix_reused_tokens, kv_bytes_per_token)
                run_payloads.append(
                    {
                        "run_index": run_idx + 1,
                        "command": cmd,
                        "env": deepcopy(variant["env"]),
                        "summary": summary,
                        "config_summary": config_summary,
                        "reused_kv_mib_equivalent": reused_kv_mib,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    }
                )

                throughput_values.append(to_float(summary, "throughput_tps"))
                ttft_values.append(to_float(summary, "ttft_ms"))
                itl_values.append(to_float(summary, "itl_ms"))
                latency_p95_values.append(to_float(summary, "latency_p95_ms"))
                avg_forward_values.append(to_float(summary, "avg_forward_ms"))
                avg_sample_values.append(to_float(summary, "avg_sample_ms"))
                prefix_hit_values.append(to_float(summary, "prefix_cache_hits"))
                prefix_reused_token_values.append(prefix_reused_tokens)
                kv_bytes_per_token_values.append(kv_bytes_per_token)
                kv_token_budget_values.append(to_float(config_summary, "auto_kv_token_budget"))
                total_kv_block_values.append(to_float(config_summary, "capacity_total_kv_blocks"))
                reused_kv_mib_values.append(reused_kv_mib)

            first_config_summary = run_payloads[0]["config_summary"] if run_payloads else {}
            config_details = {
                key: first_config_summary[key]
                for key in CONFIG_VALUE_KEYS
                if key in first_config_summary
            }
            variant_reports.append(
                {
                    "name": variant["name"],
                    "description": variant["description"],
                    "env": deepcopy(variant["env"]),
                    "config": config_details,
                    "runs": run_payloads,
                    "aggregate": {
                        "throughput_tps": summarize_runs(throughput_values),
                        "ttft_ms": summarize_runs(ttft_values),
                        "itl_ms": summarize_runs(itl_values),
                        "latency_p95_ms": summarize_runs(latency_p95_values),
                        "avg_forward_ms": summarize_runs(avg_forward_values),
                        "avg_sample_ms": summarize_runs(avg_sample_values),
                        "prefix_cache_hits": summarize_runs(prefix_hit_values),
                        "prefix_cache_tokens_reused": summarize_runs(prefix_reused_token_values),
                        "capacity_kv_bytes_per_token": summarize_runs(kv_bytes_per_token_values),
                        "auto_kv_token_budget": summarize_runs(kv_token_budget_values),
                        "capacity_total_kv_blocks": summarize_runs(total_kv_block_values),
                        "reused_kv_mib_equivalent": summarize_runs(reused_kv_mib_values),
                    },
                }
            )

        first_variant_config = variant_reports[0]["config"] if variant_reports else {}
        workload_reports.append(
            {
                "name": workload["name"],
                "description": workload["description"],
                "prompts": deepcopy(workload["prompts"]),
                "config": {
                    "max_new_tokens": workload["max_new_tokens"],
                    "max_batched_tokens_request": first_variant_config.get(
                        "max_batched_tokens_request", str(workload["max_batched_tokens"])
                    ),
                    "max_batched_tokens_resolved": first_variant_config.get(
                        "max_batched_tokens_resolved", str(workload["max_batched_tokens"])
                    ),
                    "prefill_chunk_cap_request": first_variant_config.get(
                        "prefill_chunk_cap_request", str(workload["prefill_chunk_cap"])
                    ),
                    "prefill_chunk_cap_resolved": first_variant_config.get(
                        "prefill_chunk_cap_resolved", str(workload["prefill_chunk_cap"])
                    ),
                    **{
                        key: first_variant_config[key]
                        for key in (
                            "prompt_count",
                            "prompt_min_tokens",
                            "prompt_p50_tokens",
                            "prompt_p95_tokens",
                            "prompt_max_tokens",
                            "prompt_total_tokens",
                            "prompt_mean_tokens",
                        )
                        if key in first_variant_config
                    },
                },
                "variant_reports": variant_reports,
            }
        )

    raw_path = output_dir / "prefix-fp8-bench.json"
    md_path = output_dir / "prefix-fp8-bench.md"
    dump_json(
        raw_path,
        {
            "context": context.__dict__,
            "variants": variants,
            "workloads": workload_reports,
        },
    )
    md_path.write_text(render_markdown(context, variants, workload_reports), encoding="utf-8")

    print(f"Wrote {raw_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
