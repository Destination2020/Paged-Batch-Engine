#!/usr/bin/env python3
"""Render README-ready benchmark tables from existing benchmark JSON files."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def fmt(value: Any, precision: int = 2, missing: str = "-") -> str:
    if value is None:
        return missing
    if isinstance(value, str):
        if not value:
            return missing
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def aggregate_mean(workload: dict[str, Any], metric: str) -> float | None:
    aggregate = workload.get("aggregate", {})
    value = aggregate.get(metric)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    return float(value)


def first_run_summary(workload: dict[str, Any]) -> dict[str, Any]:
    runs = workload.get("runs", [])
    if not runs:
        return {}
    first = runs[0]
    summary = first.get("summary")
    return summary if isinstance(summary, dict) else first


def first_config(workload: dict[str, Any]) -> dict[str, Any]:
    config = workload.get("config", {})
    return config if isinstance(config, dict) else {}


def prompt_shape(workload: dict[str, Any]) -> str:
    config = first_config(workload)
    p50 = config.get("prompt_p50_tokens")
    p95 = config.get("prompt_p95_tokens")
    max_tokens = config.get("prompt_max_tokens")
    if p50 is None and p95 is None and max_tokens is None:
        return "-"
    return f"{fmt(p50, 0)}/{fmt(p95, 0)}/{fmt(max_tokens, 0)}"


def success_rate(workload: dict[str, Any]) -> float | None:
    summary = first_run_summary(workload)
    completed = summary.get("completed_requests")
    failed = summary.get("failed_requests")
    if completed is None:
        request_count = summary.get("request_count")
        if request_count is None:
            return None
        completed = request_count
        failed = 0
    completed = int(float(completed))
    failed = int(float(failed or 0))
    total = completed + failed
    if total <= 0:
        return None
    return 100.0 * completed / total


def decode_tokens(workload: dict[str, Any]) -> int | None:
    summary = first_run_summary(workload)
    value = summary.get("total_decode_tokens", summary.get("decode_tokens"))
    if value is None:
        return None
    return int(float(value))


def request_count(workload: dict[str, Any]) -> int:
    config = first_config(workload)
    value = config.get("prompt_count")
    if value is not None:
        return int(float(value))
    prompts = workload.get("prompts", [])
    return len(prompts) if isinstance(prompts, list) else 0


def workload_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in payload.get("workloads", [])
        if isinstance(item, dict) and "name" in item
    }


def context_line(name: str, payload: dict[str, Any]) -> list[str]:
    context = payload.get("context", {})
    if not isinstance(context, dict):
        context = {}
    return [
        f"- {name} model: `{context.get('model_path', '-')}`",
        f"- {name} tokenizer: `{context.get('tokenizer_path', '-')}`",
        f"- {name} GPU: `{context.get('gpu_name', '-')}` ({context.get('gpu_memory_mb', '-')} MB)",
        f"- {name} commit: `{context.get('git_commit', '-')}`",
    ]


def benchmark_row(engine: str, workload: dict[str, Any]) -> str:
    config = first_config(workload)
    max_new_tokens = config.get("max_new_tokens")
    return (
        f"| {engine} | {workload.get('name', '-')} | {request_count(workload)} | "
        f"{prompt_shape(workload)} | {fmt(max_new_tokens, 0)} | "
        f"{fmt(aggregate_mean(workload, 'throughput_tps'))} | "
        f"{fmt(aggregate_mean(workload, 'ttft_p95_ms'))} | "
        f"{fmt(aggregate_mean(workload, 'itl_p95_ms'))} | "
        f"{fmt(aggregate_mean(workload, 'latency_p99_ms'))} | "
        f"{fmt(success_rate(workload), 1)}% |"
    )


def profiling_row(workload: dict[str, Any]) -> str:
    summary = first_run_summary(workload)
    active_steps = float(summary.get("active_steps", 0.0) or 0.0)
    decode_only_steps = float(summary.get("decode_only_steps", 0.0) or 0.0)
    decode_only_ratio = aggregate_mean(workload, "decode_only_ratio")
    if decode_only_ratio is None and active_steps > 0.0:
        decode_only_ratio = decode_only_steps / active_steps
    return (
        f"| {workload.get('name', '-')} | "
        f"{fmt(aggregate_mean(workload, 'avg_schedule_ms'), 3)} | "
        f"{fmt(aggregate_mean(workload, 'avg_build_metadata_ms'), 3)} | "
        f"{fmt(aggregate_mean(workload, 'avg_forward_ms'), 3)} | "
        f"{fmt(aggregate_mean(workload, 'avg_sample_ms'), 3)} | "
        f"{fmt(aggregate_mean(workload, 'avg_process_outputs_ms'), 3)} | "
        f"{fmt(decode_only_ratio, 3)} | "
        f"{fmt(summary.get('avg_waiting_queue'), 2)} | "
        f"{fmt(summary.get('max_waiting_queue'), 0)} | "
        f"{fmt(summary.get('avg_running_queue'), 2)} | "
        f"{fmt(summary.get('max_running_queue'), 0)} | "
        f"{fmt(summary.get('scheduler_decode_kv_preemptions'), 0)} | "
        f"{fmt(decode_tokens(workload), 0)} |"
    )


def render_tables(paged: dict[str, Any],
                  vllm: dict[str, Any] | None,
                  workloads: list[str]) -> str:
    paged_workloads = workload_map(paged)
    vllm_workloads = workload_map(vllm) if vllm else {}
    if not workloads:
        workloads = sorted(set(paged_workloads) | set(vllm_workloads))

    lines = [
        "# README Benchmark Tables",
        "",
        "## Benchmark Environment",
        "",
        *context_line("PagedBatchEngine", paged),
    ]
    if vllm:
        lines.extend(context_line("vLLM", vllm))
    lines.extend(
        [
            "",
            "## Offline Benchmark Comparison",
            "",
            "Prompt tokens are shown as `p50/p95/max`. Throughput is output tokens per second.",
            "",
            "| Engine | Workload | Requests | Prompt tokens p50/p95/max | Max new tokens | Output tok/s | TTFT p95 ms | ITL p95 ms | E2E p99 ms | Success |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in workloads:
        if name in paged_workloads:
            lines.append(benchmark_row("PagedBatchEngine", paged_workloads[name]))
        if name in vllm_workloads:
            lines.append(benchmark_row("vLLM", vllm_workloads[name]))

    lines.extend(
        [
            "",
            "## PagedBatchEngine Internal Profiling",
            "",
            "| Workload | Schedule ms | Metadata ms | Forward ms | Sample ms | Process ms | Decode-only ratio | Avg waiting | Max waiting | Avg running | Max running | KV preemptions | Decode tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in workloads:
        if name in paged_workloads:
            lines.append(profiling_row(paged_workloads[name]))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paged-json", default="docs/benchmarks/pagedbench-baseline.json")
    parser.add_argument("--vllm-json", default="docs/benchmarks/vllm-baseline.json")
    parser.add_argument("--output", default="docs/benchmarks/readme-tables.md")
    parser.add_argument("--workloads", default="",
                        help="Comma-separated workload names. Default uses all workloads.")
    parser.add_argument("--allow-missing-vllm", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    paged_path = (repo_root / args.paged_json).resolve()
    vllm_path = (repo_root / args.vllm_json).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paged = load_json(paged_path)
    vllm = None
    if vllm_path.exists():
        vllm = load_json(vllm_path)
    elif not args.allow_missing_vllm:
        raise RuntimeError(f"vLLM JSON not found: {vllm_path}")

    workloads = [name.strip() for name in args.workloads.split(",") if name.strip()]
    output_path.write_text(render_tables(paged, vllm, workloads), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
