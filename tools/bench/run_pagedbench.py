#!/usr/bin/env python3
import argparse
import re
import subprocess
from copy import deepcopy
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
CONFIG_DETAIL_KEYS = [
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


def parse_summary(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("FINAL_SUMMARY "):
            return dict(SUMMARY_RE.findall(line))
    raise RuntimeError("FINAL_SUMMARY line not found in serving_qwen output")


def parse_config_summary(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("CONFIG_SUMMARY "):
            return dict(SUMMARY_RE.findall(line))
    return {}


def parse_step_profiles(stdout: str) -> list[dict[str, str]]:
    profiles = []
    for line in stdout.splitlines():
        if line.startswith("STEP_PROFILE "):
            profiles.append(dict(SUMMARY_RE.findall(line)))
    return profiles


def parse_request_metrics(stdout: str) -> list[dict[str, str]]:
    metrics = []
    for line in stdout.splitlines():
        if line.startswith("REQUEST_METRIC "):
            metrics.append(dict(SUMMARY_RE.findall(line)))
    return metrics


def request_metric_values(request_metrics: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for metric in request_metrics:
        value = float(metric.get(key, "0") or 0.0)
        if value > 0.0:
            values.append(value)
    return values


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
        "--step-profile=1",
        "--final-summary=1",
    ]
def render_markdown(context, workload_reports: list[dict]) -> str:
    lines = [
        "# PagedBatchEngine Baseline Report",
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
              f"- Max batched tokens: `{workload['config']['max_batched_tokens_request']}` -> `{workload['config']['max_batched_tokens_resolved']}`",
              f"- Prefill chunk cap: `{workload['config']['prefill_chunk_cap_request']}` -> `{workload['config']['prefill_chunk_cap_resolved']}`",
              f"- GPU memory utilization: `{workload['config']['gpu_memory_utilization']}`",
              f"- Warmup rounds: `{workload['config']['warmup_rounds']}`",
              f"- Prompt tokens: count=`{workload['config'].get('prompt_count', 'n/a')}`, p50=`{workload['config'].get('prompt_p50_tokens', 'n/a')}`, p95=`{workload['config'].get('prompt_p95_tokens', 'n/a')}`, max=`{workload['config'].get('prompt_max_tokens', 'n/a')}`",
              f"- KV blocks free/total: `{workload['config'].get('capacity_free_kv_blocks', 'n/a')}` / `{workload['config'].get('capacity_total_kv_blocks', 'n/a')}`",
              f"- KV block size: `{workload['config'].get('capacity_block_size', 'n/a')}`",
              f"- Model capacity shape: layers=`{workload['config'].get('capacity_layer_num', 'n/a')}`, heads=`{workload['config'].get('capacity_head_num', 'n/a')}`, kv_heads=`{workload['config'].get('capacity_kv_head_num', 'n/a')}`, head_size=`{workload['config'].get('capacity_head_size', 'n/a')}`",
              f"- Bytes per token: KV=`{workload['config'].get('capacity_kv_bytes_per_token', 'n/a')}`, workspace=`{workload['config'].get('capacity_workspace_bytes_per_token', 'n/a')}`",
              f"- GPU memory free/total bytes: `{workload['config'].get('capacity_gpu_free_memory_bytes', 'n/a')}` / `{workload['config'].get('capacity_gpu_total_memory_bytes', 'n/a')}`",
              f"- Auto budgets: KV tokens=`{workload['config'].get('auto_kv_token_budget', 'n/a')}`, workspace tokens=`{workload['config'].get('auto_workspace_token_budget', 'n/a')}`, capacity tokens=`{workload['config'].get('auto_capacity_token_budget', 'n/a')}`",
              f"- Auto scheduler budget: seq_window=`{workload['config'].get('auto_scheduler_seq_window', 'n/a')}`, decode_target=`{workload['config'].get('auto_target_decode_concurrency', 'n/a')}`, prefill_reserved=`{workload['config'].get('auto_prefill_reserved_seqs', 'n/a')}`, prefill_target=`{workload['config'].get('auto_prompt_prefill_chunk_target', 'n/a')}`, scheduler_tokens=`{workload['config'].get('auto_scheduler_token_budget', 'n/a')}`",
              f"- Throughput mean: `{format_metric(aggregate['throughput_tps']['mean'], 'tokens/s')}`",
              f"- TTFT mean: `{format_metric(aggregate['ttft_ms']['mean'], 'ms')}`",
              f"- TTFT p95: `{format_metric(aggregate['ttft_p95_ms']['mean'], 'ms')}`",
              f"- ITL mean: `{format_metric(aggregate['itl_ms']['mean'], 'ms')}`",
              f"- ITL p95: `{format_metric(aggregate['itl_p95_ms']['mean'], 'ms')}`",
              f"- Latency mean: `{format_metric(aggregate['latency_ms']['mean'], 'ms')}`",
              f"- Latency p99: `{format_metric(aggregate['latency_p99_ms']['mean'], 'ms')}`",
              f"- Decode-only steps mean: `{format_metric(aggregate['decode_only_steps']['mean'])}`",
              f"- Decode-only ratio mean: `{format_metric(aggregate['decode_only_ratio']['mean'])}`",
              f"- Avg schedule ms: `{format_metric(aggregate['avg_schedule_ms']['mean'], 'ms')}`",
              f"- Avg build metadata ms: `{format_metric(aggregate['avg_build_metadata_ms']['mean'], 'ms')}`",
              f"- Avg forward ms: `{format_metric(aggregate['avg_forward_ms']['mean'], 'ms')}`",
              f"- Avg sample ms: `{format_metric(aggregate['avg_sample_ms']['mean'], 'ms')}`",
              f"- Avg process outputs ms: `{format_metric(aggregate['avg_process_outputs_ms']['mean'], 'ms')}`",
              "",
              "| Run | Throughput (tokens/s) | TTFT mean ms | ITL mean ms | Latency p99 ms | Active Steps | Decode-only Steps | Decode-only Ratio | Decode Tokens | Prefill Tokens | Avg Schedule ms | Avg Build Metadata ms | Avg Forward ms | Avg Sample ms | Avg Process Outputs ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
          ]
      )
      for idx, payload in enumerate(workload["runs"], start=1):
        s = payload["summary"]
        active_steps = max(1, int(s["active_steps"]))
        decode_only_ratio = float(s.get("decode_only_steps", 0)) / active_steps
        lines.append(
            f"| {idx} | {s['throughput_tps']} | {s.get('ttft_ms', '0')} | {s.get('itl_ms', '0')} | {s.get('latency_p99_ms', '0')} | "
            f"{s['active_steps']} | {s.get('decode_only_steps', '0')} | {decode_only_ratio:.3f} | {s['total_decode_tokens']} | "
            f"{s['total_prefill_tokens']} | {s['avg_schedule_ms']} | {s['avg_build_metadata_ms']} | "
            f"{s['avg_forward_ms']} | {s['avg_sample_ms']} | {s['avg_process_outputs_ms']} |"
        )
      lines.append("")

    lines.extend(
        [
            "## Profiling Coverage",
            "",
            "- `schedule`",
            "- `build metadata`",
            "- `forward`",
            "- `sample`",
            "- `process outputs`",
            "- per-request `TTFT / ITL / latency`",
            "- dedicated decode-only workload coverage",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/demo/serving_qwen")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", default="tools/bench/workloads.json")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-batched-tokens", default="auto")
    parser.add_argument("--prefill-chunk-cap", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--force-cli-config", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = load_workloads(
        (repo_root / args.prompts).resolve(),
        selected_names=parse_workload_names(args.workloads),
        default_name="default",
        default_description="Legacy prompt-list workload.",
        max_new_tokens=args.max_new_tokens,
        max_batched_tokens=args.max_batched_tokens,
        prefill_chunk_cap=args.prefill_chunk_cap,
        force_cli_config=args.force_cli_config,
    )

    context = build_report_context(
        repo_root=repo_root,
        output_dir=output_dir,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        prompts_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
        runs=args.runs,
    )
    context.warmup_rounds = args.warmup_rounds
    context.gpu_memory_utilization = args.gpu_memory_utilization

    workload_reports = []
    for workload in workloads:
      run_payloads = []
      throughput_values = []
      schedule_values = []
      build_values = []
      forward_values = []
      sample_values = []
      process_values = []
      decode_only_step_values = []
      decode_only_ratio_values = []
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
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
          raise RuntimeError(
              f"serving_qwen failed on workload {workload['name']} run {run_idx + 1}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
          )
        summary = parse_summary(proc.stdout)
        config_summary = parse_config_summary(proc.stdout)
        step_profiles = parse_step_profiles(proc.stdout)
        request_metrics = parse_request_metrics(proc.stdout)
        run_payloads.append(
            {
                "run_index": run_idx + 1,
                "command": cmd,
                "summary": summary,
                "config_summary": config_summary,
                "step_profiles": step_profiles,
                "request_metrics": request_metrics,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        active_steps = max(1, int(summary["active_steps"]))
        decode_only_steps = float(summary.get("decode_only_steps", 0))
        throughput_values.append(float(summary["throughput_tps"]))
        decode_only_step_values.append(decode_only_steps)
        decode_only_ratio_values.append(decode_only_steps / active_steps)
        schedule_values.append(float(summary["avg_schedule_ms"]))
        build_values.append(float(summary["avg_build_metadata_ms"]))
        forward_values.append(float(summary["avg_forward_ms"]))
        sample_values.append(float(summary["avg_sample_ms"]))
        process_values.append(float(summary["avg_process_outputs_ms"]))
        ttft_values.append(float(summary.get("ttft_ms", 0.0)))
        ttft_p50_values.append(float(summary.get("ttft_p50_ms", 0.0)))
        ttft_p95_values.append(float(summary.get("ttft_p95_ms", 0.0)))
        ttft_p99_values.append(float(summary.get("ttft_p99_ms", 0.0)))
        itl_values.append(float(summary.get("itl_ms", 0.0)))
        itl_p50_values.append(float(summary.get("itl_p50_ms", 0.0)))
        itl_p95_values.append(float(summary.get("itl_p95_ms", 0.0)))
        itl_p99_values.append(float(summary.get("itl_p99_ms", 0.0)))
        latency_values.append(float(summary.get("latency_ms", 0.0)))
        latency_p50_values.append(float(summary.get("latency_p50_ms", 0.0)))
        latency_p95_values.append(float(summary.get("latency_p95_ms", 0.0)))
        latency_p99_values.append(float(summary.get("latency_p99_ms", 0.0)))

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
          "decode_only_steps": summarize_runs(decode_only_step_values),
          "decode_only_ratio": summarize_runs(decode_only_ratio_values),
          "avg_schedule_ms": summarize_runs(schedule_values),
          "avg_build_metadata_ms": summarize_runs(build_values),
          "avg_forward_ms": summarize_runs(forward_values),
          "avg_sample_ms": summarize_runs(sample_values),
          "avg_process_outputs_ms": summarize_runs(process_values),
      }
      first_config_summary = run_payloads[0]["config_summary"] if run_payloads else {}
      max_batched_tokens_request = first_config_summary.get(
          "max_batched_tokens_request", str(workload["max_batched_tokens"]))
      max_batched_tokens_resolved = first_config_summary.get(
          "max_batched_tokens_resolved", str(workload["max_batched_tokens"]))
      prefill_chunk_cap_request = first_config_summary.get(
          "prefill_chunk_cap_request", str(workload["prefill_chunk_cap"]))
      prefill_chunk_cap_resolved = first_config_summary.get(
          "prefill_chunk_cap_resolved", str(workload["prefill_chunk_cap"]))
      gpu_memory_utilization = first_config_summary.get(
          "gpu_memory_utilization", str(args.gpu_memory_utilization))
      config_details = {
          key: first_config_summary[key]
          for key in CONFIG_DETAIL_KEYS
          if key in first_config_summary
      }
      workload_reports.append(
          {
              "name": workload["name"],
              "description": workload["description"],
              "prompts": deepcopy(workload["prompts"]),
                "config": {
                    "max_new_tokens": workload["max_new_tokens"],
                    "max_batched_tokens_request": max_batched_tokens_request,
                    "max_batched_tokens_resolved": max_batched_tokens_resolved,
                    "prefill_chunk_cap_request": prefill_chunk_cap_request,
                    "prefill_chunk_cap_resolved": prefill_chunk_cap_resolved,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "warmup_rounds": args.warmup_rounds,
                    **config_details,
                },
              "runs": run_payloads,
              "aggregate": aggregate,
          }
      )

    raw_path = output_dir / "pagedbench-baseline.json"
    md_path = output_dir / "pagedbench-baseline.md"
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
