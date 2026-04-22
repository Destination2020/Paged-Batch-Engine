#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from report_utils import load_workloads, parse_workload_names


def parse_summary_lines(stdout: str) -> tuple[dict, dict]:
    config_summary = {}
    final_summary = {}
    for line in stdout.splitlines():
        if line.startswith("CONFIG_SUMMARY "):
            for item in line.split()[1:]:
                key, value = item.split("=", 1)
                config_summary[key] = value
        elif line.startswith("FINAL_SUMMARY "):
            for item in line.split()[1:]:
                key, value = item.split("=", 1)
                final_summary[key] = value
    return config_summary, final_summary


def parse_metric(summary: dict, key: str) -> float:
    return float(summary.get(key, 0.0) or 0.0)


def aggregate_runs(run_rows: list[dict]) -> dict:
    successful_runs = [row for row in run_rows if not row["failed"]]
    aggregate = {
        "runs": len(run_rows),
        "successful_runs": len(successful_runs),
        "failed_runs": len(run_rows) - len(successful_runs),
    }
    if not successful_runs:
        return aggregate

    throughput_values = [parse_metric(row["final_summary"], "throughput_tps") for row in successful_runs]
    ttft_values = [parse_metric(row["final_summary"], "ttft_ms") for row in successful_runs]
    itl_values = [parse_metric(row["final_summary"], "itl_ms") for row in successful_runs]
    latency_p99_values = [parse_metric(row["final_summary"], "latency_p99_ms") for row in successful_runs]

    aggregate.update(
        {
            "throughput_tps_mean": statistics.fmean(throughput_values),
            "throughput_tps_min": min(throughput_values),
            "throughput_tps_max": max(throughput_values),
            "ttft_ms_mean": statistics.fmean(ttft_values),
            "itl_ms_mean": statistics.fmean(itl_values),
            "latency_p99_ms_mean": statistics.fmean(latency_p99_values),
        }
    )
    return aggregate


def resolve_inputs(args, repo_root: Path) -> tuple[list[str], int, dict]:
    if args.workload:
        workloads = load_workloads(
            (repo_root / args.workloads_file).resolve(),
            selected_names=parse_workload_names(args.workload),
            default_name="default",
            default_description="Sweep workload.",
            max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 0,
            max_batched_tokens=None,
            prefill_chunk_cap=None,
            force_cli_config=False,
        )
        if len(workloads) != 1:
            raise RuntimeError(f"Expected exactly one workload, got {len(workloads)}")
        workload = workloads[0]
        return (
            workload["prompts"],
            int(workload["max_new_tokens"]),
            {
                "source": "workload",
                "name": workload["name"],
                "description": workload.get("description", ""),
                "workloads_file": args.workloads_file,
            },
        )

    if not args.prompts:
        raise RuntimeError("Either --prompts or --workload must be provided")
    if args.max_new_tokens is None:
        raise RuntimeError("--max-new-tokens is required when using --prompts")
    return (
        args.prompts,
        args.max_new_tokens,
        {
            "source": "prompts",
            "name": "",
            "description": "",
            "workloads_file": "",
        },
    )


def run_once(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )

    config_summary, final_summary = parse_summary_lines(proc.stdout)
    row = {
        "failed": proc.returncode != 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "config_summary": config_summary,
        "final_summary": final_summary,
    }

    if proc.returncode == 0 and (not config_summary or not final_summary):
        raise RuntimeError(
            f"Failed to parse CONFIG_SUMMARY/FINAL_SUMMARY\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/demo/serving_qwen")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompts", nargs="+")
    input_group.add_argument("--workload")
    parser.add_argument("--workloads-file", default="tools/bench/workloads.json")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batched-token-candidates", default="192,224,256,288")
    parser.add_argument("--prefill-candidates", default="32,48,64")
    parser.add_argument("--output", default="docs/benchmarks/auto-schedule-sweep.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts, max_new_tokens, input_meta = resolve_inputs(args, repo_root)

    batched_candidates = [int(x) for x in args.batched_token_candidates.split(",") if x.strip()]
    prefill_candidates = [int(x) for x in args.prefill_candidates.split(",") if x.strip()]

    if input_meta["source"] == "workload":
        print(
            "Loaded workload:",
            f"name={input_meta['name']}",
            f"max_new_tokens={max_new_tokens}",
            f"prompts={len(prompts)}",
        )

    rows = []
    for max_batched_tokens in batched_candidates:
        for prefill_chunk_cap in prefill_candidates:
            run_rows = []
            for run_idx in range(args.runs):
                cmd = [
                    args.binary,
                    args.model,
                    args.tokenizer,
                    *prompts,
                    f"--max-new-tokens={max_new_tokens}",
                    f"--max-batched-tokens={max_batched_tokens}",
                    f"--prefill-chunk-cap={prefill_chunk_cap}",
                    f"--gpu-memory-utilization={args.gpu_memory_utilization}",
                    f"--warmup-rounds={args.warmup_rounds}",
                    "--quiet=1",
                    "--step-profile=0",
                    "--final-summary=1",
                ]
                print(f"+ [run {run_idx + 1}/{args.runs}]", " ".join(cmd))
                run_row = run_once(cmd, repo_root)
                run_row["run_index"] = run_idx + 1
                run_rows.append(run_row)

            row = {
                "max_batched_tokens": max_batched_tokens,
                "prefill_chunk_cap": prefill_chunk_cap,
                "aggregate": aggregate_runs(run_rows),
                "runs": run_rows,
                "failed": any(run_row["failed"] for run_row in run_rows),
            }
            successful_runs = [run_row for run_row in run_rows if not run_row["failed"]]
            if successful_runs:
                row["config_summary"] = successful_runs[0]["config_summary"]
                row["final_summary"] = successful_runs[0]["final_summary"]
            else:
                row["config_summary"] = {}
                row["final_summary"] = {}
                row["returncode"] = run_rows[0]["returncode"] if run_rows else 0
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["aggregate"].get("successful_runs", 0) == 0,
            -row["aggregate"].get("throughput_tps_mean", 0.0),
            row["aggregate"].get("ttft_ms_mean", float("inf")),
            row["aggregate"].get("latency_p99_ms_mean", float("inf")),
        )
    )

    output = {
        "binary": args.binary,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "input": input_meta,
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_rounds": args.warmup_rounds,
        "runs": args.runs,
        "rows": rows,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")

    successful_rows = [row for row in rows if row["aggregate"].get("successful_runs", 0) > 0]
    failed_rows = [row for row in rows if row["aggregate"].get("successful_runs", 0) == 0]
    print(f"Successful runs: {len(successful_rows)}  Failed runs: {len(failed_rows)}")

    if successful_rows:
        best = successful_rows[0]
        summary = best["aggregate"]
        print(
            "Best:",
            f"max_batched_tokens={best['max_batched_tokens']}",
            f"prefill_chunk_cap={best['prefill_chunk_cap']}",
            f"throughput_tps_mean={summary.get('throughput_tps_mean', '0'):.3f}",
            f"ttft_ms_mean={summary.get('ttft_ms_mean', '0'):.3f}",
            f"latency_p99_ms_mean={summary.get('latency_p99_ms_mean', '0'):.3f}",
        )
    if failed_rows:
        print("Failed configs:")
        for row in failed_rows:
            print(
                " ",
                f"max_batched_tokens={row['max_batched_tokens']}",
                f"prefill_chunk_cap={row['prefill_chunk_cap']}",
                f"failed_runs={row['aggregate'].get('failed_runs', 0)}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
