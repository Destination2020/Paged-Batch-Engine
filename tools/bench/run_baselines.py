#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_checked(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/demo/serving_qwen")
    parser.add_argument("--model-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--vllm-tokenizer", default=None)
    parser.add_argument("--prompts", default="tools/bench/workloads.json")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-batched-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-cap", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--force-cli-config", action="store_true")
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--readme-tables-output",
                        default=None,
                        help="Markdown file for README-ready summary tables. "
                             "Defaults to <output-dir>/readme-tables.md.")
    parser.add_argument("--skip-readme-tables", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-disable-prefix-caching", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--no-align-vllm-prompt-stats", action="store_true",
                        help="Do not copy PagedBatchEngine prompt token stats into "
                             "the vLLM README table config.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    pagedbench_cmd = [
        sys.executable,
        "tools/bench/run_pagedbench.py",
        "--binary",
        args.binary,
        "--model",
        args.model_bin,
        "--tokenizer",
        args.tokenizer,
        "--prompts",
        args.prompts,
        "--workloads",
        args.workloads,
        "--runs",
        str(args.runs),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-batched-tokens",
        str(args.max_batched_tokens),
        "--prefill-chunk-cap",
        str(args.prefill_chunk_cap),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--output-dir",
        args.output_dir,
    ]
    if args.force_cli_config:
        pagedbench_cmd.append("--force-cli-config")
    run_checked(pagedbench_cmd, repo_root)

    vllm_cmd = [
        sys.executable,
        "tools/bench/run_vllm_bench.py",
        "--model",
        args.vllm_model,
        "--prompts",
        args.prompts,
        "--workloads",
        args.workloads,
        "--runs",
        str(args.runs),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--output-dir",
        args.output_dir,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
    ]
    if args.force_cli_config:
        vllm_cmd.append("--force-cli-config")
    if args.vllm_disable_prefix_caching:
        vllm_cmd.append("--disable-prefix-caching")
    if args.vllm_enforce_eager:
        vllm_cmd.append("--enforce-eager")
    if not args.no_align_vllm_prompt_stats:
        vllm_cmd.extend([
            "--prompt-stats-override-json",
            str(Path(args.output_dir) / "pagedbench-baseline.json"),
        ])
    if args.vllm_tokenizer:
        vllm_cmd.extend(["--tokenizer", args.vllm_tokenizer])
    run_checked(vllm_cmd, repo_root)

    if not args.skip_readme_tables:
        readme_tables_output = (
            args.readme_tables_output
            if args.readme_tables_output
            else str(Path(args.output_dir) / "readme-tables.md")
        )
        readme_cmd = [
            sys.executable,
            "tools/bench/render_readme_tables.py",
            "--paged-json",
            str(Path(args.output_dir) / "pagedbench-baseline.json"),
            "--vllm-json",
            str(Path(args.output_dir) / "vllm-baseline.json"),
            "--output",
            readme_tables_output,
            "--workloads",
            args.workloads,
        ]
        run_checked(readme_cmd, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
