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
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
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
        "--output-dir",
        args.output_dir,
    ]
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
        "--output-dir",
        args.output_dir,
        "--gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
    ]
    if args.vllm_tokenizer:
        vllm_cmd.extend(["--tokenizer", args.vllm_tokenizer])
    run_checked(vllm_cmd, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
