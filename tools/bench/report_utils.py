#!/usr/bin/env python3
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def try_git_info(repo_root: Path) -> dict[str, str]:
    info = {
        "commit": "unknown",
        "branch": "unknown",
    }
    code, out, _ = run_command(["git", "rev-parse", "HEAD"], repo_root)
    if code == 0:
      info["commit"] = out.strip()
    code, out, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if code == 0:
      info["branch"] = out.strip()
    return info


def try_nvidia_smi() -> dict[str, str]:
    code, out, _ = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0 or not out.strip():
        return {"gpu": "unknown", "gpu_memory_mb": "unknown", "driver": "unknown"}
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    while len(parts) < 3:
        parts.append("unknown")
    return {"gpu": parts[0], "gpu_memory_mb": parts[1], "driver": parts[2]}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_runs(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def format_metric(value: float, unit: str = "", precision: int = 3) -> str:
    suffix = f" {unit}" if unit else ""
    return f"{value:.{precision}f}{suffix}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=default)
        f.write("\n")


def parse_workload_names(csv_text: str) -> set[str] | None:
    if not csv_text:
        return None
    names = {name.strip() for name in csv_text.split(",") if name.strip()}
    return names or None


def load_workloads(
    path: Path,
    *,
    selected_names: set[str] | None,
    default_name: str,
    default_description: str,
    max_new_tokens: int,
    max_batched_tokens: int | None = None,
    prefill_chunk_cap: int | None = None,
    force_cli_config: bool = False,
) -> list[dict[str, Any]]:
    def resolve_config_value(spec: dict[str, Any], key: str, fallback: int | str | None):
        value = spec.get(key, fallback)
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() == "auto":
                return "auto"
            return int(value)
        return int(value)

    payload = load_json(path)
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError("legacy prompts file must contain a non-empty JSON array")
        return [
            {
                "name": default_name,
                "description": default_description,
                "prompts": list(payload),
                "max_new_tokens": max_new_tokens,
                "max_batched_tokens": max_batched_tokens,
                "prefill_chunk_cap": prefill_chunk_cap,
            }
        ]

    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("workloads file must contain a non-empty JSON object or array")

    workloads = []
    for name, spec in payload.items():
        if selected_names and name not in selected_names:
            continue
        if not isinstance(spec, dict):
            raise RuntimeError(f"workload '{name}' must be a JSON object")

        prompts = spec.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"workload '{name}' must define a non-empty 'prompts' list")

        workloads.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "prompts": list(prompts),
                "max_new_tokens": max_new_tokens if force_cli_config else int(
                    spec.get("max_new_tokens", max_new_tokens)),
                "max_batched_tokens": (
                    max_batched_tokens if force_cli_config else resolve_config_value(
                        spec, "max_batched_tokens", max_batched_tokens)),
                "prefill_chunk_cap": (
                    prefill_chunk_cap if force_cli_config else resolve_config_value(
                        spec, "prefill_chunk_cap", prefill_chunk_cap)),
            }
        )

    if not workloads:
        raise RuntimeError("no workloads selected")
    return workloads


@dataclass
class ReportContext:
    repo_root: Path
    output_dir: Path
    model_path: str
    tokenizer_path: str
    prompts_path: str
    max_new_tokens: int
    runs: int
    created_at: str
    git_commit: str
    git_branch: str
    gpu_name: str
    gpu_memory_mb: str
    driver_version: str


def build_report_context(
    repo_root: Path,
    output_dir: Path,
    model_path: str,
    tokenizer_path: str,
    prompts_path: str,
    max_new_tokens: int,
    runs: int,
) -> ReportContext:
    git = try_git_info(repo_root)
    gpu = try_nvidia_smi()
    return ReportContext(
        repo_root=repo_root,
        output_dir=output_dir,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        prompts_path=prompts_path,
        max_new_tokens=max_new_tokens,
        runs=runs,
        created_at=utc_now_iso(),
        git_commit=git["commit"],
        git_branch=git["branch"],
        gpu_name=gpu["gpu"],
        gpu_memory_mb=gpu["gpu_memory_mb"],
        driver_version=gpu["driver"],
    )
