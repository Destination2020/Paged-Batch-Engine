#!/usr/bin/env python3
"""Historical cold-start topology microbenchmark; not serving acceptance evidence."""
import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = {"rounds": [[{"request_id": "deployment-ab", "generation": 1,
             "parts": [{"type": "image", "path": str(args.image), "size": 224},
                       {"type": "text", "text":
                        "Describe the image briefly and name its main subject."}],
             "max_new_tokens": 16, "timeout_ms": 120000}]]}
    spec_path = args.output / "merged-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    order = [mode for _ in range(5) for mode in ("merged_language", "separated_pd")]
    random.Random(20260913).shuffle(order)
    records = []
    counts = {"merged_language": 0, "separated_pd": 0}
    root = Path(__file__).resolve().parents[3]
    for order_index, mode in enumerate(order):
        repetition = counts[mode]; counts[mode] += 1
        run_dir = args.output / mode / f"repeat-{repetition}"
        run_dir.mkdir(parents=True)
        if mode == "merged_language":
            command = ["tools/launch/persistent_vlm_online.sh", str(args.build),
                       str(args.model), str(args.tokenizer), str(args.model_dir), str(args.image)]
            env = os.environ.copy(); env.update(
                PBE_RUN_DIR=str(run_dir), PBE_SPEC=str(spec_path),
                PBE_PYTHON=str(root / ".venv/bin/python"))
        else:
            command = ["tools/launch/separated_vlm_one_request.sh", str(args.build),
                       str(args.model), str(args.tokenizer), str(args.model_dir),
                       str(args.image), str(run_dir)]
            env = os.environ.copy()
        started = time.perf_counter()
        process = subprocess.run(command, cwd=root, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        wall_ms = (time.perf_counter() - started) * 1000
        (run_dir / "launch.log").write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f"{mode} failed: {process.stdout[-2000:]}")
        if mode == "merged_language":
            value = json.loads((run_dir / "result.json").read_text())
            tokens = value["rounds"][0]["outputs"][0]["tokens"]
            peak = value["status"]["device_memory"]["peak_observed_used_bytes"]
        else:
            line = (run_dir / "decode.log").read_text()
            token_text = line.split(" tokens=", 1)[1].split(" text=", 1)[0]
            tokens = [int(x) for x in token_text.split(",") if x]
            peak = None
        records.append({"mode": mode, "repetition": repetition,
                        "order": order_index, "wall_ms": wall_ms,
                        "tokens": tokens, "peak_observed_used_bytes": peak})
    # BF16 process topology may alter ties; require each mode to be internally stable.
    for mode in counts:
        values = [r["tokens"] for r in records if r["mode"] == mode]
        assert all(value == values[0] for value in values)
    summaries = {}
    for mode in counts:
        rows = [r for r in records if r["mode"] == mode]
        summaries[mode] = {"repeats": 5,
                           "median_wall_ms": statistics.median(r["wall_ms"] for r in rows),
                           "min_wall_ms": min(r["wall_ms"] for r in rows),
                           "max_wall_ms": max(r["wall_ms"] for r in rows),
                           "model_processes_per_request": 1 if mode == "merged_language" else 2}
    result = {"ok": True, "seed": 20260913,
              "scope": "cold_start_plus_one_request_microbenchmark",
              "counts_as_persistent_serving_acceptance": False,
              "same_gpu_count": 1, "same_kv_blocks": 64,
              "same_output_tokens": records[0]["tokens"] == next(
                  r["tokens"] for r in records if r["mode"] != records[0]["mode"]),
              "summaries": summaries, "records": records}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
