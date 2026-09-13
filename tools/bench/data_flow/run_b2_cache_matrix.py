#!/usr/bin/env python3
"""Run the frozen B2 feature-cache x semantic-KV characterization matrix."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_diagnostics(items):
    """Validate BF16 error only on equal histories and justify first divergence."""
    chains = {}
    for item in items:
        chains.setdefault((item.get("diagnostic_key"), item.get("mode")), []).append(item)
    audit = {"same_history_steps": 0, "justified_first_divergences": 0,
             "post_divergence_steps": 0, "failures": []}
    for chain_key, chain in chains.items():
        diverged = False
        for item in sorted(chain, key=lambda value: value.get("step", -1)):
            if item.get("same_history") is not True:
                if not diverged:
                    audit["failures"].append({"chain": chain_key, "step": item.get("step"),
                                              "reason": "history_changed_without_observed_first_divergence"})
                audit["post_divergence_steps"] += 1
                continue
            audit["same_history_steps"] += 1
            if item.get("logits_max_abs", 1) >= .75 or item.get("logits_mean_abs", 1) >= .20:
                audit["failures"].append({"chain": chain_key, "step": item.get("step"),
                                          "reason": "same_history_logits_error"})
            if item.get("baseline_selected_token") != item.get("selected_token"):
                baseline_top2 = set(item.get("baseline_top2_ids", []))
                candidate_top2 = set(item.get("top2_ids", []))
                justified = (baseline_top2 == candidate_top2 and
                             item.get("baseline_selected_token") in candidate_top2 and
                             item.get("selected_token") in baseline_top2 and
                             item.get("baseline_top2_margin", 1) <= .25 and
                             item.get("top2_margin", 1) <= .25)
                if justified:
                    diverged = True
                    audit["justified_first_divergences"] += 1
                else:
                    audit["failures"].append({"chain": chain_key, "step": item.get("step"),
                                              "reason": "unjustified_first_token_divergence"})
    return not audit["failures"], audit


def make_variants(source: Path, output: Path) -> list[Path]:
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    paths = []
    for index in range(11):
        variant = image.copy()
        pixels = variant.load()
        for offset in range(4):
            x = (index * 17 + offset * 31) % variant.width
            y = (index * 29 + offset * 13) % variant.height
            r, g, b = pixels[x, y]
            pixels[x, y] = ((r + index + offset + 1) % 256, g, b)
        path = output / f"variant-{index:02d}.png"
        variant.save(path)
        paths.append(path)
    return paths


def measured_images(variants, reuse):
    if reuse == 0:
        return variants[:10]
    if reuse == 50:
        return [variants[index // 2] for index in range(10)]
    if reuse == 90:
        return [variants[0]] * 10
    raise ValueError(reuse)


def spec_for(variants, reuse, resolution, context, trial):
    short = ["Describe the main subject and layout of this image.",
             "Describe the main layout and subject of this image."]
    long_base = " ".join(["Describe every visible component, relationship, direction, label, and layout detail."] * 12)
    long = [long_base + " Finish with the main subject.",
            long_base + " Finish with the central object."]
    warmup = {"request_id": f"warmup-{trial}", "generation": 1,
              "parts": [{"type": "image", "path": str(variants[10]), "size": resolution},
                        {"type": "text", "text": "Warm the model without entering the measured identities."}],
              "max_new_tokens": 8}
    rounds = [[warmup]]
    images = measured_images(variants, reuse)
    groups = list(range(10)) if reuse == 0 else ([index // 2 for index in range(10)]
                                                  if reuse == 50 else [0] * 10)
    for index, (image, group) in enumerate(zip(images, groups)):
        # Paired arms see the exact same prompts.  At 0% this gives different
        # media with the same placeholder/text; at 50% repeated pairs exercise
        # exact-prefix reuse; at 90% both exact-prefix and different-question
        # reuse are present.
        variant = (index % 2) if reuse == 90 else 0
        text = (long if context == "long" else short)[variant]
        rounds.append([{"request_id": f"b2-{trial}-{index:02d}", "generation": 1,
                        "parts": [{"type": "image", "path": str(image), "size": resolution},
                                  {"type": "text", "text": text}],
                        "max_new_tokens": 8,
                        "diagnostic_key": f"b2-{trial}-content-{group}-question-{variant}",
                        "diagnostic_mode": f"request-{index}",
                        "diagnostic_dump_logits_step": 0}])
    return {"rounds": rounds}


def extract(trial_dir, cell, arm, repetition, order, process_ms, command, reuse):
    result = json.loads((trial_dir / "result.json").read_text())
    traces = [json.loads(line) for line in (trial_dir / "trace.jsonl").read_text().splitlines()]
    encodes = {}
    for event in traces:
        if event.get("stage") == "encode" and isinstance(event.get("result"), dict):
            encodes[event["request_id"]] = event["result"]
    request_rows = []
    diagnostics = []
    for round_index, round_result in enumerate(result["rounds"][1:], 1):
        diagnostics.extend(round_result.get("numerical_diagnostics", []))
        for output in round_result.get("outputs", []):
            identity = output["request_id"]
            encode = encodes.get(identity, {})
            request_rows.append({
                "schema": "pbe-v4-performance-request-v1", "experiment": "B2",
                "cell": cell, "arm": arm, "repetition": repetition, "order": order,
                "request_id": identity, "round_index": round_index,
                "scheduled_arrival_ns": None,
                "actual_send_ns": round_result.get("coordinator_started_ns"),
                "first_visible_token_ns": None,
                "terminal_ns": round_result.get("coordinator_terminal_ns"),
                "timing_boundary": "server and coordinator durations; non-streaming client",
                "status": "failed" if output.get("failed") else "completed",
                "server_ttft_ms": output.get("ttft_ms"), "server_itl_ms": output.get("itl_ms"),
                "token_itl_ms": output.get("token_itl_ms", []),
                "round_latency_ms": round_result.get("round_latency_ms"),
                "language_latency_ms": output.get("latency_ms"),
                "prompt_tokens": output.get("semantic_prompt_tokens"),
                "matched_tokens": output.get("semantic_matched_tokens"),
                "actual_computed_tokens": output.get("semantic_actual_computed_prompt_tokens"),
                "saved_tokens": (output.get("semantic_prompt_tokens", 0) -
                                 output.get("semantic_actual_computed_prompt_tokens", 0)),
                "lookup_us": output.get("semantic_lookup_us"),
                "output_tokens": output.get("tokens", []),
                "feature_cache_hit": encode.get("feature_cache_hit"),
                "feature_cache_enabled": encode.get("feature_cache_enabled"),
                "feature_cache_tier": encode.get("feature_cache_tier"),
                "canonical_media_content": encode.get("feature_content"),
                "vision_forward_ms": encode.get("forward_ms"),
                "vision_physical_forward_batch": encode.get("physical_forward_batch_size"),
                "feature_rows": encode.get("feature_rows"),
                "bundle_bytes": encode.get("bundle_bytes"),
                "worker_pid": output.get("worker_pid", result["status"].get("worker_pid")),
                "identity_scenario": (
                    "different_media_same_placeholder_and_text" if cell.startswith("reuse00")
                    else "same_media_same_prefix" if cell.startswith("reuse50")
                    else ("same_media_same_prefix" if round_index % 2 == 1
                          else "same_media_different_question")),
            })
    comparison_diagnostics = [item for item in diagnostics
                              if item.get("comparison") != "baseline_captured"]
    numerical_ok, numerical_audit = validate_diagnostics(comparison_diagnostics)
    canonical_ids = [row["canonical_media_content"] for row in request_rows]
    expected_unique_media = {0: 10, 50: 5, 90: 1}[reuse]
    identity_ok = len(set(canonical_ids)) == expected_unique_media
    feature_hits = sum(row["feature_cache_hit"] is True for row in request_rows)
    matched_tokens = sum(row["matched_tokens"] or 0 for row in request_rows)
    cache_behavior_ok = (
        all(row["feature_cache_enabled"] is (arm[0] == "1") for row in request_rows)
        and (feature_hits == 0 if arm[0] == "0" else
             feature_hits >= {0: 0, 50: 5, 90: 9}[reuse])
        and (matched_tokens == 0 if arm[1] == "0" else
             (reuse == 0 or matched_tokens > 0))
    )
    cohort_wall_ms = ((max(row["terminal_ns"] for row in request_rows) -
                       min(row["actual_send_ns"] for row in request_rows)) / 1_000_000)
    return {
        "schema": "pbe-v4-b2-trial-v1", "cell": cell, "arm": arm,
        "repetition": repetition, "order": order, "ok": result.get("ok") is True,
        "feature_cache": arm[0] == "1", "semantic_kv_cache": arm[1] == "1",
        "process_wall_ms_including_startup": process_ms,
        "measured_cohort_wall_ms": cohort_wall_ms,
        "cohort_window_definition": "first measured coordinator send to last terminal, including drain",
        "command": command, "requests": request_rows,
        "numerical_diagnostics": comparison_diagnostics, "numerical_ok": numerical_ok,
        "numerical_audit": numerical_audit,
        "identity_ok": identity_ok, "cache_behavior_ok": cache_behavior_ok,
        "canonical_media_identities": sorted(set(canonical_ids)),
        "resource_recovered": "free_slots=" in (trial_dir / "ipc-after-language.log").read_text()
                              and "active_grants=0" in (trial_dir / "ipc-after-language.log").read_text(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    variants = make_variants(args.image, output / "inputs")
    cells = [
        ("reuse00_short_r224", 0, 224, "short", ("00", "11")),
        ("reuse50_short_r224", 50, 224, "short", ("00", "10", "01", "11")),
        ("reuse90_long_r280", 90, 280, "long", ("00", "11")),
    ]
    jobs = [(name, reuse, resolution, context, arm, repetition)
            for name, reuse, resolution, context, arms in cells
            for arm in arms for repetition in range(args.repeats)]
    random.Random(20260913).shuffle(jobs)
    records = []
    commands = []
    for order, (name, reuse, resolution, context, arm, repetition) in enumerate(jobs):
        trial_name = f"{order:02d}-{name}-arm{arm}-r{repetition}"
        trial_dir = output / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        spec = spec_for(variants, reuse, resolution, context, trial_name)
        spec_path = trial_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        invocation = ["tools/launch/persistent_vlm_online.sh", str(args.build),
                      str(args.model_bin), str(args.tokenizer), str(args.model_dir),
                      str(args.image)]
        environment = dict(os.environ, PBE_DEVICE=str(args.device), PBE_RUN_DIR=str(trial_dir),
                           PBE_SPEC=str(spec_path), PBE_WEIGHT_MODE="shared", PYTHONPATH="python")
        environment["PBE_DETERMINISTIC_VISION"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if arm[0] == "0": environment["PBE_DISABLE_FEATURE_CACHE"] = "1"
        if arm[1] == "0": environment["PBE_DISABLE_RADIX_CACHE"] = "1"
        command_record = {"argv": invocation, "environment": {key: environment[key] for key in
            ("PBE_DEVICE", "PBE_RUN_DIR", "PBE_SPEC", "PBE_WEIGHT_MODE")},
            "feature_cache": arm[0] == "1", "semantic_kv_cache": arm[1] == "1"}
        if arm[0] == "0": command_record["environment"]["PBE_DISABLE_FEATURE_CACHE"] = "1"
        if arm[1] == "0": command_record["environment"]["PBE_DISABLE_RADIX_CACHE"] = "1"
        command_record["environment"]["PBE_DETERMINISTIC_VISION"] = "1"
        command_record["environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        commands.append(command_record)
        print(json.dumps({"event": "trial_start", "order": order, "trial": trial_name}), flush=True)
        started = time.perf_counter()
        with (trial_dir / "launch.log").open("w") as log:
            completed = subprocess.run(invocation, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, text=True, timeout=600)
        process_ms = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            raise RuntimeError(f"B2 trial failed: {trial_name}, exit={completed.returncode}")
        record = extract(trial_dir, name, arm, repetition, order, process_ms,
                         command_record, reuse)
        record.update({"image_reuse_percent": reuse, "resolution": resolution,
                       "context": context})
        (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)
    raw = output / "requests.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                           for trial in records for row in trial["requests"]))
    (output / "commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")
    summaries = {}
    for name, _, _, _, arms in cells:
        for arm in arms:
            rows = [row for row in records if row["cell"] == name and row["arm"] == arm]
            requests = [request for row in rows for request in row["requests"]]
            trial_throughputs = [
                sum(request["status"] == "completed" for request in row["requests"]) /
                (row["measured_cohort_wall_ms"] / 1000) for row in rows]
            summaries[f"{name}.arm{arm}"] = {
                "trials": len(rows), "requests": len(requests),
                "feature_cache": arm[0] == "1", "semantic_kv_cache": arm[1] == "1",
                "throughput_rps_median": statistics.median(trial_throughputs),
                "throughput_rps_min": min(trial_throughputs),
                "throughput_rps_max": max(trial_throughputs),
                "round_latency_ms_median": statistics.median(r["round_latency_ms"] for r in requests),
                "server_ttft_ms_median": statistics.median(r["server_ttft_ms"] for r in requests),
                "server_itl_ms_median": statistics.median(r["server_itl_ms"] for r in requests),
                "feature_hits": sum(r["feature_cache_hit"] is True for r in requests),
                "vision_physical_forwards": sum((r["vision_physical_forward_batch"] or 0)
                                                for r in requests),
                "matched_tokens": sum(r["matched_tokens"] or 0 for r in requests),
                "actual_computed_tokens": sum(r["actual_computed_tokens"] or 0 for r in requests),
                "saved_tokens": sum(r["saved_tokens"] or 0 for r in requests),
                "errors": sum(r["status"] != "completed" for r in requests),
            }
    identity_across_arms_ok = all(
        len({tuple(row["canonical_media_identities"]) for row in records
             if row["cell"] == name}) == 1
        for name, _, _, _, _ in cells)
    result = {"schema": "pbe-v4-b2-cache-matrix-v1", "ok": len(records) == len(jobs) and
              all(row["ok"] and row["numerical_ok"] and row["identity_ok"] and
                  row["cache_behavior_ok"] and row["resource_recovered"]
                  for row in records) and identity_across_arms_ok,
              "random_seed": 20260913, "records": records, "summary": summaries,
              "identity_across_cache_policies_ok": identity_across_arms_ok,
              "timing_limit": "non-streaming client; server TTFT and coordinator round latency only"}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "ok": result["ok"],
                      "trials": len(records), "requests": len(records) * 10}, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
