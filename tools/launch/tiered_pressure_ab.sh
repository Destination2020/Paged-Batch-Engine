#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 7 ]]; then echo "usage: $0 BUILD MODEL TOKENIZER MODEL_DIR IMAGE KV_AB RUN_DIR" >&2; exit 2; fi
build=$1; model=$2; tokenizer=$3; model_dir=$4; image=$5; kv_ab=$6; run_dir=$7
python_bin=${PBE_PYTHON:-.venv/bin/python}; mkdir -p "$run_dir"
"$python_bin" tools/bench/data_flow/make_tiered_recovery_spec.py --image "$image" --output "$run_dir/spec.json"
for seed in 11 23 37 41 53; do
  output="$run_dir/seed-$seed"; mkdir -p "$output"
  PBE_RUN_DIR="$output" PBE_SPEC="$run_dir/spec.json" PBE_PYTHON="$python_bin" \
    tools/launch/persistent_vlm_online.sh "$build" "$model" "$tokenizer" "$model_dir" "$image" \
    >"$output/launch.log" 2>&1
  "$python_bin" tools/bench/data_flow/validate_tiered_recovery.py \
    "$output/result.json" "$output/trace.jsonl" >"$output/validation.json"
done
"$python_bin" tools/bench/data_flow/summarize_pressure_policies.py \
  --kv-ab "$kv_ab" --host-runs "$run_dir" --output "$run_dir/results.json" \
  | tee "$run_dir/validation.log"
echo "PBE_TIERED_PRESSURE_AB_OK evidence=$run_dir"
