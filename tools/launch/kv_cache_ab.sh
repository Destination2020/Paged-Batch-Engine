#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 6 ]]; then echo "usage: $0 BUILD MODEL TOKENIZER MODEL_DIR IMAGE RUN_DIR" >&2; exit 2; fi
build=$1; model=$2; tokenizer=$3; model_dir=$4; image=$5; run_dir=$6
python_bin=${PBE_PYTHON:-.venv/bin/python}; mkdir -p "$run_dir"
"$python_bin" tools/bench/data_flow/make_kv_cache_ab_spec.py --image "$image" --output "$run_dir/spec.json"
for mode in off on; do
  for seed in 11 23 37 41 53; do
    output="$run_dir/$mode/seed-$seed"; mkdir -p "$output"
    if [[ "$mode" == off ]]; then disable=1; else disable=; fi
    PBE_DISABLE_RADIX_CACHE="$disable" PBE_RUN_DIR="$output" PBE_SPEC="$run_dir/spec.json" \
      PBE_PYTHON="$python_bin" tools/launch/persistent_vlm_online.sh \
      "$build" "$model" "$tokenizer" "$model_dir" "$image" >"$output/launch.log" 2>&1
  done
done
"$python_bin" tools/bench/data_flow/summarize_kv_cache_ab.py \
  --runs "$run_dir" --output "$run_dir/results.json" | tee "$run_dir/validation.log"
echo "PBE_KV_CACHE_AB_OK evidence=$run_dir"
