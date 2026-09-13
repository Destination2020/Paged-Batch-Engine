#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then echo "usage: $0 BUILD MODEL_DIR IMAGE RUN_DIR PYTHON" >&2; exit 2; fi
build=$1; model_dir=$2; image=$3; run_dir=$4; python_bin=$5
service="$build/demo/pbe_data_service"; mkdir -p "$run_dir"
for mode in off on; do
  for seed in 11 23 37 41 53; do
    output="$run_dir/$mode/seed-$seed"; mkdir -p "$output"
    data_endpoint="/tmp/pbe-singleflight-data-$$-$mode-$seed.sock"
    vision_endpoint="/tmp/pbe-singleflight-vision-$$-$mode-$seed.sock"
    "$service" serve "$data_endpoint" 1073741824 1024 >"$output/service.log" 2>&1 & service_pid=$!
    for _ in $(seq 1 400); do [[ -S "$data_endpoint" ]] && break; sleep .01; done
    extra=(); [[ "$mode" == off ]] && extra+=(--disable-singleflight)
    PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
      --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
      --device cuda:0 --batch-window-ms 200 --max-batch 8 "${extra[@]}" \
      >"$output/vision.log" 2>&1 & vision_pid=$!
    for _ in $(seq 1 6000); do [[ -S "$vision_endpoint" ]] && break; sleep .01; done
    PYTHONPATH=python "$python_bin" tools/bench/data_flow/probe_vision_singleflight.py \
      --endpoint "$vision_endpoint" --image "$image" --output "$output/result.json" \
      >"$output/stdout.json"
    "$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" --op shutdown >/dev/null
    wait "$vision_pid"
    "$service" shutdown "$data_endpoint" >/dev/null; wait "$service_pid"
  done
done
"$python_bin" tools/bench/data_flow/summarize_singleflight_ab.py \
  --runs "$run_dir" --output "$run_dir/results.json" | tee "$run_dir/validation.log"
echo "PBE_VISION_SINGLEFLIGHT_AB_OK evidence=$run_dir"
