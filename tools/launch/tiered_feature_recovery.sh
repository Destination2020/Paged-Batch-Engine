#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: $0 BUILD MODEL_DIR IMAGE RUN_DIR PYTHON" >&2
  exit 2
fi
build=$1; model_dir=$2; image=$3; run_dir=$4; python_bin=$5
mkdir -p "$run_dir"
data_endpoint="/tmp/pbe-tiered-feature-data-$$.sock"
vision_endpoint="/tmp/pbe-tiered-feature-vision-$$.sock"
"$build/demo/pbe_data_service" serve "$data_endpoint" 1073741824 1024 \
  >"$run_dir/data-service.log" 2>&1 &
service_pid=$!
vision_pid=
cleanup() {
  if [[ -n "$vision_pid" ]]; then
    "$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" \
      --op shutdown >/dev/null 2>&1 || true
    wait "$vision_pid" 2>/dev/null || true
  fi
  "$build/demo/pbe_data_service" shutdown "$data_endpoint" >/dev/null 2>&1 || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 400); do [[ -S "$data_endpoint" ]] && break; sleep .01; done
PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
  --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
  --device cuda:0 --batch-window-ms 10 --max-batch 8 --gpu-feature-cache-bytes 33554432 \
  >"$run_dir/vision.log" 2>&1 &
vision_pid=$!
for _ in $(seq 1 6000); do [[ -S "$vision_endpoint" ]] && break; sleep .01; done
PYTHONPATH=python "$python_bin" tools/bench/data_flow/probe_tiered_feature_recovery.py \
  --endpoint "$vision_endpoint" --image "$image" --output "$run_dir/result.json" \
  | tee "$run_dir/validation.log"
"$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" \
  --op shutdown >"$run_dir/vision-shutdown.json"
wait "$vision_pid"; vision_pid=
"$build/demo/pbe_data_service" stats "$data_endpoint" | tee "$run_dir/data-after.log"
grep -q 'leases=0' "$run_dir/data-after.log"
cleanup
trap - EXIT
echo "PBE_TIERED_FEATURE_RECOVERY_OK evidence=$run_dir"
