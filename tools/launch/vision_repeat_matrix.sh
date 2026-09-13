#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: $0 BUILD MODEL_DIR IMAGE RUN_DIR PYTHON" >&2
  exit 2
fi
build=$1
model_dir=$2
image=$3
run_dir=$4
python_bin=$5
mkdir -p "$run_dir"
data_endpoint="/tmp/pbe-repeat-matrix-data-$$.sock"
vision_endpoint="/tmp/pbe-repeat-matrix-vision-$$.sock"
"$build/demo/pbe_data_service" serve "$data_endpoint" 1073741824 1024 \
  >"$run_dir/data-service.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 400); do [[ -S "$data_endpoint" ]] && break; sleep .01; done
PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
  --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
  --device cuda:0 --batch-window-ms 10 --max-batch 8 \
  >"$run_dir/vision.log" 2>&1 &
vision_pid=$!
cleanup() {
  "$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" \
    --op shutdown >/dev/null 2>&1 || true
  wait "$vision_pid" 2>/dev/null || true
  "$build/demo/pbe_data_service" shutdown "$data_endpoint" >/dev/null 2>&1 || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 6000); do [[ -S "$vision_endpoint" ]] && break; sleep .01; done
PYTHONPATH=python "$python_bin" tools/bench/data_flow/probe_vision_repeat_matrix.py \
  --endpoint "$vision_endpoint" --image "$image" --output "$run_dir/result.json" \
  | tee "$run_dir/validation.log"
cleanup
trap - EXIT
echo "PBE_VISION_REPEAT_MATRIX_OK evidence=$run_dir"
