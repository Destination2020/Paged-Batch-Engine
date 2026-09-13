#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 6 ]]; then
  echo "usage: $0 BUILD MODEL_BIN TOKENIZER MODEL_DIR IMAGE RUN_DIR" >&2
  exit 2
fi
build=$1; model_bin=$2; tokenizer=$3; model_dir=$4; image=$5; run_dir=$6
device=${PBE_DEVICE:-0}; blocks=${PBE_KV_BLOCKS:-64}
python_bin=${PBE_PYTHON:-.venv/bin/python}
data_endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-placement-data-$$.sock}
vision_endpoint=${PBE_VISION_ENDPOINT:-/tmp/pbe-placement-vision-$$.sock}
mkdir -p "$run_dir"
read -r layers kv_heads head_size < <(
  "$python_bin" - "$model_dir/config.json" <<'PY'
import json, sys
c=json.load(open(sys.argv[1])); t=c.get("text_config",c)
print(t["num_hidden_layers"],t["num_key_value_heads"],t["hidden_size"]//t["num_attention_heads"])
PY
)
service="$build/demo/pbe_data_service"
"$service" serve-gpu "$data_endpoint" 2147483648 "$device" "$layers" "$blocks" \
  16 "$kv_heads" "$head_size" 4 2048 >"$run_dir/data-service.log" 2>&1 &
service_pid=$!; vision_pid=
cleanup() {
  if [[ -n "$vision_pid" ]]; then
    "$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" \
      --op shutdown >/dev/null 2>&1 || true
    wait "$vision_pid" 2>/dev/null || true
  fi
  "$service" shutdown "$data_endpoint" >/dev/null 2>&1 || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 400); do [[ -S "$data_endpoint" ]] && break; sleep .01; done
[[ -S "$data_endpoint" ]]
PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
  --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
  --device "cuda:$device" --batch-window-ms 20 --max-batch 8 \
  >"$run_dir/vision-role.log" 2>&1 &
vision_pid=$!
for _ in $(seq 1 6000); do
  [[ -S "$vision_endpoint" ]] && break
  kill -0 "$vision_pid" 2>/dev/null || { cat "$run_dir/vision-role.log" >&2; exit 1; }
  sleep .01
done
[[ -S "$vision_endpoint" ]]
"$python_bin" tools/bench/data_flow/make_placement_spec.py \
  --image "$image" --output "$run_dir/spec.json"
for policy in ${PBE_PLACEMENT_POLICIES:-fixed round_robin data_aware}; do
  for seed in 11 23 37 41 53; do
    output_dir="$run_dir/$policy/seed-$seed"
    mkdir -p "$output_dir"
    PYTHONPATH=python "$python_bin" python/pbe_roles/multi_worker_coordinator.py \
      --vision-endpoint "$vision_endpoint" --language-binary "$build/demo/pbe_vlm_language_role" \
      --model-bin "$model_bin" --tokenizer "$tokenizer" --data-endpoint "$data_endpoint" \
      --device "$device" --policy "$policy" --seed "$seed" --spec "$run_dir/spec.json" \
      --calibration docs/data_flow_evidence/v4/E2/calibration.json \
      --output "$output_dir/result.json" >"$output_dir/stdout.json" \
      2>"$output_dir/language.log"
  done
done
"$python_bin" tools/bench/data_flow/summarize_placement_runs.py \
  --runs "$run_dir" --output "$run_dir/results.json" | tee "$run_dir/validation.log"
"$service" ipc-stats "$data_endpoint" | tee "$run_dir/ipc-after.log"
"$service" stats "$data_endpoint" | tee "$run_dir/data-after.log"
grep -q "free_slots=$blocks active_grants=0" "$run_dir/ipc-after.log"
grep -q 'leases=0' "$run_dir/data-after.log"
echo "PBE_MULTI_WORKER_PLACEMENT_OK evidence=$run_dir"
