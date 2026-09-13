#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 7 ]]; then echo "usage: $0 BUILD MODEL_BIN TOKENIZER MODEL_DIR IMAGE RUN_DIR DEVICE" >&2; exit 2; fi
build=$1; model_bin=$2; tokenizer=$3; model_dir=$4; image=$5; run_dir=$6; device=$7
python_bin=${PBE_PYTHON:-.venv/bin/python}
data_endpoint=/tmp/pbe-persistent-deployment-data-$$.sock
vision_endpoint=/tmp/pbe-persistent-deployment-vision-$$.sock
mkdir -p "$run_dir"
read -r layers kv_heads head_size < <("$python_bin" - "$model_dir/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));t=c.get("text_config",c)
print(t["num_hidden_layers"],t["num_key_value_heads"],t["hidden_size"]//t["num_attention_heads"])
PY
)
service="$build/demo/pbe_data_service"
"$service" serve-gpu "$data_endpoint" 2147483648 "$device" "$layers" 128 16 "$kv_heads" "$head_size" 4 4096 >"$run_dir/data.log" 2>&1 & data_pid=$!
vision_pid=
cleanup(){
  [[ -z "$vision_pid" ]] || "$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" --op shutdown >/dev/null 2>&1 || true
  [[ -z "$vision_pid" ]] || wait "$vision_pid" 2>/dev/null || true
  "$service" shutdown "$data_endpoint" >/dev/null 2>&1 || true
  wait "$data_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 1000); do [[ -S "$data_endpoint" ]] && break; sleep .01; done
PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" --device "cuda:$device" >"$run_dir/vision.log" 2>&1 & vision_pid=$!
for _ in $(seq 1 6000); do [[ -S "$vision_endpoint" ]] && break; kill -0 "$vision_pid" 2>/dev/null || exit 1; sleep .01; done
PYTHONPATH=python "$python_bin" tools/bench/data_flow/run_persistent_deployment_ab.py --vision-endpoint "$vision_endpoint" --language-binary "$build/demo/pbe_vlm_language_role" --model-bin "$model_bin" --tokenizer "$tokenizer" --data-endpoint "$data_endpoint" --image "$image" --device "$device" --output "$run_dir/results.json" | tee "$run_dir/validation.log"
"$service" ipc-stats "$data_endpoint" | tee "$run_dir/ipc-after.log"
"$service" stats "$data_endpoint" | tee "$run_dir/data-after.log"
grep -q 'active_grants=0' "$run_dir/ipc-after.log"
grep -q 'leases=0' "$run_dir/data-after.log"
