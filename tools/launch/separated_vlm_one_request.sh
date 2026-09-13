#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 6 ]]; then echo "usage: $0 BUILD MODEL TOKENIZER MODEL_DIR IMAGE RUN_DIR" >&2; exit 2; fi
build=$1; model=$2; tokenizer=$3; model_dir=$4; image=$5; run_dir=$6
device=${PBE_DEVICE:-0}; blocks=64; endpoint="/tmp/pbe-separated-one-$$.sock"; mkdir -p "$run_dir"
read -r layers kv_heads head_size < <(python3 - "$model_dir/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));t=c.get('text_config',c)
print(t['num_hidden_layers'],t['num_key_value_heads'],t['hidden_size']//t['num_attention_heads'])
PY
)
service="$build/demo/pbe_data_service"; role="$build/demo/pbe_multi_role_vlm"
prefix_service=; prefix_consumer=; prefix_grant=
"$service" serve-gpu "$endpoint" 2147483648 "$device" "$layers" "$blocks" 16 \
  "$kv_heads" "$head_size" 4 512 >"$run_dir/service.log" 2>&1 & service_pid=$!
cleanup(){
  if [[ -n "$prefix_grant" ]]; then "$service" release-ipc "$endpoint" "$prefix_service" "$prefix_consumer" "$prefix_grant" >/dev/null 2>&1 || true; fi
  "$service" shutdown "$endpoint" >/dev/null 2>&1 || true; wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 400); do [[ -S "$endpoint" ]] && break; sleep .01; done
PYTHONPATH=python .venv/bin/python python/pbe_roles/vision/worker.py --endpoint "$endpoint" \
  --model "$model_dir" --image "$image" --size 224 --device "cuda:$device" >"$run_dir/vision.log"
vision=$(sed -n 's/.* content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/vision.log")
"$role" prefill-ipc "$model" "$tokenizer" "$endpoint" "$device" - "$vision" \
  qwen25-vl-separated-one-v1 0 >"$run_dir/prefill.log"
kv=$(sed -n 's/.* kv_content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/prefill.log")
prefix_service=$(sed -n 's/.* pool_grant_service=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_consumer=$(sed -n 's/.* pool_grant_consumer=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_grant=$(sed -n 's/.* pool_grant=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
"$role" decode-ipc "$model" "$tokenizer" "$endpoint" "$device" - "$kv" \
  qwen25-vl-separated-one-v1 16 >"$run_dir/decode.log"
grep -q 'transport=same_gpu_ipc_map' "$run_dir/decode.log"
grep -q 'prefill_tokens_saved=' "$run_dir/decode.log"
"$service" release-ipc "$endpoint" "$prefix_service" "$prefix_consumer" "$prefix_grant" >/dev/null
prefix_grant=
"$service" ipc-stats "$endpoint" >"$run_dir/ipc-after.log"
grep -q 'free_slots=64 active_grants=0' "$run_dir/ipc-after.log"
echo "PBE_SEPARATED_ONE_REQUEST_OK evidence=$run_dir"
