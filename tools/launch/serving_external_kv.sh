#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 BUILD MODEL_BIN TOKENIZER MODEL_DIR" >&2
  exit 2
fi

build=$1
model_bin=$2
tokenizer=$3
model_dir=$4
device=${PBE_DEVICE:-0}
blocks=${PBE_KV_BLOCKS:-64}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-serving-external-kv-$$}
endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-serving-external-kv-$$.sock}
mkdir -p "$run_dir"

read -r layers kv_heads head_size < <(
  python3 - "$model_dir/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
text = config.get("text_config", config)
print(text["num_hidden_layers"], text["num_key_value_heads"],
      text["hidden_size"] // text["num_attention_heads"])
PY
)

service="$build/demo/pbe_data_service"
serving="$build/demo/serving_qwen"

"$service" serve-gpu "$endpoint" 2147483648 "$device" "$layers" "$blocks" \
  16 "$kv_heads" "$head_size" 4 512 >"$run_dir/service.log" 2>&1 &
service_pid=$!

cleanup() {
  "$service" shutdown "$endpoint" >/dev/null 2>&1 || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 400); do
  [[ -S "$endpoint" ]] && break
  sleep .01
done
[[ -S "$endpoint" ]]

"$serving" "$model_bin" "$tokenizer" \
  "What is paged attention?" \
  "Give one benefit of continuous batching." \
  --device-id="$device" --kv-cache-blocks-per-layer="$blocks" \
  --max-new-tokens=16 --max-batched-tokens=128 --prefill-chunk-cap=64 \
  --radix-cache=off --warmup-rounds=0 --step-trace=1 \
  --data-service-endpoint="$endpoint" >"$run_dir/serving.log" 2>&1 &
serving_pid=$!

: >"$run_dir/ipc-during-serving.log"
for _ in $(seq 1 3000); do
  stats=$("$service" ipc-stats "$endpoint" 2>/dev/null || true)
  if [[ "$stats" == *"free_slots=0 active_grants=1"* ]]; then
    printf '%s\n' "$stats" >"$run_dir/ipc-during-serving.log"
    break
  fi
  if ! kill -0 "$serving_pid" 2>/dev/null; then
    break
  fi
  sleep .01
done

wait "$serving_pid"
cat "$run_dir/serving.log"
[[ -s "$run_dir/ipc-during-serving.log" ]]
grep -q 'PBE_SERVING_EXTERNAL_KV_BOUND' "$run_dir/serving.log"
grep -q 'PBE_SERVING_EXTERNAL_KV_RELEASED' "$run_dir/serving.log"
grep -q 'completed_requests=2 failed_requests=0' "$run_dir/serving.log"

"$service" ipc-stats "$endpoint" | tee "$run_dir/ipc-after-serving.log"
grep -q "free_slots=$blocks active_grants=0" "$run_dir/ipc-after-serving.log"
"$service" stats "$endpoint" | tee "$run_dir/data-after-serving.log"
grep -q 'leases=0' "$run_dir/data-after-serving.log"

echo "PBE_SERVING_EXTERNAL_KV_OK evidence=$run_dir"
