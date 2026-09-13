#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "usage: $0 BUILD MODEL_BIN TOKENIZER MODEL_DIR IMAGE" >&2
  exit 2
fi

build=$1
model_bin=$2
tokenizer=$3
model_dir=$4
image=$5
prefill_device=${PBE_PREFILL_DEVICE:-${PBE_DEVICE:-0}}
decode_device=${PBE_DECODE_DEVICE:-$prefill_device}
blocks=${PBE_KV_BLOCKS:-64}
block_size=${PBE_KV_BLOCK_SIZE:-16}
steps=${PBE_DECODE_STEPS:-16}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-vlm-ipc-$$}
endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-vlm-ipc-$$.sock}
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
pool_bytes=$((layers * 2 * blocks * block_size * kv_heads * head_size * 2))

service="$build/demo/pbe_data_service"
role="$build/demo/pbe_multi_role_vlm"
prefix_service=
prefix_consumer=
prefix_grant=

"$service" serve-gpu "$endpoint" 2147483648 "$prefill_device" "$layers" "$blocks" \
  "$block_size" "$kv_heads" "$head_size" 4 512 >"$run_dir/service.log" 2>&1 &
service_pid=$!

cleanup() {
  if [[ -n "$prefix_grant" ]]; then
    "$service" release-ipc "$endpoint" "$prefix_service" "$prefix_consumer" \
      "$prefix_grant" >/dev/null 2>&1 || true
  fi
  "$service" shutdown "$endpoint" >/dev/null 2>&1 || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 400); do
  [[ -S "$endpoint" ]] && break
  sleep .01
done
[[ -S "$endpoint" ]]

PYTHONPATH=python .venv/bin/python python/pbe_roles/vision/worker.py \
  --endpoint "$endpoint" --model "$model_dir" --image "$image" --size 224 \
  --device "cuda:$prefill_device" | tee "$run_dir/vision.log"
vision=$(sed -n 's/.* content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/vision.log")
[[ -n "$vision" ]]

"$role" prefill-ipc "$model_bin" "$tokenizer" \
  "$endpoint" "$prefill_device" - "$vision" qwen25-vl-3b-kv-v2 16 | tee "$run_dir/prefill.log"
kv=$(sed -n 's/.* kv_content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/prefill.log")
prefix_service=$(sed -n 's/.* pool_grant_service=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_consumer=$(sed -n 's/.* pool_grant_consumer=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_grant=$(sed -n 's/.* pool_grant=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
[[ -n "$kv" && -n "$prefix_service" && -n "$prefix_consumer" && -n "$prefix_grant" ]]

"$role" decode-ipc "$model_bin" "$tokenizer" \
  "$endpoint" "$decode_device" - "$kv" qwen25-vl-3b-kv-v2 "$steps" >"$run_dir/decode-a.log" 2>&1 &
decode_a_pid=$!
"$role" decode-ipc "$model_bin" "$tokenizer" \
  "$endpoint" "$decode_device" - "$kv" qwen25-vl-3b-kv-v2 "$steps" >"$run_dir/decode-b.log" 2>&1 &
decode_b_pid=$!

: >"$run_dir/ipc-during-decodes.log"
for _ in $(seq 1 1200); do
  stats=$("$service" ipc-stats "$endpoint" 2>/dev/null || true)
  if [[ "$stats" == *"active_grants=3"* ]]; then
    printf '%s\n' "$stats" >"$run_dir/ipc-during-decodes.log"
    break
  fi
  if ! kill -0 "$decode_a_pid" 2>/dev/null && \
     ! kill -0 "$decode_b_pid" 2>/dev/null; then
    break
  fi
  sleep .01
done
wait "$decode_a_pid"
wait "$decode_b_pid"
cat "$run_dir/decode-a.log"
cat "$run_dir/decode-b.log"
[[ -s "$run_dir/ipc-during-decodes.log" ]]

oracle=$(sed -n 's/.* oracle_tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/prefill.log")
decode_a=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-a.log")
decode_b=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-b.log")
[[ -n "$oracle" && "$oracle" == "$decode_a" && "$decode_a" == "$decode_b" ]]
grep -q 'cow_bytes=589824' "$run_dir/decode-a.log"
grep -q 'cow_bytes=589824' "$run_dir/decode-b.log"
if [[ "$decode_device" == "$prefill_device" ]]; then
  grep -q 'transport=same_gpu_ipc_map' "$run_dir/decode-a.log"
  grep -q 'replica_bytes=0' "$run_dir/decode-a.log"
else
  grep -q 'transport=cross_gpu_p2p_page_subset' "$run_dir/decode-a.log"
  grep -q 'transport=cross_gpu_p2p_page_subset' "$run_dir/decode-b.log"
  replica_a=$(sed -n 's/.* replica_bytes=\([0-9]*\).*/\1/p' "$run_dir/decode-a.log")
  replica_b=$(sed -n 's/.* replica_bytes=\([0-9]*\).*/\1/p' "$run_dir/decode-b.log")
  [[ "$replica_a" -gt 0 && "$replica_a" -lt "$pool_bytes" ]]
  [[ "$replica_b" -gt 0 && "$replica_b" -lt "$pool_bytes" ]]
fi

"$service" ipc-stats "$endpoint" | tee "$run_dir/ipc-before-prefix-release.log"
grep -q 'active_grants=1' "$run_dir/ipc-before-prefix-release.log"
"$service" release-ipc "$endpoint" "$prefix_service" "$prefix_consumer" \
  "$prefix_grant" | tee "$run_dir/prefix-release.log"
prefix_grant=
"$service" ipc-stats "$endpoint" | tee "$run_dir/ipc-after-prefix-release.log"
grep -q "free_slots=$blocks active_grants=0" "$run_dir/ipc-after-prefix-release.log"
"$service" stats "$endpoint" | tee "$run_dir/stats.log"
grep -q 'leases=0' "$run_dir/stats.log"

echo "PBE_MULTI_ROLE_VLM_IPC_OK evidence=$run_dir"
