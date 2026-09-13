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
device=${PBE_DEVICE:-0}
blocks=${PBE_KV_BLOCKS:-64}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-four-branch-$$}
endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-four-branch-$$.sock}
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
role="$build/demo/pbe_multi_role_vlm"
prefix_service=
prefix_consumer=
prefix_grant=

"$service" serve-gpu "$endpoint" 2147483648 "$device" "$layers" "$blocks" \
  16 "$kv_heads" "$head_size" 4 512 >"$run_dir/service.log" 2>&1 &
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
for _ in $(seq 1 400); do [[ -S "$endpoint" ]] && break; sleep .01; done
[[ -S "$endpoint" ]]

PYTHONPATH=python .venv/bin/python python/pbe_roles/vision/worker.py \
  --endpoint "$endpoint" --model "$model_dir" --image "$image" --size 224 \
  --device "cuda:$device" >"$run_dir/vision.log"
vision=$(sed -n 's/.* content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/vision.log")

"$role" prefill-ipc "$model_bin" "$tokenizer" "$endpoint" "$device" - \
  "$vision" qwen25-vl-four-branch-v1 16 >"$run_dir/prefill.log"
kv=$(sed -n 's/.* kv_content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/prefill.log")
prefix_service=$(sed -n 's/.* pool_grant_service=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_consumer=$(sed -n 's/.* pool_grant_consumer=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")
prefix_grant=$(sed -n 's/.* pool_grant=\([0-9]*\).*/\1/p' "$run_dir/prefill.log")

"$role" decode-ipc "$model_bin" "$tokenizer" "$endpoint" "$device" - \
  "$kv" qwen25-vl-four-branch-v1 16 >"$run_dir/branch-a.log" 2>&1 &
pid_a=$!
"$role" decode-ipc "$model_bin" "$tokenizer" "$endpoint" "$device" - \
  "$kv" qwen25-vl-four-branch-v1 16 >"$run_dir/branch-b.log" 2>&1 &
pid_b=$!
"$role" decode-cancel-ipc "$model_bin" "$tokenizer" "$endpoint" "$device" - \
  "$kv" qwen25-vl-four-branch-v1 4 >"$run_dir/branch-c-cancel.log" 2>&1 &
pid_c=$!
"$role" decode-cancel-ipc "$model_bin" "$tokenizer" "$endpoint" "$device" - \
  "$kv" qwen25-vl-four-branch-v1 4 >"$run_dir/branch-d-cancel.log" 2>&1 &
pid_d=$!

: >"$run_dir/ipc-five-grants.log"
for _ in $(seq 1 3000); do
  stats=$("$service" ipc-stats "$endpoint" 2>/dev/null || true)
  if [[ "$stats" == *"active_grants=5"* ]]; then
    printf '%s\n' "$stats" >"$run_dir/ipc-five-grants.log"
    break
  fi
  sleep .01
done

wait "$pid_c"
wait "$pid_d"
wait "$pid_a"
wait "$pid_b"
[[ -s "$run_dir/ipc-five-grants.log" ]]
grep -q 'cancelled=1' "$run_dir/branch-c-cancel.log"
grep -q 'cancelled=1' "$run_dir/branch-d-cancel.log"
grep -q 'cow_bytes=589824' "$run_dir/branch-c-cancel.log"
grep -q 'cow_bytes=589824' "$run_dir/branch-d-cancel.log"

oracle=$(sed -n 's/.* oracle_tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/prefill.log")
tokens_a=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/branch-a.log")
tokens_b=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/branch-b.log")
[[ -n "$oracle" && "$oracle" == "$tokens_a" && "$tokens_a" == "$tokens_b" ]]
grep -q 'cow_bytes=589824' "$run_dir/branch-a.log"
grep -q 'cow_bytes=589824' "$run_dir/branch-b.log"

"$service" ipc-stats "$endpoint" | tee "$run_dir/ipc-before-prefix-release.log"
grep -q 'active_grants=1' "$run_dir/ipc-before-prefix-release.log"
"$service" release-ipc "$endpoint" "$prefix_service" "$prefix_consumer" \
  "$prefix_grant" >"$run_dir/prefix-release.log"
prefix_grant=
"$service" ipc-stats "$endpoint" | tee "$run_dir/ipc-after-prefix-release.log"
grep -q "free_slots=$blocks active_grants=0" "$run_dir/ipc-after-prefix-release.log"
"$service" stats "$endpoint" | tee "$run_dir/data-final.log"
grep -q 'leases=0' "$run_dir/data-final.log"
echo "PBE_FOUR_BRANCH_VLM_IPC_OK evidence=$run_dir"
