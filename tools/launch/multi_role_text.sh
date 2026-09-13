#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 BUILD_DIR MODEL_BIN TOKENIZER_JSON [PROMPT]" >&2
  exit 2
fi
build_dir=$1
model_bin=$2
tokenizer_json=$3
prompt=${4:-Explain why cached prefixes reduce latency in one sentence.}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-multi-role-text-$$}
mkdir -p "$run_dir"
endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-multi-role-text-$$.sock}
service="$build_dir/demo/pbe_data_service"
role="$build_dir/demo/pbe_multi_role_text"
ipc="$build_dir/demo/pbe_ipc_pool_probe"
"$service" serve "$endpoint" 1073741824 4096 >"$run_dir/service.log" 2>&1 &
service_pid=$!
trap '"$service" shutdown "$endpoint" >/dev/null 2>&1 || true; wait "$service_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 200); do [[ -S "$endpoint" ]] && break; sleep 0.01; done
[[ -S "$endpoint" ]]
CUDA_VISIBLE_DEVICES=0 "$role" prefill "$model_bin" "$tokenizer_json" "$endpoint" qwen2-bf16-kv-v1 0 "$prompt" | tee "$run_dir/prefill.log"
content=$(sed -n 's/.* content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/prefill.log")
[[ -n "$content" ]]
CUDA_VISIBLE_DEVICES=0 "$role" decode "$model_bin" "$tokenizer_json" "$endpoint" qwen2-bf16-kv-v1 0 "$content" 16 | tee "$run_dir/decode-a.log"
CUDA_VISIBLE_DEVICES=0,1 "$role" decode "$model_bin" "$tokenizer_json" "$endpoint" qwen2-bf16-kv-v1 1 "$content" 16 | tee "$run_dir/decode-b.log"
oracle_tokens=$(sed -n 's/.* oracle_tokens=\([^ ]*\) oracle_text=.*/\1/p' "$run_dir/prefill.log")
decode_a_tokens=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-a.log")
decode_b_tokens=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-b.log")
[[ "$oracle_tokens" == "$decode_a_tokens" && "$oracle_tokens" == "$decode_b_tokens" ]]
descriptor="$run_dir/ipc.bin"
"$ipc" export "$descriptor" 0 67108864 165 >"$run_dir/ipc-export.log" 2>&1 &
ipc_pid=$!
"$ipc" import "$descriptor" 1 0 165 | tee "$run_dir/ipc-import.log"
wait "$ipc_pid"
"$service" stats "$endpoint" | tee "$run_dir/stats.log"
echo "PBE_MULTI_ROLE_TEXT_OK evidence=$run_dir"
