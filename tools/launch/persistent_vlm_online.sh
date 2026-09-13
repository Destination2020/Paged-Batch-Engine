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
run_dir=${PBE_RUN_DIR:-/tmp/pbe-persistent-vlm-online-$$}
data_endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-persistent-vlm-data-$$.sock}
vision_endpoint=${PBE_VISION_ENDPOINT:-/tmp/pbe-persistent-vlm-vision-$$.sock}
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
python_bin=${PBE_PYTHON:-.venv/bin/python}
weight_mode=${PBE_WEIGHT_MODE:-private}
weight_layout=${PBE_WEIGHT_LAYOUT:-pbe-qwen2-bf16-v1}
weight_args=()
coordinator_weight_args=()
if [[ "$weight_mode" == shared ]]; then
  model_sha256=$(sha256sum "$model_bin" | awk '{print $1}')
  weight_args=("$model_bin" "$model_sha256" "$weight_layout" "${PBE_WEIGHT_STAGING_BYTES:-67108864}")
  coordinator_weight_args=(--model-sha256 "$model_sha256" --weight-layout "$weight_layout")
  service_command=("$service" serve-gpu-weights "$data_endpoint" 2147483648 "$device" "$layers" "$blocks" 16 "$kv_heads" "$head_size" 4 "${weight_args[@]}" 2048)
elif [[ "$weight_mode" == private ]]; then
  model_sha256=
  service_command=("$service" serve-gpu "$data_endpoint" 2147483648 "$device" "$layers" "$blocks" 16 "$kv_heads" "$head_size" 4 2048)
else
  echo "unsupported PBE_WEIGHT_MODE=$weight_mode" >&2
  exit 2
fi
"${service_command[@]}" >"$run_dir/data-service.log" 2>&1 &
service_pid=$!
vision_pid=

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

for _ in $(seq 1 400); do
  [[ -S "$data_endpoint" ]] && break
  sleep .01
done
[[ -S "$data_endpoint" ]]

vision_cache_args=()
if [[ "${PBE_DISABLE_FEATURE_CACHE:-0}" == 1 ]]; then
  vision_cache_args+=(--disable-feature-cache)
fi
if [[ "${PBE_DISABLE_SINGLEFLIGHT:-0}" == 1 ]]; then
  vision_cache_args+=(--disable-singleflight)
fi
PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
  --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
  --device "cuda:$device" --batch-window-ms 80 --max-batch 8 "${vision_cache_args[@]}" \
  >"$run_dir/vision-role.log" 2>&1 &
vision_pid=$!
for _ in $(seq 1 6000); do
  [[ -S "$vision_endpoint" ]] && break
  if ! kill -0 "$vision_pid" 2>/dev/null; then
    cat "$run_dir/vision-role.log" >&2
    exit 1
  fi
  sleep .01
done
[[ -S "$vision_endpoint" ]]

spec_path=${PBE_SPEC:-$run_dir/spec.json}
generated_spec=0
if [[ -z "${PBE_SPEC:-}" ]]; then
generated_spec=1
python3 - "$spec_path" "$image" <<'PY'
import json, sys
output, image = sys.argv[1:]
requests = []
for index, size in enumerate((224, 252, 280, 308)):
    requests.append({
        "request_id": f"mixed-{index}", "generation": 1,
        "parts": [{"type": "image", "path": image, "size": size},
                  {"type": "text", "text": f"Describe the image, variant {index}."}],
        "max_new_tokens": 8,
    })
requests.append({
    "request_id": "decode-cancel", "generation": 1,
    "parts": [{"type": "image", "path": image, "size": 224},
              {"type": "text", "text": "Cancel this generation during decode."}],
    "max_new_tokens": 16, "cancel_after_tokens": 3,
})
requests.append({
    "request_id": "join-cancel", "generation": 1, "cancel_stage": "join",
    "parts": [{"type": "image", "path": image, "size": 252},
              {"type": "text", "text": "Cancel this request at join."}],
    "max_new_tokens": 8,
})
json.dump({"rounds": [requests]}, open(output, "w"))
PY
fi

PYTHONPATH=python "$python_bin" python/pbe_roles/coordinator.py \
  --vision-endpoint "$vision_endpoint" \
  --language-binary "$build/demo/pbe_vlm_language_role" \
  --model-bin "$model_bin" --tokenizer "$tokenizer" \
  --data-endpoint "$data_endpoint" --device "$device" \
  --spec "$spec_path" --trace "$run_dir/trace.jsonl" \
  --bundle-capacity "${PBE_BUNDLE_CAPACITY:-67108864}" \
  --staging-capacity "${PBE_STAGING_CAPACITY:-33554432}" \
  --weight-mode "$weight_mode" \
  "${coordinator_weight_args[@]}" \
  ${PBE_DISABLE_RADIX_CACHE:+--disable-radix-cache} \
  >"$run_dir/result.json" 2>"$run_dir/language-role.log"

"$service" ipc-stats "$data_endpoint" | tee "$run_dir/ipc-after-language.log"
"$service" stats "$data_endpoint" | tee "$run_dir/data-after-language.log"
if [[ "$weight_mode" == shared ]]; then
  "$service" weight-stats "$data_endpoint" | tee "$run_dir/weight-after-language.log"
  grep -q 'active_leases=0' "$run_dir/weight-after-language.log"
fi
grep -q "free_slots=$blocks active_grants=0" "$run_dir/ipc-after-language.log"
grep -q 'leases=0' "$run_dir/data-after-language.log"

if [[ "$generated_spec" == 1 ]]; then
python3 - "$run_dir/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
round_result = result["rounds"][0]
assert result["ok"] and round_result["budget_invariant"]
assert round_result["admitted"] == 5 and round_result["rejected"] == 0
assert round_result["mixed_prefill_decode_steps"] > 0
assert round_result["stages"]["join-cancel"] == "cancelled"
cancel = next(x for x in round_result["outputs"] if x["request_id"] == "decode-cancel")
assert cancel["failed"] and cancel["finish_reason"] == "cancelled_by_coordinator"
assert result["status"]["worker_pid"] == round_result["worker_pid"]
assert result["status"]["budget_invariant"]
print(json.dumps({"event": "validated", "persistent_language_pid": round_result["worker_pid"],
                  "mixed_prefill_decode_steps": round_result["mixed_prefill_decode_steps"],
                  "join_cancelled": True, "decode_cancelled": True,
                  "unified_budget_invariant": True}, sort_keys=True))
PY
fi

"$python_bin" python/pbe_roles/vision/client.py --listen "$vision_endpoint" \
  --op shutdown >"$run_dir/vision-shutdown.json"
wait "$vision_pid"
vision_pid=
echo "PBE_PERSISTENT_VLM_ONLINE_OK evidence=$run_dir"
