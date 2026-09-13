#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 && $# != 6 ]]; then
  echo "usage: $0 BUILD MODEL_DIR IMAGE_A IMAGE_B [MODEL_BIN TOKENIZER]" >&2
  exit 2
fi

build=$1
model_dir=$2
image_a=$3
image_b=$4
model_bin=${5:-}
tokenizer=${6:-}
device=${PBE_DEVICE:-0}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-persistent-vision-$$}
data_endpoint=${PBE_DATA_ENDPOINT:-/tmp/pbe-persistent-vision-data-$$.sock}
vision_endpoint=${PBE_VISION_ENDPOINT:-/tmp/pbe-persistent-vision-role-$$.sock}
mkdir -p "$run_dir"

service="$build/demo/pbe_data_service"
python_bin=${PBE_PYTHON:-.venv/bin/python}
client=python/pbe_roles/vision/client.py

"$service" serve "$data_endpoint" 2147483648 512 >"$run_dir/data-service.log" 2>&1 &
service_pid=$!

cleanup() {
  "$python_bin" "$client" --listen "$vision_endpoint" --op shutdown \
    >/dev/null 2>&1 || true
  "$service" shutdown "$data_endpoint" >/dev/null 2>&1 || true
  wait "$vision_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
vision_pid=
trap cleanup EXIT

for _ in $(seq 1 400); do
  [[ -S "$data_endpoint" ]] && break
  sleep .01
done
[[ -S "$data_endpoint" ]]

PYTHONPATH=python "$python_bin" python/pbe_roles/vision/persistent_worker.py \
  --listen "$vision_endpoint" --endpoint "$data_endpoint" --model "$model_dir" \
  --device "cuda:$device" --batch-window-ms 100 --max-batch 8 \
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

"$python_bin" "$client" --listen "$vision_endpoint" --request-id req-a \
  --image "$image_a" --size 252 --text "Describe image A." \
  >"$run_dir/request-a.json" &
request_a_pid=$!
"$python_bin" "$client" --listen "$vision_endpoint" --request-id req-b \
  --image "$image_b" --size 308 --text "Describe image B." \
  >"$run_dir/request-b.json" &
request_b_pid=$!
wait "$request_a_pid"
wait "$request_b_pid"

if [[ -n "$model_bin" ]]; then
  vision_content=$(python3 -c \
    'import json,sys;print(json.load(open(sys.argv[1]))["content"])' \
    "$run_dir/request-a.json")
  "$build/demo/pbe_multi_role_vlm" prefill "$model_bin" "$tokenizer" \
    "$data_endpoint" "$device" - "$vision_content" persistent-vlm-kv-v1 16 \
    >"$run_dir/prefill.log"
  kv_content=$(sed -n 's/.* kv_content=\([0-9a-f]\{64\}\).*/\1/p' \
    "$run_dir/prefill.log")
  "$build/demo/pbe_multi_role_vlm" decode "$model_bin" "$tokenizer" \
    "$data_endpoint" "$device" - "$kv_content" persistent-vlm-kv-v1 16 \
    >"$run_dir/decode.log"
  oracle=$(sed -n 's/.* oracle_tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/prefill.log")
  decoded=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode.log")
  [[ -n "$oracle" && "$oracle" == "$decoded" ]]
fi

"$python_bin" "$client" --listen "$vision_endpoint" --request-id req-a-followup \
  --image "$image_a" --size 252 --text "Count the main subjects in image A." \
  >"$run_dir/request-cache-hit.json"

"$python_bin" "$client" --listen "$vision_endpoint" --request-id req-cancel \
  --image "$image_b" --size 336 --text "This request will be cancelled." \
  >"$run_dir/request-cancelled.json" 2>&1 &
cancelled_pid=$!
sleep .02
"$python_bin" "$client" --listen "$vision_endpoint" --op cancel \
  --request-id req-cancel >"$run_dir/cancel-response.json"
if wait "$cancelled_pid"; then
  echo "cancelled request unexpectedly succeeded" >&2
  exit 1
fi

"$python_bin" "$client" --listen "$vision_endpoint" --op shutdown \
  >"$run_dir/shutdown.json"
wait "$vision_pid"
vision_pid=

python3 - "$run_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
a = json.loads((root / "request-a.json").read_text())
b = json.loads((root / "request-b.json").read_text())
hit = json.loads((root / "request-cache-hit.json").read_text())
cancel = json.loads((root / "request-cancelled.json").read_text())
assert a["ok"] and b["ok"]
assert a["worker_pid"] == b["worker_pid"] == hit["worker_pid"]
assert a["batch_size"] == b["batch_size"] == 2
assert a["forward_batch_size"] == b["forward_batch_size"] == 2
assert a["feature_rows"] != b["feature_rows"]
assert hit["ok"] and hit["feature_cache_hit"]
assert hit["feature_content"] == a["feature_content"]
assert hit["content"] != a["content"]
assert hit["forward_batch_size"] == 0
assert not cancel["ok"] and cancel["error"] == "cancelled"
print(json.dumps({"event": "validated", "worker_pid": a["worker_pid"],
                  "mixed_rows": [a["feature_rows"], b["feature_rows"]],
                  "feature_reused_across_questions": True,
                  "request_bundle_identity_separated": True,
                  "cancelled_before_forward": True}, sort_keys=True))
PY

"$service" stats "$data_endpoint" | tee "$run_dir/data-after-requests.log"
grep -q 'leases=0' "$run_dir/data-after-requests.log"
echo "PBE_PERSISTENT_VISION_OK evidence=$run_dir"
