#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 && $# != 6 ]]; then echo "usage: $0 BUILD MODEL_BIN TOKENIZER MODEL_DIR IMAGE [ORACLE_FIXTURE_DIR]" >&2;exit 2;fi
build=$1; model_bin=$2; tokenizer=$3; model_dir=$4; image=$5; fixtures=${6:-}
run_dir=${PBE_RUN_DIR:-/tmp/pbe-vlm-$$};mkdir -p "$run_dir";endpoint=/tmp/pbe-vlm-$$.sock
service="$build/demo/pbe_data_service";role="$build/demo/pbe_multi_role_vlm"
"$service" serve "$endpoint" 2147483648 512 >"$run_dir/service.log" 2>&1 & service_pid=$!
trap '"$service" shutdown "$endpoint" >/dev/null 2>&1 || true;wait "$service_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 200);do [[ -S "$endpoint" ]]&&break;sleep .01;done;[[ -S "$endpoint" ]]
vision_args=(--endpoint "$endpoint" --model "$model_dir" --image "$image" --size 224 --device cuda:0)
if [[ -n "$fixtures" ]]; then vision_args+=(--reference "$fixtures/one_image_224.npz");fi
PYTHONPATH=python .venv/bin/python python/pbe_roles/vision/worker.py "${vision_args[@]}" | tee "$run_dir/vision.log"
vision=$(sed -n 's/.* content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/vision.log")
CUDA_VISIBLE_DEVICES=0 "$role" prefill "$model_bin" "$tokenizer" "$endpoint" 0 - "$vision" qwen25-vl-3b-kv-v2 16 | tee "$run_dir/prefill.log"
kv=$(sed -n 's/.* kv_content=\([0-9a-f]\{64\}\).*/\1/p' "$run_dir/prefill.log")
CUDA_VISIBLE_DEVICES=0 "$role" decode "$model_bin" "$tokenizer" "$endpoint" 0 - "$kv" qwen25-vl-3b-kv-v2 16 | tee "$run_dir/decode-a.log"
CUDA_VISIBLE_DEVICES=0,1 "$role" decode "$model_bin" "$tokenizer" "$endpoint" 1 - "$kv" qwen25-vl-3b-kv-v2 16 | tee "$run_dir/decode-b.log"
oracle=$(sed -n 's/.* oracle_tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/prefill.log");a=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-a.log");b=$(sed -n 's/.* tokens=\([^ ]*\) text=.*/\1/p' "$run_dir/decode-b.log")
if [[ -n "$fixtures" ]]; then reference=$(.venv/bin/python - "$fixtures/reference.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));case=next(x for x in p['cases'] if x['case']=='one_image_224');print(','.join(map(str,case['generated_token_ids'][0][:16]))+',')
PY
);[[ "$a" == "$reference" ]];fi
[[ "$oracle" == "$a" && "$a" == "$b" ]]
"$service" stats "$endpoint" | tee "$run_dir/stats.log"
echo "PBE_MULTI_ROLE_VLM_OK evidence=$run_dir"
