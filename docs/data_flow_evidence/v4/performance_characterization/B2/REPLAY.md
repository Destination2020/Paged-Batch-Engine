# B2 exact replay

Run from the repository root with GPU 0 idle:

```bash
PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_b2_cache_matrix.py \
  --build build-v3 \
  --model-bin /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct.pbe-bf16-v1.bin \
  --tokenizer /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct/tokenizer.json \
  --model-dir /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct \
  --image /tmp/pbe-e1-different.png \
  --device 0 --repeats 5 \
  --output-dir docs/data_flow_evidence/v4/performance_characterization/B2
```

The runner fixes random seed `20260913`, deterministic Vision execution and
the cuBLAS workspace configuration. It records the 40-entry order and every
launcher's environment in `commands.json`. Replaying into the same directory
overwrites aggregate files and same-named trial artifacts; use a different
`--output-dir` when preservation is required.
