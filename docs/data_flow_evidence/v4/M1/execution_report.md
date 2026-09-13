# M1 execution report

M1 is accepted for the pinned Qwen2.5-VL-3B checkpoint. The official processor
and vision tower produced repeatable fixtures for five required input shapes.
The versioned language-core exporter mapped all 434 required tensors with no
missing or unexpected keys and retained the legacy Qwen reader contract.

PBE now accepts controlled embedding replacement and Qwen three-axis mRoPE in
its real mixed/prefill and decode CUDA paths. It also applies repetition
penalties to the complete prompt plus generated history, matching the official
generation contract. Real PBE execution passed logits, final hidden, selected
paged KV layers, 32 incremental decode steps, and the required input/batch
matrix. Exact commands and results are in `checks.json` and `logs/`.

The one observed free-greedy token difference is at an exact top-2 tie and is
recorded rather than hidden. Subsequent comparisons use the same reference
history so numerical agreement for all remaining decode steps remains
measurable.
