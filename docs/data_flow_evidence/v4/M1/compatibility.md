# M1 Qwen2.5-VL compatibility record

## Frozen candidate

- Repository: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Architecture: `Qwen2_5_VLForConditionalGeneration`
- Primary numeric path: BF16 weights, Transformers SDPA, greedy decoding
- License: the checkpoint contains the Qwen Research License Agreement. It
  permits non-commercial research/evaluation and requires a separate license
  for commercial use. The full unmodified text remains in the checkpoint's
  `LICENSE` file.

Per-file SHA-256 values and the local dependency versions are written to
`model_manifest.json` after the pinned snapshot has completed.

## Language-core mapping

| Checkpoint property | Frozen value | PBE representation |
| --- | ---: | --- |
| vocabulary | 151,936 | signed legacy vocabulary field; positive means tied output |
| hidden / intermediate | 2,048 / 11,008 | `ModelConfig` dimensions |
| layers / Q heads / KV heads | 36 / 16 / 2 | `ModelConfig` dimensions |
| QKV bias | enabled | serialized directly after each Q/K/V matrix |
| RMSNorm epsilon | 1e-6 | Qwen build constant, checked by exporter |
| RoPE theta | 1,000,000 | PBE runtime sin/cos generation and export sidecar |
| mRoPE sections | 16, 24, 24 | temporal, height, width CUDA dispatch |
| embedding / lm_head | tied | classifier points at embedding weight |
| checkpoint dtype | BF16 | `DTYP=4` compatible reader header |

`tools/models/export_qwen25_vl.py` exports only the language core. The output
sidecar uses schema `pbe.qwen25_vl.language_core.bf16.v1`, lists every mapped
source key, and states that visual weights are excluded. Existing Qwen2 text
containers and their reader remain valid.

## Input ABI

`MultimodalSequencePlan` version 1 fixes these independent values:

- expanded token IDs, including one placeholder row per merged visual feature;
- sorted media token spans and references to immutable feature bundles;
- the processor grid `(T,H,W)` for each media span;
- axis-major temporal, height, and width position IDs;
- decode `next_token_offset` and `rope_delta` as separate fields.

Tensor bundle schema version 1 carries named components with explicit dtype,
shape, byte offset, byte length, and checksum. Validation rejects invalid span
coverage, feature lengths, grids, references, and overlapping byte ranges
before a copy is started.

## Implemented PBE adapter surface

`MixedBatchMetadata` accepts an optional `[tokens, hidden]` embedding override
and optional axis-major `[3,tokens]` mRoPE positions. The Qwen CUDA path copies
the supplied embeddings into its owned workspace and dispatches mRoPE only
when the three-axis tensor is present; ordinary text requests retain the prior
embedding lookup and scalar RoPE path.

The mRoPE CUDA unit test independently calculates all three section outputs.
The post-change Qwen build regression reports 185 passed and 11 explicitly
gated skips in `logs/qwen2_cuda_regression.log`. Three legacy `test_load`
cases were excluded because they require an absent `./tmp/test.bin` fixture;
their failure is recorded in `logs/qwen2_cuda_full.log` and is unrelated to
the model adapter.

## Numerical probe matrix

The reference runner freezes these cases from the repository's `image.png`,
using deterministic Lanczos resize before the official slow processor:

| Case | Inputs |
| --- | --- |
| `text` | text only |
| `one_image_224` | one 224×224 image |
| `one_image_wide` | one 280×168 image |
| `two_images` | the two resized variants in one request |
| `mixed_batch` | two requests with different visual lengths and left padding |

Each case records processor tensors, merged visual features, three-axis
positions, embeddings, selected hidden states, final logits top-10, a 32-token
greedy result, and a repeated greedy result. Raw BF16/int32/FP32 files are
emitted beside compressed NPZ fixtures for the C++ test.

## Frozen numerical thresholds and result

The PBE test uses BF16 end to end. Thresholds were frozen after repeated runs:
per-step logits max absolute error `<0.75` and mean absolute error `<0.20`;
initial/prefill logits max `<0.75` and mean `<0.13`; final hidden max `<1.5`
and mean `<0.13`; selected KV max `<1.25` and mean `<0.08`. Relative mean
error is mean absolute error divided by the reference mean absolute value.

The final evidence run passed both real-model tests. Its maxima were:

| Comparison | max abs | mean abs | relative mean |
| --- | ---: | ---: | ---: |
| 32 teacher-forced decode steps | 0.375 | 0.091682 | 0.033542 |
| initial logits | 0.3125 | 0.060417 | 0.007599 |
| final hidden | 0.5 | 0.044796 | 0.021286 |
| KV K/V at layers 0, 18, 35 | 0.49707 | 0.028744 | 0.033410 |
| text/image/two-image/mixed prefill | 0.25 | 0.048421 | 0.008102 |

The actual PBE greedy choice matched the official processor through 24 steps.
At step 25 it chose token 11589 rather than 990; the frozen official
post-processor reports a top-2 margin of exactly zero at that step. The test
records the divergence, then teacher-forces the reference history so all 32
incremental KV/position steps are independently compared. This is treated as
a BF16 tie rather than a state divergence.

## Acceptance state

M1 is accepted. The pinned official checkpoint and processor were hashed; the
reference path repeated exactly; the versioned export loaded in PBE; real PBE
language execution passed pure text, one image, two images, two resolutions,
and a mixed-length batch; selected hidden/KV and all 32 decode steps passed the
frozen numerical thresholds. The vision tower remains a reference-process
probe here and becomes a separate framework role in M4 as planned.
