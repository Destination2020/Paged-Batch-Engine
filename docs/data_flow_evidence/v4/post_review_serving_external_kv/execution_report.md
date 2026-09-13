# M2/M3 serving-path external KV acceptance

Date: 2026-09-13

`serving_qwen` now accepts `--data-service-endpoint`. Before model
initialization it obtains the typed CUDA pool descriptor, reserves the complete
slot set from the data-service owner, imports the CUDA IPC allocation, and
binds the resulting `ExternalKVPoolBinding` to every Qwen layer. The normal
Scheduler → MixedBatchBuilder → Qwen prefill/decode attention path therefore
uses the service-owned allocation; this is not the standalone IPC probe or the
specialized multi-role forward loop.

On H20 GPU 0, two requests ran together through one 71-token prefill step and
15 two-row decode steps. Both completed, producing 32 decode tokens. While the
model was live the service reported one active grant and 0/64 free slots. The
teardown synchronizes CUDA, destroys request page tables and model tensor views,
closes the importer, then returns the full lease token. The service subsequently
reported 64/64 free slots, zero grants, and zero data leases.

The separate multi-role run placed the pool owner and Prefill on GPU 0 and two
concurrent Decode consumers on GPU 1. Each consumer copied the 37,748,736-byte
pool once through the measured `cross_gpu_p2p_replica` path (432.271 ms and
407.309 ms), then executed real decode attention and a 589,824-byte partial-tail
COW. Both 16-token outputs exactly matched the Prefill oracle. After consumers
and the retained prefix exited in order, all 64 service slots and metadata
leases were returned.

Raw evidence is in this directory (`serving.log`, pool/data stats and
`regression.log`) and in sibling `post_review_cross_gpu_kv/`. Reproduction
entrypoints are `tools/launch/serving_external_kv.sh` and
`tools/launch/multi_role_vlm_ipc.sh` with `PBE_PREFILL_DEVICE=0` and
`PBE_DECODE_DEVICE=1`.

The relinked standard test binary passed 23/23 targeted contract regressions.
An earlier invocation used a stale test executable against the rebuilt shared
library; its ABI/layout failures are invalid and are not counted as a source
regression.
