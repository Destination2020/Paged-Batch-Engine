# M5 post-review execution report

H20 real-model acceptance used one persistent Python Vision PID and one persistent C++ PBE language PID. The language role acquired the complete 64-slot Agent-owned CUDA IPC KV pool once, loaded Qwen2.5-VL-3B once, and served 25 typed online batches through Scheduler, MixedBatchBuilder, real Prefill/Decode attention, sampling, and request cleanup.

Twenty fixed image/text cases were each executed in a four-request variable-length mixed batch and replayed as a single request in the same PID. Five steps contained Prefill and Decode rows together. Eighteen sequences matched exactly. Across all 160 generated tokens, 153 matched; the two non-exact cases each have one first greedy divergence followed by autoregressive cascade. This is recorded under the M1 frozen BF16 tie policy rather than reported as exact equality. The initial failed directory `post_review_persistent_vlm_20` exposed a dangling TensorComponent pointer (15/20 exact); the fix stores validated offsets and lengths by value, and the failed evidence remains intact.

The separate `post_review_persistent_vlm_online` run additionally cancelled one request at Join and one after three real Decode tokens. Both roles remained resident, mixed Prefill/Decode occurred, and shutdown returned 64/64 slots, zero grants and zero leases.
