# M9 post-review execution report

The final matrix consolidates only real main-path measurements. Persistent multimodal cache off/on was randomized with seed 20260913 and repeated five times per mode inside the same resident Vision/language deployment. Image, text, model, generation length and budgets were fixed; cache-off uses request-private feature identities. Output tokens were identical across all ten measured requests.

Cache off/on medians were 2491.70/2483.70 ms multimodal TTFT, 46.17/46.23 ms ITL, and 3193.22/3178.80 ms end-to-end latency. Cache-off Vision forward was 37.27 ms median and cache-hit forward was zero. Processor/RPC work dominates the encode interval, so saved encoder compute is not described as equal end-to-end speedup. The previously completed five-repeat local/external serving control remains in the matrix: the external pool had +7.93% TTFT, +3.36% latency and -3.26% throughput under low reuse.

The matrix also references real four-branch COW, same/cross-GPU transport, online rejection/recovery, and the complete fault suite. Equal-budget merged full-VLM is not implemented in V4 and remains an explicit null field; no speedup is inferred from a different model or execution path. The delivery guide provides the one-command persistent demo, text fallback, protocol/failure rules, five-minute walkthrough and interview questions.

Final standard regression ran 209 applicable tests successfully with 11 explicitly gated skips. The unfiltered log also preserves three known `test_load.*` failures caused by the absent legacy `./tmp/test.bin` fixture, already documented in M1; the clean applicable command excludes only that suite. Python compilation, launcher syntax, JSON parsing and `git diff --check` passed.
