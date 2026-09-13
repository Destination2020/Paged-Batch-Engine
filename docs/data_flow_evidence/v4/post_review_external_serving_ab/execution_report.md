# Fair external-KV serving A/B (M9 partial)

Date: 2026-09-13

This experiment compares the normal process-owned KV allocation with the new
Agent-owned CUDA IPC allocation in the same `serving_qwen` scheduler path. Both
modes use the frozen Qwen2.5-VL-3B language checkpoint, GPU 0, the same two text
requests, 64 blocks per layer, a 128-token batch cap, a 64-token prefill cap and
16 decode tokens per request. The ten runs were shuffled with seed 20260913 and
contain five repetitions per mode. Every run completed two requests with zero
failures, 71 prefill tokens and 32 decode tokens.

Median serving-only results:

| mode | TTFT ms | ITL ms | latency ms | throughput token/s |
| --- | ---: | ---: | ---: | ---: |
| process-owned KV | 84.140 | 6.212 | 177.491 | 180.248 |
| Agent-owned IPC KV | 90.812 | 6.188 | 183.453 | 174.367 |

For this small low-reuse workload, external ownership changes median TTFT by
+7.93%, request latency by +3.36%, throughput by -3.26%, and ITL by -0.39%.
This is an overhead/control result, not a speedup claim. Process wall time also
contains model and IPC setup and is retained in `results.json`, but is not mixed
with serving-only TTFT/ITL. Every external run ended at 64/64 free slots and
zero active grants.

Raw `FINAL_SUMMARY` output for every run, service logs, randomized ordering and
all parsed fields are retained here. This closes only the five-repeat fair A/B
for the external-pool change. M9 remains open until the persistent multimodal
cache/share, deployment, lane and pressure-policy experiments run under their
full matched budgets.
