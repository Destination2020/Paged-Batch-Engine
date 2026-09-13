# M9 final execution report

M9 is accepted on 2026-09-13. `final_results.json` consolidates all requested real-model or real-CUDA single-factor comparisons, with fixed model, devices, budget and request traces. Performance comparisons use five randomized repeats unless the matrix is a direct workload-coverage probe.

- Feature cache off/on and KV sharing off/on retain token-level ITL samples. KV sharing reused 80/92 prompt tokens but median Language latency changed from 751.75 to 756.62 ms, so no speedup is claimed.
- Vision singleflight reduced the physical batch from 8 to 1 and median forward time from 629.21 to 417.16 ms.
- Equal-stream lane scheduling used two streams, max active=2 and 256 MiB per transfer in both modes; median D2H completion was 10.51 ms off and 5.67 ms on.
- Merged Language and separated Prefill/Decode used the same GPU and 64 KV blocks and produced the same 16 tokens. Median wall time was 23,504.24 versus 24,544.52 ms; separated mode pays for two model processes.
- Dependency-aware Host/checkpoint, fixed-GPU retain and drop/recompute each ran five times. Host/checkpoint restored 25/25 pages with no failure but had higher median TTFT (64.72 ms) than fixed GPU (57.45 ms) and recompute (55.79 ms); this negative result is retained.
- Fixed, round-robin and data-aware placement each served 55 requests over five repeats on the same two worker grants. The simple model's 1,101.28 ms mean prediction error is retained, and no global-optimality claim is made.
- A real persistent-Vision workload produced exact cache-hit rates 0%, 50% and 90% in one worker. At high resolution, eliminating the Vision forward did not eliminate processor/RPC wall time; logical hits are not reported as equal saved end-to-end compute.

Host copy, same-GPU IPC and sparse cross-GPU page copy evidence is linked from the consolidated result. The branch 1/2/4, short/long prefix, two-resolution and pressure-level representative matrix remains non-Cartesian as required. `unmeasured_fields` is empty. No comparison with SGLang or another external system is claimed.
