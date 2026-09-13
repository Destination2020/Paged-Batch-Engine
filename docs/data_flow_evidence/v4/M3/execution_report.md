# M3 execution report

M3 is accepted on checkout `f3597a763b6cd5cf60b826baa2dc94395c3d02fd` with the inherited dirty worktree preserved.

The new data service uses a fixed Unix-domain endpoint and a versioned, fixed-width little-endian binary protocol. Its `ContentRegistry` owns bounded object and lease metadata; eight worker threads service a bounded 256-connection queue, while runtime metadata changes remain in short critical sections. Reserve and seal converge on the canonical `DataRef`; producer, read, compute, and IO references continue to govern physical lifetime. Host payload bytes are counted and reported as `host_copy`.

The CUDA pool probe exports Agent-owned memory and producer/consumer interprocess events. A separate importer PID validated a same-GPU 16 MiB mapping with zero inter-GPU transfer bytes. A device-1 importer validated a device-0 64 MiB pool through an explicit 64 MiB P2P copy; this path is reported as `cross_gpu_p2p_copy`.

The real-model run used PBE C++/CUDA with Qwen2-0.5B-Instruct. Prefill PID 1705261 computed and sealed two complete KV pages (32 tokens, 395,380 serialized bytes), then exited. Decode A PID 1705337 on GPU 0 and Decode B PID 1705366 on GPU 1 subsequently acquired the same ContentId. Both restored the KV into their own page tables, skipped all 32 prefill tokens, and produced token IDs `34,11829,62576,525,264,13`, exactly matching the producer oracle. After both releases, the service reported one retained canonical object and zero active leases.

Validation:

- Data/runtime/codec tests: 14 passed.
- PD engine, handoff, checkpoint, and pool-view regression: 27 passed; two NCCL-only tests skipped because NCCL was disabled. CUDA P2P passed.
- Standalone ASan/UBSan/LSan lifecycle probe: passed.
- `git diff --check`: passed.

Raw logs are under `raw/`. The one-command reproduction entry is `tools/launch/multi_role_text.sh`.
