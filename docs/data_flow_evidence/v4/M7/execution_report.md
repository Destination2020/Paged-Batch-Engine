# M7 execution report

Accepted on 2026-09-12. `NodeMemoryBudget` freezes and independently conserves weights, workspaces, KV, bundles, staging, and quarantine, while joint reservations either commit every pool or change none. The capacity-four/two-grower case makes one explicit rejection and progresses after release instead of holding partial resources.

Seventeen pressure contracts passed: allocator exhaustion rollback, stale allocation intent rejection, host restore, direction lanes, demand entry reserve, absolute deadlines, donation/cancellation, aging, restore dependencies, decode checkpoint preemption, and 10,000 checkpoint lifecycles with no record/payload/revision growth. Real H20 gated CUDA copy coverage is carried into M8. Raw output is `raw/budget_pressure_tests.log`.

## 2026-09-13 final remediation

Admission now uses the model's real preallocated workspace profile (643,840 bytes/token; 329,646,080 bytes at 512 tokens), measured model-process allocation, pre-existing same-device Vision/service use and the external owner KV pool. The final observed peak was 10,028,777,472 bytes under a 149,753,298,944-byte admission limit; the model process accounted 7,327,449,088 bytes and the external KV owner 37,748,736 bytes. These are device readings, not only ledger conservation.

The real multimodal recovery run demoted five prefix pages to Host, restored all five through physical H2D, restored a 14-token private tail, 276 position values, sampling counter and exact feature dependency, then continued generation with zero restore failures. A separate Vision run demoted and restored a 262,144-byte feature GPU→Host→GPU, then served it without encoder recompute. The first BF16 divergence was validated under identical history with the same numerical contract. Five randomized policy repeats retain the observed overhead of Host/checkpoint versus fixed GPU and recompute. Evidence is under `../M5_online_lifecycle_final`, `../E3/feature_gpu_host_recovery`, `../E3/real_recovery/final_run`, and `../M9/pressure_policy_ab`.
