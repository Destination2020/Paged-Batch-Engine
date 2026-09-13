# M8 execution report

Accepted on 2026-09-12. The matrix covers invisible unsealed writes, idempotent reserve/release, stale generation and owner rejection, canonical races, checksum quarantine, malformed bundle preflight, independent consumer cancellation, copy drain, checkpoint revision/outbox recovery, and 10,000 acquire/release cycles. Five real CUDA tests gated transfers while cancelling owners and verified source/target retention, plain/FP8 restore, and 32-waiter singleflight.

A data-service process was hard-killed after publication and restarted on the same endpoint. Its incarnation changed and lookup of the old content returned protocol error 5 (`NotReady`); no old allocation was revived. Normal shutdown drains workers and destroys remote leases before the runtime. Raw evidence distinguishes normal churn from quarantine and service-restart recovery.

## 2026-09-13 final remediation

The online matrix now includes an acknowledged external cancel received during Language inference, a cross-stage absolute deadline, stale/duplicate generations, and real multimodal checkpoint/Host restoration. Recovery installs dependencies before KV and restores mRoPE positions, RNG and outbox cursor; cancellation only drops the request's references. Thirty E3 contract/CUDA tests passed. Targeted compute-sanitizer ran the Host plain/FP8 scatter and Host-only radix restore tests with zero errors. Evidence: `../M5_online_lifecycle_final`, `../E3/faults`, and `../final_regression`.
