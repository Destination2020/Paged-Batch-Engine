# M2 execution report

M2 is accepted. The data module defines versioned typed identities, errors,
references, bundle validation, fixed-width wire encoding, and an authoritative
single-owner local runtime. Reserve is idempotent and performs whole-object
capacity checks; seal publishes only complete data and returns the canonical
winner; Read/Compute/IO leases delay physical reuse; conflicting producers are
quarantined until explicit owner-confirmed quiescence.

`KVPoolView` separates compute binding from physical allocation ownership.
Paged scatter, prefill attention, and decode attention now consume the view;
the local `BlockAllocator` remains the sole free-queue owner. External pools
can later import the same view contract without constructing another allocator.

The existing checkpoint serving path now uses `LocalDataRuntime` for immutable
publication metadata. Prepare reserves and writes an invisible object, commit
seals it, restore holds a typed lease, and erase withdraws and releases it.
The integration retains the existing PageDirectory, HostStore, and
TransferScheduler paths and their idle completion behavior.

All CPU and sanitizer tests passed, as did the CUDA pool-view attention tests,
the ordinary CUDA suite, three real Qwen2 text regressions, and the two M1 real
Qwen2.5-VL regressions. Exact counts and log paths are in `checks.json`.
