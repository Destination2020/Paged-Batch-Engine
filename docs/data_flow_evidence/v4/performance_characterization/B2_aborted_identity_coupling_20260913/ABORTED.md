# Aborted B2 run: cache identities were coupled

This run was stopped after eight trial directories had been created. It is
retained as raw audit evidence and is excluded from every performance result.

The pre-run implementation made `--disable-feature-cache` salt the value
returned as `feature_content`. The language worker correctly includes that
content identity in its context-dependent semantic KV prefix key, so disabling
the visual feature cache also disabled semantic KV reuse. Consequently arm
`01` did not isolate “feature cache off, semantic KV cache on”.

The correction keeps the canonical content-derived media identity stable and
changes only whether the Vision worker looks up or publishes the encoded
feature object. The complete randomized sequence is rerun from trial zero;
partial measurements here are not reused.

Original command and stderr/stdout are preserved in
`../logs/b2_aborted_identity_coupling_20260913.log` and the per-trial
`launch.log` files.
