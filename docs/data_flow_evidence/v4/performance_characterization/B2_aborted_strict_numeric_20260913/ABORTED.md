# Aborted B2 run: nondeterministic Vision recompute violated frozen numerics

This run was stopped after 20 completed trials and is excluded from all
performance summaries. Its first KV-only trial correctly separated cache
identities (0 feature hits, 10 Vision forwards, 400 semantic KV tokens saved),
but one same-history comparison had logits max/mean absolute error
1.375/0.257504, exceeding the existing M5/E4 limits 0.75/0.20. The selected
token happened to match; that does not make the failure acceptable.

A non-measured pilot enabled deterministic Torch/cuDNN algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The same KV-only cell then passed all 40
same-history steps with maxima 0.3125/0.090580. The final B2 matrix freezes
that setting for every arm, not only the candidate. Pilot evidence is in
`../B2_pilot_deterministic_arm01/` and
`../logs/b2_deterministic_pilot_validation.log`.

Original output is retained here and in
`../logs/b2_aborted_strict_numeric_20260913.log`.
