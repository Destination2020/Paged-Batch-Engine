# Aborted B2 run: BF16 first-divergence rule was incomplete

This run was stopped after eight completed trials and is excluded from all
performance summaries. A long-context cache-off trial exposed an expected
BF16 divergence: on the same history the baseline top-2 logits tied (margin
0), the comparison margin was 0.125, both selected IDs belonged to the same
top-2 set, and max/mean absolute logits error was 0.21875/0.03536. Later tokens
then necessarily had different histories.

The final gate checks error thresholds only while histories match, requires a
first differing selection to have the same top-2 set and margins <=0.25, and
accepts later `same_history=false` entries only after that observed justified
divergence. The preserved trial independently re-evaluates as 54 same-history
steps, 2 justified first divergences, 10 post-divergence steps and 0 failures.

Original output is retained here and in
`../logs/b2_aborted_bf16_gate_20260913.log`.
