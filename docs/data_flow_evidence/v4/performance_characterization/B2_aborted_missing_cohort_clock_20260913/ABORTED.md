# Aborted B2 run: direct cohort clock was missing

This run was stopped after three completed trials. Its cache/identity data are
valid, but it is excluded from performance summaries because throughput used
the sum of sequential round durations instead of directly recording the first
measured coordinator-send through last-terminal wall window.

The coordinator now emits monotonic `coordinator_started_ns` and
`coordinator_terminal_ns` for every round. The final run derives the closed-loop
cohort window directly from these fields. Original output is preserved in this
directory and `../logs/b2_aborted_missing_cohort_clock_20260913.log`.
