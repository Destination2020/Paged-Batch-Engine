# E2 execution report

E2 is accepted on 2026-09-13. The Coordinator runs two real compatible Language workers, each with a separate 32-slot grant from one 64-slot owner pool. Placement filters model/representation/incarnation and capacity, treats stale statistics conservatively, then scores queue wait, missing bytes over calibrated bandwidth, compute and RPC cost. Hotspot limits, bounded in-flight placement, hysteresis and retry exclusion are enforced.

The real trace covers cached-but-long-queue, empty remote, stale statistics, budget fallback and worker restart. Fixed, round-robin and data-aware policies each ran 55 requests over five randomized repeats with the same endpoints and total capacity. Mean client Language time was 1096.32, 1012.02 and 951.82 ms respectively; client TTFT was not uniformly better for data-aware. Its mean decision cost was 27.80 microseconds and mean absolute prediction error was 1101.28 ms. The intentionally conservative estimator error is retained rather than fitted after the run.

Five C++ and three Python placement contracts pass. All final status records show two active 32-slot grants during service, then 64 free slots and zero leases after shutdown.
