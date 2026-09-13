# GitHub evidence snapshot

This repository includes source code, tests, execution plans and selected experiment reports. Raw experiment artifacts remain in the original local workspace.

`EVIDENCE_PUBLICATION_MANIFEST.json` records the paths, sizes and SHA-256 hashes at publication time, and whether each artifact is included in Git. Small Markdown, JSON, CSV and SVG artifacts below 1,000,000 bytes are included. Raw logs, request traces, tensors, images and larger aggregate files are excluded to keep the source repository manageable. Local files have not been deleted.

Historical reports and completion manifests refer to their original execution snapshots. Their references may include artifacts absent from this checkout. A matching hash is an integrity check, not an independent correctness or performance validation. Full validation requires regenerating or obtaining the corresponding raw artifacts; do not interpret missing files as passing tests.

Use the commands in the execution/delivery guides and experiment runners under `tools/bench/data_flow/` to reproduce evidence with the specified model, hardware and dependencies. Model weights and build products are not distributed here.
