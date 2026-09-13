# M5 execution report

M5 is accepted with a real Encode → Join → Prefill → Decode process chain. The Vision PID published the M4 feature bundle. The PBE 3B Prefill PID acquired it, validated the typed bundle, replaced the 64 image-placeholder embedding rows, applied the frozen three-axis positions, ran all 95 prompt tokens, and published a 3,546,780-byte KV object. Two later PBE Decode PIDs on GPUs 0 and 1 independently restored that object and skipped the same 95 prefill tokens.

Both decoders produced the first 16 frozen reference tokens exactly: `785,2168,374,264,6396,15941,429,45380,279,1882,315,821,8317,1948,264,2943`. The output begins “The image is a flowchart that illustrates the process of data transfer between a client”. The service ended with zero active leases.

The numerical matrix also covers text, one image at two resolutions, two images, a variable-length mixed batch, 32 teacher-forced decode steps, and eight chunked-prefill cells at caps 17 and 31. These chunks cross visual spans and compare final logits to the frozen reference. Coordinator tests enforce generation, stage order, deadline, failure, and cancellation. The multimodal contract rejects span/feature, overlap, grid, and byte-coverage mismatches before copying.

Raw logs and the standalone command are in `raw/epd` and `tools/launch/multi_role_vlm.sh`.

## 2026-09-13 final remediation

The executable 20-case gate now accepts BF16 divergence only at the first differing step when both paths have identical prompt, position, RoPE and generated-token history. It requires logits max error below 0.75, mean error below 0.20, at least one top-2 margin at or below 0.25, and both selected tokens in the union of the two top-2 sets. The final run has 15/20 exact sequences and 141/160 equal tokens; all five non-exact cases satisfy that contract. This is not a per-token equality claim and does not reuse another fixture's tie evidence.

The Language role now uses bounded JSONL v2 with an owner queue and operation IDs. A reader remains active while inference runs, so a client cancel interrupted Decode after two emitted tokens. The 10,500 ms absolute deadline was created before Encode and expired at 10,503.42 ms after 7,676.86 ms of Encode/Join; it was not restarted at Language. Evidence: `../M5_numerics_final/validation_final.json` and `../M5_online_lifecycle_final/validation.json`.
