# Persistent Vision role acceptance

Date: 2026-09-13

The Qwen2.5-VL Vision model and processor remained resident in one process. A
typed Unix JSON-line protocol accepted request id/generation, ordered image and
text parts, absolute timeout, cancel, and shutdown operations. The role uses a
bounded queue and a 100 ms batching window.

Two concurrent requests with different resolutions entered one actual visual
forward. Their merged feature lengths were 81 and 121 and their admitted bundle
sizes were 334,024 and 498,568 bytes. All feature and request-bundle
reservations completed before the forward began. The single packed forward took
447.190 ms and split the output back into the two canonical publications.

A later request reused the first image and processor settings with a different
question. It acquired the same feature object, performed no additional visual
forward, and published a distinct request bundle containing its own input ids,
mRoPE positions and delta. This closes the prior identity bug where text-specific
position state could be returned under an image-only key. A queued request was
then cancelled by matching request generation before forward and published no
request bundle.

The worker stopped normally with `forward_count=1`, `batch_count=2`; the data
service reported zero active leases. Raw request/reply JSON, role lifecycle and
service statistics are in this directory. Reproduce with
`tools/launch/persistent_vision_role.sh`.

The same launcher was also run with the PBE model and tokenizer arguments. The
new text-specific v3 request bundle was acquired by the C++ PBE Prefill path,
which consumed 106 dynamic tokens and 81 feature rows with `rope_delta=-72`.
The subsequent PBE Decode output matched the Prefill oracle for all 16 tokens.
Those raw logs are in sibling `post_review_persistent_vlm_bundle/`; the KV
transport in this compatibility run is explicitly recorded as `host_snapshot`,
so the separate M2/M3 external-pool evidence remains the authority for GPU KV
ownership.
