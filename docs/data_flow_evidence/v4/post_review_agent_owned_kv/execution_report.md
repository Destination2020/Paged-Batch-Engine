# Agent-owned GPU KV serving main-path evidence

Date: 2026-09-13

The data service created and retained one 37,748,736-byte CUDA allocation with
the exact Qwen2.5-VL language KV layout. Prefill requested 16 slots from the
service and real Qwen2 paged-attention kernels wrote KV into those imported
slots. The producer synchronized the CUDA work and published a 1,456-byte page
table and sequence-state object; it did not publish a Host KV snapshot.

After the Prefill process exited, two independent Decode processes acquired the
same published prefix. Each received four disjoint private slots from the same
service-owned pool. A live pool sample observed three simultaneous grants: one
retained prefix and two Decode grants. Both Decode results exactly matched the
Prefill oracle. Each Decode copied 589,824 bytes when appending to the shared
partial tail (36 layers times one BF16 key/value page), which demonstrates real
cross-process COW in the attention path.

The decoders synchronized their streams before releasing their private grants.
After both exited, 48 slots were free and only the 16-slot prefix grant remained.
The coordinator then released the full prefix lease token; all 64 slots became
free and the metadata service reported zero active leases.

Reproduce with `tools/launch/multi_role_vlm_ipc.sh`. The raw process and pool
output is in `raw/mainpath.log`; machine-readable assertions are in
`checks.json`.

This accepts the same-GPU Agent-owned attention, normal lifecycle, concurrent
consumer, and cross-role partial-tail COW subitems. Cross-GPU accounted copy,
persistent online roles, and unified admission remain open and are not inferred
from this run.
