# Four-branch cross-role COW and cancellation acceptance

Date: 2026-09-13

One real 95-token multimodal Prefill published an external page table backed by
the data-service CUDA pool. Four independent Decode PIDs acquired that same
prefix. The live service sample showed five simultaneous grants (one retained
prefix and four disjoint private grants), with 32/64 slots free.

Every branch executed real Qwen paged attention and copied the immutable partial
tail on first append: 589,824 bytes per branch. Two branches completed 16 tokens
and exactly matched the Prefill oracle. Two other branches generated four
additional tokens, were cancelled through `Scheduler::cancel_request`, reached
the terminal failed/cancelled state, synchronized CUDA, and released only their
own metadata/private grants. Their cancellation did not change either surviving
branch's output or prefix pages.

After all four branches exited, only the prefix grant remained and 48 slots were
free. Releasing the prefix last restored 64/64 slots, zero active grants and zero
metadata leases.

The first two attempts are deliberately retained. Attempt one exposed a missing
initial-page binding in the new cancellation role. Attempt two demonstrated the
scheduler correctly refusing a declared 68-token growth window against a
four-page grant; the test helper was corrected to declare the bounded intended
window. The third run is the accepted evidence. Reproduce with
`tools/launch/four_branch_vlm_ipc.sh`.
