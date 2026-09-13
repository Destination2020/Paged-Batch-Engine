# M6 execution report

Accepted on 2026-09-12. `BranchSnapshot` captures committed/computed/sampled/pending boundaries and first-token inheritance. A fork increments physical page references; a partial tail is immutable and performs a device-local bitwise COW before the first append. Four branches shared the original full/tail pages, each copied 512 bytes across two layers, two were cancelled, surviving KV remained intact, and all 48 allocator pages returned free after arbitrary exit order. The aligned case shared two full pages and appended with zero COW bytes.

The multimodal radix key embeds all 256 image ContentId bits inside the placeholder span without changing sequence length. Same tokens plus a different image missed. `PrefixBuilder` ran as a temporary Prefill task, exited with no active request, and a later request reused eight resident tokens. Raw test output is in `raw/branch_prefix_tests.log`.
