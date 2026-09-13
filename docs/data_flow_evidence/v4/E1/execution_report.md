# E1 execution report

E1 is accepted on 2026-09-13. The real Language serving path computes its semantic radix identity before Prefill from model/adapter representation, ordered text tokens, raw media ContentId, processor/feature representation, media span, three-axis positions and RoPE delta. The identity does not depend on request ID, GPU address or a producer's logged key.

In the final five-round run, a fresh exact replay and a same-image/different-question request each matched 80 prefix tokens. A changed processor/size and a different local image with identical placeholder structure both matched zero. The exact replay used the normal attention path and passed either exact-output equality or the executable same-history BF16 logits/top-2 contract. The local `test/data_core/fixtures/e1_different.ppm` makes the negative image case reproducible.

Cross-GPU Decode copied 3,538,944 bytes of required pages per consumer instead of the 37,748,736-byte owner pool, produced the frozen 16-token output and performed the 589,824-byte tail COW. Twenty-five directory, generation, contiguous-prefix, transfer, Host restore and identity contracts passed. Five-repeat cache off/on results retain the small lookup overhead and lack of end-to-end speedup.
