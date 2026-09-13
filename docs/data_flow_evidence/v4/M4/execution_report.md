# M4 execution report

M4 is accepted with the Python client speaking the same fixed-width binary protocol as the C++ service. The real Vision role constructs only `Qwen2_5_VisionTransformerPretrainedModel` and loads the 390 `visual.*` tensors from the frozen Qwen2.5-VL-3B checkpoint; it does not load language-model weights.

For the frozen 224×224 M1 image, the worker produced a `[64, 2048]` BF16 feature tensor and `[1, 16, 16]` grid. Against the M1 reference, max absolute error was 0.59375 and mean absolute error was 0.0172948. The atomically sealed 262,424-byte bundle contains typed shapes, offsets, byte lengths, and per-component checksums. A C++ decoder accepted a Python-produced golden bundle and rejects malformed coverage through the existing bundle tests.

A repeated request acquired the service object with `forward_count=0`. Two simultaneous cold requests for a new resize elected one node-local leader: the leader ran one visual forward and the follower waited for the canonical publication with `forward_count=0`. A real two-item variable-length visual batch split 81 and 121 feature tokens into independently sealed bundles. A cancelled consumer released its lease, after which another independent process read and verified the same bundle; service stats returned to zero active leases.

The current Python data path is explicitly recorded as bounded binary Host copy. CUDA IPC import and cross-GPU P2P were separately proven in M3; this report does not label the Python Host path zero-copy.
