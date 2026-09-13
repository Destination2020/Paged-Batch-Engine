# PBE V4 diagrams

- `pbe_v4_overall_architecture.png` / `.svg`: full-project module map, following the original overview's level of detail; includes engine scheduling, model operators, role orchestration, cache/data ownership, deployment and validation. Regenerate both with `python3 imgs/build_overall_architecture.py` (Pillow + fontconfig).
- `pbe_v4_architecture.png`: conceptual control / compute / data / resource layers. Arrows indicate logical integration, not tensor copies through the Coordinator.
- `pbe_v4_same_gpu_sharing.png`: one-GPU 1P1D shared allocations and private execution state. It is not a claim of arbitrary multi-worker scaling or hardware-enforced read-only memory.

The overview was created with image generation. The same-GPU diagram has an editable SVG source and a PNG rendering; it uses explicit allocation labels to avoid ambiguous pointer arrows. Both were reviewed against the V4 plan, persistent PD report, E4 shared-weight report and implementation. Labels are English for readability at README scale; surrounding README explanations are Chinese. These are explanatory illustrations, not measured performance evidence.

To regenerate the same-GPU PNG with Pillow and fontconfig installed:

```bash
python3 imgs/render_sharing_diagram.py
```

The original root-level architecture PNG and ZMQ/NCCL JPG are retained as historical assets. The latter remains in the README's compatibility section because its launch commands describe that older path.
