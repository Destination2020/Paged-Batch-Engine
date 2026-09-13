"""Build the editable PBE overview and its high-resolution README rendering."""
from pathlib import Path
from html import escape
from render_sharing_diagram import render_diagram

ROOT = Path(__file__).resolve().parent
items = []


def rect(x, y, w, h, fill, stroke='#d6dfeb', radius=14):
    items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>')


def text(x, y, value, cls='body', fill='#10244c', anchor='start'):
    items.append(f'<text x="{x}" y="{y}" class="{cls}" fill="{fill}" '
                 f'text-anchor="{anchor}">{escape(value)}</text>')


def panel(x, y, w, h, title, subtitle, color, bg):
    rect(x, y, w, h, bg, color, 20)
    text(x + 22, y + 40, title, 'heading', color)
    text(x + 22, y + 71, subtitle, 'small', '#526782')


def card(x, y, w, h, title, lines, color='#10244c'):
    rect(x, y, w, h, '#ffffff')
    text(x + 18, y + 33, title, 'heading', color)
    for i, line in enumerate(lines):
        text(x + 18, y + 64 + i * 28, line, 'body')


def arrow(x1, y1, x2, y2, color='#527391'):
    items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}"/>')
    if x1 == x2:
        sign = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2-7, y2-sign*12), (x2+7, y2-sign*12)]
    else:
        sign = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2-sign*12, y2-7), (x2-sign*12, y2+7)]
    points = ' '.join(f'{x},{y}' for x, y in pts)
    items.append(f'<polygon points="{points}" fill="{color}"/>')


def build():
    rect(0, 0, 2400, 1670, '#ffffff', '#ffffff', 0)
    text(40, 65, 'PBE | Overall Architecture & Request Flow', 'title')
    text(40, 108, 'C++ / CUDA paged inference • Multimodal roles • Framework-owned KV, features and weights', 'subtitle')
    text(2360, 65, 'V4 + E1–E4', 'heading', '#526782', 'end')

    # Top-level execution and orchestration: explicit scope in each panel.
    panel(40, 148, 330, 710, '01  ENTRYPOINTS', 'Text, multimodal and offline paths', '#2563a0', '#f1f7ff')
    card(60, 247, 290, 155, 'Text HTTP / SSE', ['/generate', '/v1/chat/completions', 'Health / metrics'])
    card(60, 426, 290, 155, 'Multimodal input', ['Image + text', 'Typed role requests', 'Processor / position plan'])
    card(60, 605, 290, 146, 'Offline / tools', ['CLI and model export', 'Benchmark runners', 'Frozen model manifests'])
    text(60, 792, 'Legacy text API and typed VLM', 'small', '#526782')
    text(60, 818, 'entry are distinct paths.', 'small', '#526782')

    panel(400, 148, 630, 710, '02  SERVING RUNTIME', 'Inside a PBE language execution process', '#19765d', '#eff9f5')
    card(420, 247, 270, 184, 'Request lifecycle', ['OnlineServingEngine', 'Queue / submit / cancel', 'SequenceState', 'Output / finish events'])
    card(720, 247, 290, 184, 'Scheduler', ['Continuous batching', 'Token / sequence budgets', 'Chunked prefill', 'FCFS / priority'])
    arrow(690, 338, 720, 338)
    card(720, 485, 290, 170, 'MixedBatchBuilder', ['Token IDs / positions', 'Slot mapping', 'Block tables / lengths'])
    arrow(865, 431, 865, 485)
    card(420, 485, 270, 170, 'Sampling / output', ['Logits → token samples', 'Update sequence state', 'Finish or next step'])
    arrow(555, 485, 555, 431)
    card(420, 697, 590, 133, 'Preemption & continuation', ['RequestCheckpoint • suspend / restore', 'KV + multimodal position / progress state'])

    panel(1060, 148, 600, 710, '03  MODEL & CUDA EXECUTION', 'Qwen2 / Qwen2.5 text + Qwen2.5-VL BF16 language', '#b56a16', '#fffaef')
    card(1080, 247, 560, 128, 'Qwen2Model forward', ['forward_mixed_batch / forward_decode_batch', 'Token / visual embeddings and position inputs'])
    arrow(1360, 375, 1360, 405)
    card(1080, 405, 560, 128, 'Transformer operators', ['Embedding • RMSNorm • Q/K/V • RoPE / mRoPE', 'Attention • output projection • SwiGLU / MLP'])
    arrow(1360, 533, 1360, 563)
    card(1080, 563, 560, 128, 'Paged attention runtime', ['Scatter KV • prefill attention • decode attention', 'External GPU page views or process-owned pool'])
    arrow(1360, 691, 1360, 721)
    card(1080, 721, 560, 109, 'CUDA kernels & cuBLAS', ['BF16 mainline; existing text FP8 KV path retained'])

    panel(1690, 148, 670, 710, '04  MULTIMODAL ROLE ORCHESTRATION', 'Logical role graph; deployment is configured separately', '#3564a2', '#f0f6ff')
    card(1710, 247, 630, 154, 'Typed Coordinator', ['Absolute deadline • external cancellation', 'Queue / locality / transfer-cost placement', 'Joint resource admission and bounded retry'])
    card(1710, 440, 630, 116, 'Vision role  |  Python / PyTorch', ['Variable-length batches • reusable feature cache'])
    arrow(2025, 401, 2025, 440, '#3564a2')
    card(1710, 587, 297, 141, 'Prefill role', ['PBE C++ / CUDA', 'Compute / publish KV', 'Uses engine core at left'])
    card(2043, 587, 297, 141, 'Decode role', ['PBE C++ / CUDA', 'Restore KV / generate', 'Uses engine core at left'])
    arrow(2007, 655, 2043, 655)
    text(2025, 762, 'P→D handoff carries identity, generation and page metadata.', 'small', '#526782', 'middle')
    text(2025, 797, 'Same-GPU IPC / cross-GPU copy are separate paths.', 'small', '#526782', 'middle')
    text(2025, 824, 'Shared weights: same GPU + compatible model / layout.', 'small', '#526782', 'middle')

    # Only inter-panel edges with a single, unambiguous meaning.
    arrow(370, 323, 400, 323, '#2563a0')
    items.append('<line x1="1010" y1="570" x2="1045" y2="570" stroke="#b56a16"/>')
    items.append('<line x1="1045" y1="570" x2="1045" y2="311" stroke="#b56a16"/>')
    arrow(1045, 311, 1080, 311, '#b56a16')
    items.append('<line x1="1060" y1="676" x2="555" y2="676" stroke="#b56a16"/>')
    arrow(555, 676, 555, 655, '#b56a16')
    text(1065, 891, 'Batch metadata → model; logits → sampler; sampled outputs advance Scheduler state.', 'small', '#526782', 'middle')

    panel(40, 923, 2320, 488, '05  SHARED DATA, PAGED CACHE & RESOURCE LIFECYCLE',
          'Workers manage logical execution views; framework owners manage shared physical allocations and grants.', '#6b46c1', '#f8f5ff')
    arrow(865, 858, 865, 923, '#6b46c1')
    arrow(2025, 858, 2025, 923, '#6b46c1')
    text(880, 916, 'Page tables / checkpoint dependencies', 'small', '#6b46c1')
    text(2040, 916, 'Acquire / publish / release', 'small', '#6b46c1')

    card(60, 1020, 545, 220, 'Data service / Node Agent', ['ContentRegistry • DataClient • typed wire protocol', 'Content / representation / allocation identities', 'Generation-bound leases • idempotent operations', 'CUDA IPC pool and shared-weight owner', 'TensorBundle: features + request-specific metadata'], '#6b46c1')
    card(625, 1020, 545, 220, 'KVCacheManager & prefix reuse', ['Per-request page tables • grants / pool views', 'Semantic multimodal prefix identity', 'Longest continuous prefix • missing-page fetch', 'Radix index • shared prefix • tail-page COW', 'New content or position semantics → distinct KV'], '#6b46c1')
    card(1190, 1020, 545, 220, 'Tiering & physical transfer', ['PageDirectory • HostStore • PageMigration', 'TransferScheduler • singleflight', 'Demand / prefetch / background lanes', 'GPU ↔ Host • dependency-aware recovery', 'Cross-GPU copy is measured data movement'], '#6b46c1')
    card(1755, 1020, 585, 220, 'Memory budget & safety', ['Weights + KV + workspace + staging + features', 'Shared physical weights counted once', 'Private state and CUDA overhead counted separately', 'Producer / consumer completion fences', 'Drain / release / quarantine / fail-stop recovery'], '#6b46c1')
    rect(60, 1261, 2280, 123, '#eee8fa', '#ded3f3')
    text(85, 1295, 'PHYSICAL OWNERSHIP', 'heading', '#6b46c1')
    text(85, 1330, 'Shared weight allocation  |  Agent-owned KV pages  |  Feature / request bundles  |  Host recovery copies', 'body')
    text(85, 1362, 'Immutable views may be shared; workspace, activations, scheduler state and writable tails stay private to their consumers.', 'body')

    panel(40, 1441, 1130, 158, '06  DEPLOYMENT & COMPATIBILITY', 'Single-node V4: TP=1, image + text, trusted processes', '#526782', '#f4f6f9')
    text(62, 1545, 'Launchers: Data service + Vision + persistent P / D • same-GPU IPC / cross-GPU copy', 'body')
    text(62, 1576, 'Legacy serving_qwen modes retained: local P+D • P2P • ZMQ / NCCL (build-dependent)', 'body')
    panel(1200, 1441, 1160, 158, '07  VALIDATION & OBSERVABILITY', 'Correctness / faults / resources / performance are separate gates', '#526782', '#f4f6f9')
    text(1222, 1545, 'C++ / CUDA tests • real VLM numerics • lifecycle faults • compute-sanitizer', 'body')
    text(1222, 1576, 'Request traces • memory accounting • repeated A/B • raw evidence / manifests', 'body')
    text(40, 1641, 'Architecture map, not a performance claim. No implication of arbitrary TP, cross-node RDMA or unrestricted multi-P/D scaling.', 'small', '#526782')
    text(2360, 1641, 'PAGED BATCH ENGINE', 'small', '#526782', 'end')

    style = '''<style>text {font-family:Arial,sans-serif;} .title {font-size:40px;font-weight:700;}
    .heading {font-size:26px;font-weight:700;} .subtitle {font-size:22px;}
    .body {font-size:20px;} .small {font-size:17px;}</style>'''
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1670" '
           'viewBox="0 0 2400 1670" role="img" aria-labelledby="title">'
           '<title id="title">PBE complete project architecture and request flow</title>'
           + style + '\n'.join(items) + '</svg>\n')
    source = ROOT / 'pbe_v4_overall_architecture.svg'
    source.write_text(svg)
    render_diagram(source)


if __name__ == '__main__':
    build()
