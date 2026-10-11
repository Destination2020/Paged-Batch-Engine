#!/usr/bin/env python3
"""Extract auditable, bounded GitHub figures from Nsight CUDA software traces.

Requires matplotlib only for plotting. SQLite timestamps are integer nanoseconds;
CPU launch intervals and CUDA API durations are deliberately not used.
"""
import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3

from validate_kv_reuse import audit_trace


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def read_kernels(path, labels):
    with sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True) as db:
        processes = dict(db.execute('SELECT globalPid, pid FROM PROCESSES'))
        rows = db.execute('''SELECT k.start, k.end, k.globalPid, k.deviceId,
            k.contextId, k.streamId, s.value, k.gridX, k.gridY, k.gridZ,
            k.blockX, k.blockY, k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start''')
        keys = ('start_ns', 'end_ns', 'pid', 'device', 'context', 'stream',
                'kernel', 'grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z')
        kernels = []
        for row in rows:
            pid = processes.get(row[2])
            if pid in labels:
                if row[1] <= row[0]:
                    raise ValueError('Invalid CUDA device timestamp')
                kernels.append(dict(zip(keys, (*row[:2], pid, *row[3:]))))
    if {k['pid'] for k in kernels} != set(labels):
        raise ValueError('Missing GPU kernel records for requested PID(s)')
    if len({k['device'] for k in kernels}) != 1:
        raise ValueError('This analysis requires the same GPU for both workers')
    return kernels


def overlap_summary(kernels):
    """Count cross-PID pairs and union intersection, without double counting streams."""
    active, endings = {}, []
    pair_count = overlap_ns = union_ns = 0
    previous = kernels[0]['start_ns']
    best = None
    for index, kernel in enumerate(kernels):
        now = kernel['start_ns']
        while endings and endings[0][0] <= now:
            end, expired = heapq.heappop(endings)
            if active:
                union_ns += end - previous
            if len({item['pid'] for item in active.values()}) > 1:
                overlap_ns += end - previous
            previous = end
            del active[expired]
        if active:
            union_ns += now - previous
        if len({item['pid'] for item in active.values()}) > 1:
            overlap_ns += now - previous
        previous = now
        for other in active.values():
            if other['pid'] == kernel['pid']:
                continue
            duration = min(other['end_ns'], kernel['end_ns']) - now
            pair_count += 1
            if best is None or duration > best['overlap_ns']:
                best = dict(overlap_ns=duration, start_ns=now,
                            end_ns=now + duration, kernels=[other, kernel])
        active[index] = kernel
        heapq.heappush(endings, (kernel['end_ns'], index))
    while endings:
        end, expired = heapq.heappop(endings)
        union_ns += end - previous
        if len({item['pid'] for item in active.values()}) > 1:
            overlap_ns += end - previous
        previous = end
        del active[expired]
    return dict(kernel_count=len(kernels), cross_process_overlapping_pairs=pair_count,
                cross_process_interval_union_ns=overlap_ns, any_kernel_union_ns=union_ns,
                longest_pair=best,
                kernel_counts_by_pid={str(pid): sum(k['pid'] == pid for k in kernels)
                                      for pid in sorted({k['pid'] for k in kernels})})


def timeline(kernels, labels, summary, output, title, probe=False):
    import matplotlib.pyplot as plt
    best = summary['longest_pair']
    if best is None:
        raise ValueError('No cross-process GPU kernel overlap found')
    start = min(k['start_ns'] for k in best['kernels'])
    end = max(k['end_ns'] for k in best['kernels'])
    margin = max((end - start) // 5, 20_000)
    lower, upper = start - margin, end + margin
    selected = [k for k in kernels if k['end_ns'] > lower and k['start_ns'] < upper]
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernels[0]))
        writer.writeheader()
        writer.writerows(selected)
    scale, unit = (1e6, 'ms') if probe else (1e3, 'us')
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = ['#1877b9', '#e18026']
    for row, pid in enumerate(labels):
        spans = [((k['start_ns'] - start) / scale,
                  (k['end_ns'] - k['start_ns']) / scale) for k in selected if k['pid'] == pid]
        ax.broken_barh(spans, (row - .22, .44), facecolors=colors[row], edgecolors='white', linewidth=.4)
    ax.axvspan((best['start_ns'] - start) / scale, (best['end_ns'] - start) / scale,
               color='#35a878', alpha=.18, label=f"Cross-PID overlap: {best['overlap_ns']/scale:.3f} {unit}")
    ax.set(yticks=range(len(labels)), yticklabels=[f'{label}\nPID {pid}' for pid, label in labels.items()],
           xlabel=f'GPU time relative to {start} ns ({unit})',
           xlim=((lower-start)/scale, (upper-start)/scale), ylim=(-.6, 1.6), title=title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='upper right', fontsize=9)
    fig.text(.02, .02, 'CUDA software trace | device kernel intervals; see SM progress evidence to exclude time slicing', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output, metadata={'Date': None})
    plt.close(fig)
    return dict(start_ns=lower, end_ns=upper, selection='window around longest cross-PID overlap pair',
                csv=output.with_suffix('.csv').name)


def kv_evidence(path, output):
    import matplotlib.pyplot as plt
    audit = audit_trace(path)
    events = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    events = [event for event in events if event.get('event') in ('acquire', 'release')]
    models = sorted({event['lease']['model'] for event in events})
    if len(models) != 2:
        raise ValueError('Expected exactly two model layouts')
    layouts, history = {}, {model: [0] for model in models}
    live = dict.fromkeys(models, 0)
    with (output / 'kv-events.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['event_index', 'event', *events[0]['lease']])
        writer.writeheader()
        for index, event in enumerate(events, 1):
            lease = event['lease']
            writer.writerow(dict(event_index=index, event=event['event'], **lease))
            model = lease['model']
            _, layers, half = lease['layout'].split(':')
            page = 2 * int(layers) * int(half)
            layouts[model] = dict(layout=lease['layout'], layers=int(layers),
                                 single_layer_k_or_v_bytes=int(half), all_layer_page_bytes=page,
                                 observed_chunk_bytes=lease['bytes'], pages_per_chunk=lease['bytes']//page)
            live[model] += lease['bytes'] * (1 if event['event'] == 'acquire' else -1)
            for item in models:
                history[item].append(live[item] / (1 << 20))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.65]})
    colors = ['#1877b9', '#e18026']
    sizes = [layouts[model]['all_layer_page_bytes']/1024 for model in models]
    axes[0].barh(models, sizes, color=colors)
    for row, size in enumerate(sizes):
        axes[0].text(size+8, row, f'{size:.0f} KiB', va='center')
    axes[0].set(xlim=(0, max(sizes)*1.35), xlabel='All-layer K + V per 16-token page', title='Different tensor layouts (BF16)')
    for model, color in zip(models, colors):
        axes[1].step(range(len(events)+1), history[model], where='post', label=model, color=color)
    axes[1].set(xlabel='Owner allocation / release event index (not time)', ylabel='Live leased MiB',
                title=f"Shared arena; {audit['acquires']} acquisitions / {audit['releases']} releases")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=.2)
    fig.text(.02, .02, f"Same physical byte ranges reused across models: {audit['acquire_events_reusing_other_models_bytes']} acquisitions | Final live leases: 0", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output / 'shared-kv.svg', metadata={'Date': None})
    plt.close(fig)
    return dict(audit=audit, layouts=layouts)


def sm_activity_figure(path, output):
    import matplotlib.pyplot as plt
    report = json.loads(path.read_text())
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1, 1, 1.2]})
    for index, backend in enumerate(('process', 'mps_process')):
        run = next(item for item in report['runs'] if item['backend'] == backend)
        with (path.parent / f'sm-{backend}-0.csv').open() as stream:
            bins = list(csv.DictReader(stream))
        reference = int(bins[0]['start_ns'])
        for row, field in enumerate(('process_a_distinct_ticks', 'process_b_distinct_ticks')):
            spans = [((int(item['start_ns'])-reference)/1e6,
                      (int(item['end_ns'])-int(item['start_ns']))/1e6)
                     for item in bins if int(item[field]) > 0]
            axes[index].broken_barh(spans, (row-.25, .5), facecolors=['#1877b9', '#e18026'][row])
        axes[index].set(yticks=[0, 1], yticklabels=[f"PID {c['pid']}" for c in run['clients']],
                        title='MPS OFF: device progress samples' if index == 0 else 'MPS ON: device progress samples',
                        xlabel='Time from first GPU sample (ms)', ylim=(-.6, 1.6))
        axes[index].grid(axis='x', alpha=.2)
        axes[index].invert_yaxis()
    values = [r['same_tick_different_sm_count'] for r in report['runs'] if r['backend'] == 'mps_process']
    axes[2].bar(range(len(values)), values, color='#24855e', width=.5)
    for index, value in enumerate(values):
        axes[2].text(index, value, f'{value:,}', ha='center', va='bottom')
    controls = [r['same_tick_different_sm_count'] for r in report['runs'] if r['backend'] == 'process']
    granularity = sorted({value for run in report['runs'] for value in run['observed_timestamp_delta_gcd_ns']})
    axes[2].set(xticks=range(len(values)), xticklabels=[f'Round {i+1}' for i in range(len(values))],
                ylim=(0, max(values)*1.25), ylabel='Coincident clock samples',
                title=f'Same GPU clock reading, different recorded SM IDs (MPS OFF: {", ".join(map(str, controls))})')
    fig.text(.02, .02, f'Top: bins with device progress (20 us). Bottom: exact clock equality, observed delta GCD {granularity} ns.\nSM IDs are recorded per context; no physical SM partition or occupancy percentage is inferred.', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(output / 'sm-activity.svg', metadata={'Date': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-sqlite', 'probe-report', 'models-sqlite', 'registry', 'owner-trace', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sm-activity-report', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    probe = json.loads(args.probe_report.read_text())
    registry = json.loads(args.registry.read_text())
    if not probe['mps_clients_registered'] or registry['backend'] != 'mps_process' or not registry['mps_server']:
        raise ValueError('MPS registration evidence is required')
    report = dict(method='Nsight Systems CUDA software GPU kernel activity; cross-PID interval intersection',
                  limitations=['Kernel interval overlap alone does not exclude context time slicing; see SM activity on/off control.',
                               'No SM occupancy percentage or throughput speedup is inferred.',
                               'Profiled timings include tracing overhead.',
                               'Models share capacity, not KV values or model weights.'])
    for name, path, labels in (
        ('probe', args.probe_sqlite, {client['pid']: f'Probe {i+1}' for i, client in enumerate(probe['clients'])}),
        ('models', args.models_sqlite, {model['worker_pid']: model['id'] for model in registry['models']})):
        kernels = read_kernels(path, labels)
        summary = overlap_summary(kernels)
        summary['figure_window'] = timeline(kernels, labels, summary, args.output_dir / f'{name}-timeline.svg',
                                           'MPS: two process kernels on the same GPU' if name == 'probe'
                                           else 'MPS: Qwen2.5-0.5B + 1.5B inference kernels', name == 'probe')
        report[name] = summary
    report['kv'] = kv_evidence(args.owner_trace, args.output_dir)
    if args.sm_activity_report:
        sm_activity_figure(args.sm_activity_report, args.output_dir)
    report['inputs'] = [fingerprint(path) for path in (args.probe_sqlite, args.probe_report,
                                                    args.models_sqlite, args.registry, args.owner_trace)]
    write_json(args.output_dir / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
