#!/usr/bin/env python3
"""Prove cross-process reuse of the same CUDA KV bytes without CUDA MPS.

The 21 MiB arena holds exactly 112 Qwen 0.5B pages or 48 Qwen 1.5B
pages. Both clients remain alive throughout A-full -> B-denied -> A-release
-> B-full -> B-release cycles. Pins and pending CUDA fences block reuse.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile

from deploy_multi_model import Domain, ROOT, rpc


def audit_trace(path):
    active, released, examples = {}, [], []
    epochs = set()
    acquired = returned = reused = peak = 0
    for line in Path(path).read_text().splitlines():
        if not line.startswith('{'):
            continue
        event = json.loads(line)
        if event.get('event') not in ('acquire', 'release'):
            continue
        lease = event['lease']
        epochs.add(lease['epoch'])
        identity = (lease['epoch'], lease['id'], lease['generation'])
        def overlap(other):
            return max(0, min(lease['offset'] + lease['bytes'], other['offset'] + other['bytes'])
                       - max(lease['offset'], other['offset']))
        if event['event'] == 'acquire':
            if identity in active or any(overlap(other) for other in active.values()):
                raise AssertionError('overlapping live leases or duplicate identity')
            matches = [old for old in released if old['model'] != lease['model'] and overlap(old)]
            if matches:
                reused += 1
                if len(examples) < 8:
                    old = matches[-1]
                    examples.append(dict(previous_model=old['model'], next_model=lease['model'],
                                         released_lease=old['id'], new_lease=lease['id'],
                                         offset=max(old['offset'], lease['offset']),
                                         overlapping_bytes=overlap(old)))
            active[identity] = lease
            acquired += 1
            peak = max(peak, sum(item['bytes'] for item in active.values()))
        else:
            if active.pop(identity, None) != lease:
                raise AssertionError('release identity mismatch or duplicate release')
            released.append(lease)
            returned += 1
    if len(epochs) != 1 or not acquired or active or not reused:
        raise AssertionError('expected one owner epoch, no leaks and cross-model physical reuse')
    return dict(acquires=acquired, releases=returned, peak_leased_bytes=peak,
                live_leases_at_end=len(active), live_intervals_never_overlap=True,
                acquire_events_reusing_other_models_bytes=reused, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-mps')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--audit-trace', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.audit_trace:
        print(json.dumps(audit_trace(args.audit_trace), indent=2))
        return
    if args.rounds < 1:
        raise ValueError('rounds must be positive')
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix='pbe-kv-reuse-'))
    directory.mkdir(exist_ok=True, mode=0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS_')}
    env['PBE_SHARED_KV_TRACE'] = '1'
    domain = Domain(directory, env)
    endpoint = f'ipc://{directory}/kv.sock'
    arena_bytes = 21 << 20
    records = []
    def receive(process):
        if not select.select([process.stdout], [], [], 30)[0]:
            raise TimeoutError('probe response timeout')
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('probe exited; inspect client log')
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result)
        return result
    def send(process, command, **kwargs):
        process.stdin.write(json.dumps(dict(command=command, **kwargs)) + '\n')
        process.stdin.flush()
        return receive(process)
    def state(free):
        result = rpc(endpoint, {'type': 'metrics'})
        assert result['free_bytes'] == free, result
        if free == arena_bytes:
            assert result['lease_count'] == 0, result
        return {k: result[k] for k in ('free_bytes', 'lease_count', 'epoch')}
    try:
        domain.spawn('kv-owner', [args.build_dir / 'demo/shared_kv_owner', endpoint, 21])
        domain.ready(endpoint)
        clients = []
        for name, layers, head in [('qwen25-0.5b-layout', 24, 64), ('qwen25-1.5b-layout', 28, 128)]:
            log = open(directory / f'{name}.log', 'w')
            domain.logs.append(log)
            client = subprocess.Popen([str(args.build_dir / 'demo/shared_kv_reuse_probe'), endpoint,
                                       name, str(layers), str(head)], env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                                      start_new_session=True)
            domain.children.append((name, client))
            receive(client)
            clients.append(client)
        a, b = clients
        for iteration in range(args.rounds):
            first = send(a, 'fill', pattern=20 + iteration)
            assert first['owned_bytes'] == arena_bytes and first['all_gpu_bytes_verified']
            state(0)
            assert send(b, 'fill', pattern=100)['pages'] == 0
            send(a, 'pin_and_release')
            state(0)
            assert send(b, 'fill', pattern=100)['pages'] == 0
            assert send(a, 'unpin_deferred')['gpu_fence_pending']
            state(0)
            assert send(b, 'fill', pattern=100)['pages'] == 0
            send(a, 'drain')
            state(arena_bytes)
            second = send(b, 'fill', pattern=100 + iteration)
            assert second['owned_bytes'] == arena_bytes and second['all_gpu_bytes_verified']
            state(0)
            send(b, 'release_first_layer')
            state(0)
            assert send(a, 'fill', pattern=20)['pages'] == 0
            send(b, 'release')
            final = state(arena_bytes)
            records.append(dict(round=iteration, a=first, b=second, final=final,
                                b_denied_while_a_live=True, b_denied_while_pinned=True,
                                b_denied_before_gpu_fence=True, a_denied_after_partial_layer_release=True))
        send(a, 'rollback')
        state(arena_bytes)
        send(b, 'rollback')
        state(arena_bytes)
        report = dict(backend='process', arena_bytes=arena_bytes,
                      persistent_client_pids=[client.pid for client in clients],
                      rounds=records, unconsumed_layer_rollback=True,
                      trace_audit=audit_trace(directory / 'kv-owner.log'))
        (directory / 'reuse.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['trace_audit'], indent=2))
        print(f'Full report: {directory}/reuse.json')
    finally:
        domain.stop()


if __name__ == '__main__':
    main()
