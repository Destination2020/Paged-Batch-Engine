#!/usr/bin/env python3
"""Measure local/shared KV management with identical BF16 geometry; no MPS.

Four all-layer pages per sample, three warmup samples, optional frozen library.
GPU writes are synthetic memset payloads and do not measure attention throughput.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from deploy_multi_model import Domain, ROOT, rpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-mps')
    parser.add_argument('--library-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error('iterations and repeats must be positive')
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS_')}
    env['PBE_SHARED_KV_TRACE'] = '0'
    if args.library_dir:
        env['LD_LIBRARY_PATH'] = str(args.library_dir.resolve()) + ':' + env.get('LD_LIBRARY_PATH', '')
    domain = Domain(directory, env)
    endpoint = f'ipc://{directory}/kv.sock'
    results = []
    try:
        domain.spawn('owner', [args.build_dir / 'demo/shared_kv_owner', endpoint, 84])
        domain.ready(endpoint)
        for repeat in range(args.repeats):
            for layers, head, capacity in [(24, 64, 448), (28, 128, 192)]:
                for held in (0, int(capacity * .9), capacity):
                    for backend in (('private', 'shared') if repeat % 2 == 0 else ('shared', 'private')):
                        result = subprocess.run([str(args.build_dir / 'demo/shared_kv_bench'),
                            endpoint if backend == 'shared' else 'private', str(layers), str(head),
                            str(held), str(args.iterations)], env=env, capture_output=True,
                            text=True, timeout=180)
                        name = f'{repeat}-{layers}-{held}-{backend}'
                        (directory / (name + '.log')).write_text(result.stderr)
                        if result.returncode:
                            raise RuntimeError(f'{name}: {result.stderr}')
                        record = json.loads(result.stdout)
                        record.update(repeat=repeat, backend=backend)
                        results.append(record)
                        metrics = rpc(endpoint, {'type': 'metrics'})
                        assert metrics['lease_count'] == 0 and metrics['free_bytes'] == 84 << 20
                        print(name, flush=True)
        summaries = {}
        for result in results:
            key = f"{result['layers']}-{result['held_pages']}-{result['backend']}"
            bucket = summaries.setdefault(key, {})
            for sample in result['samples']:
                for metric, value in sample.items():
                    bucket.setdefault(metric, []).append(value)
        for bucket in summaries.values():
            for metric, values in bucket.items():
                bucket[metric] = dict(p50=statistics.median(values),
                    p95=sorted(values)[math.ceil(.95 * len(values)) - 1], samples=len(values))
        report = dict(arena_bytes=84 << 20, iterations=args.iterations, repeats=args.repeats,
                      library_dir=str(args.library_dir) if args.library_dir else None,
                      summary=summaries, raw=results)
        (directory / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        domain.stop()


if __name__ == '__main__':
    main()
