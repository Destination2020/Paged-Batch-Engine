#!/usr/bin/env python3
"""MPS on/off device instruction sampling. Requires numpy; no model weights needed.

The diagnostic samples %globaltimer and %smid after fixed arithmetic work. It
reports coincident device clock readings, unlike kernel lifetime overlap which
can also occur under time slicing. Raw binary samples remain in the run directory.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from deploy_multi_model import control, deployment_environment, gpu_info, mps_control_path, registered_server

SAMPLE = np.dtype([('time', '<u8'), ('sm', '<u4'), ('value', '<u4')])


def analyze(directory, clients):
    arrays, fingerprints = [], []
    for index, client in enumerate(clients):
        path = directory / f'client-{index}.bin'
        data = np.fromfile(path, dtype=SAMPLE)
        if len(data) != client['blocks'] * client['samples_per_block']:
            raise ValueError('Incomplete device sample file')
        blocks = data.reshape(client['blocks'], client['samples_per_block'])
        if any(np.any(block['time'][1:] < block['time'][:-1]) for block in blocks):
            raise ValueError('Non-monotonic device clock')
        times, unique = np.unique(data['time'], return_index=True)
        arrays.append((times, data['sm'][unique]))
        fingerprints.append(dict(file=path.name, bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    common, first, second = np.intersect1d(arrays[0][0], arrays[1][0], assume_unique=True, return_indices=True)
    different = arrays[0][1][first] != arrays[1][1][second]
    # This is a conservative count: when several blocks in one process sample
    # the same tick we retain only one SM, so additional matches may be omitted.
    witnesses = [dict(globaltimer_ns=int(common[i]), pid_a=clients[0]['pid'],
                      sm_a=int(arrays[0][1][first[i]]), pid_b=clients[1]['pid'],
                      sm_b=int(arrays[1][1][second[i]]))
                 for i in np.flatnonzero(different)[:32]]
    start = min(int(item[0][0]) for item in arrays)
    end = max(int(item[0][-1]) for item in arrays)
    bin_ns = 20000
    edges = np.arange(start, end + 2 * bin_ns, bin_ns, dtype=np.uint64)
    counts = [np.histogram(item[0], bins=edges)[0] for item in arrays]
    with (directory / 'activity-bins.csv').open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['start_ns', 'end_ns', 'process_a_distinct_ticks', 'process_b_distinct_ticks'])
        writer.writerows((int(edges[i]), int(edges[i+1]), int(counts[0][i]), int(counts[1][i]))
                         for i in range(len(edges)-1))
    return dict(clients=clients, raw_samples=fingerprints,
                sample_format='little endian: globaltimer uint64, smid uint32, accumulator bits uint32',
                same_tick_count=int(len(common)), same_tick_different_sm_count=int(np.sum(different)),
                witnesses=witnesses, histogram_bin_ns=bin_ns,
                histogram_both_processes_nonempty=int(np.sum((counts[0] > 0) & (counts[1] > 0))),
                observed_timestamp_delta_gcd_ns=[int(np.gcd.reduce(np.diff(item[0]))) for item in arrays],
                sampled_sm_ids=[sorted(map(int, np.unique(item[1]))) for item in arrays],
                first_sample_ns=start, last_sample_ns=end,
                interpretation='Coincident device clock readings on different SMs; resolution is hardware dependent. Not an occupancy percentage.')


def run(binary, directory, backend, gpu):
    directory.mkdir(parents=True, mode=0o700)
    env = deployment_environment(os.environ, gpu['uuid'], directory, backend)
    daemon = mps_control_path()
    owned = False
    children, logs = [], []
    try:
        if backend == 'mps_process':
            if not daemon:
                raise RuntimeError('MPS control executable missing')
            for kind in ('pipe', 'log'):
                path = directory / ('mps-' + kind)
                path.mkdir(mode=0o700)
                env['CUDA_MPS_' + kind.upper() + '_DIRECTORY'] = str(path)
            subprocess.run([daemon, '-d'], check=True, env=env, timeout=10)
            owned = True
            env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '50'
        barrier = directory / 'go'
        for index in range(2):
            log = (directory / f'client-{index}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen([str(binary), str(barrier), str(directory / f'client-{index}')],
                                             stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + 60
        while not all((directory / f'client-{i}.ready').exists() for i in range(2)):
            if any(child.poll() is not None for child in children):
                raise RuntimeError('Probe exited before readiness; inspect client logs')
            if time.monotonic() > deadline:
                raise TimeoutError('Probe readiness timeout')
            time.sleep(.01)
        server = registered_server(daemon, [child.pid for child in children], env) if owned else None
        if owned and server is None:
            raise RuntimeError('Clients are not registered with the same MPS server')
        barrier.touch()
        for child in children:
            if child.wait(timeout=60):
                raise RuntimeError('GPU sampling failed; inspect client logs')
        clients = [json.loads((directory / f'client-{i}.log').read_text().splitlines()[-1]) for i in range(2)]
        result = analyze(directory, clients)
        result.update(backend=backend, mps_server=server, gpu=gpu)
        (directory / 'activity.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        for log in logs:
            log.close()
        if owned:
            control(daemon, 'quit', env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    gpu = gpu_info(args.gpu_uuid)
    results = []
    for index in range(args.rounds):
        for backend in ('process', 'mps_process'):
            result = run(args.binary.resolve(), args.output_dir.resolve() / f'{backend}-{index}', backend, gpu)
            results.append(result)
            print(f"{backend} round {index}: same-tick/different-SM witnesses = {result['same_tick_different_sm_count']}", flush=True)
    summary = dict(rounds=args.rounds, runs=results,
                   mps_all_rounds_have_witnesses=all(r['same_tick_different_sm_count'] > 0 for r in results if r['backend'] == 'mps_process'),
                   control_all_rounds_have_no_witnesses=all(r['same_tick_count'] == 0 for r in results if r['backend'] == 'process'))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['mps_all_rounds_have_witnesses'] or not summary['control_all_rounds_have_no_witnesses']:
        raise RuntimeError('MPS/control separation not established; retain results and investigate')


if __name__ == '__main__':
    main()
