#!/usr/bin/env python3
"""Exercise real process shutdown with CUDA IPC; MPS control is tested separately."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from deploy_multi_model import ROOT, rpc
from validate_multi_model import request, ready_http


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-mps')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenario', choices=('drain', 'cancel', 'owner-failure'), default='drain')
    parser.add_argument('--port', type=int, default=18083)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config.update(backend='process', listen_port=args.port, shutdown_timeout_s=20)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    runtime = directory / 'domain'
    log = open(directory / 'supervisor.log', 'w')
    supervisor = subprocess.Popen([os.sys.executable, str(ROOT / 'tools/serving/deploy_multi_model.py'),
        '--config', str(directory / 'config.json'), '--build-dir', str(args.build_dir),
        '--runtime-dir', str(runtime)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    registry = None
    try:
        ready_http(args.port, supervisor)
        registry = json.loads((runtime / 'registry.json').read_text())
        endpoint = f'ipc://{runtime}/kv.sock'
        health = rpc(endpoint, {'type': 'health'})
        identity = dict(epoch=health['epoch'], model='shutdown-guard-test')
        rpc(endpoint, dict(type='register', hard_limit=4096, **identity))
        lease = rpc(endpoint, dict(type='acquire', layout='unused', bytes=4096,
                                   alignment=256, **identity))['lease']
        try:
            rpc(endpoint, dict(type='shutdown', epoch=health['epoch']))
        except RuntimeError as error:
            assert str(error) == 'arena_has_live_leases'
        else:
            raise AssertionError('owner accepted shutdown with a live lease')
        rpc(endpoint, dict(type='release', lease=lease, gpu_complete=True, **identity))
        report = dict(scenario=args.scenario, owner_refuses_live_leases=True,
                      epoch=health['epoch'])
        if args.scenario == 'owner-failure':
            os.kill(registry['arena_owner_pid'], signal.SIGKILL)
        else:
            model = config['models'][1]['id']
            req = request(args.port, 'Write a long numbered list of 100 tips for learning Python.',
                          model, stream=True, tokens=128)
            with urllib.request.urlopen(req, timeout=60) as response:
                first = response.readline()
                assert first.startswith(b'data: '), first
                active = rpc(endpoint, {'type': 'metrics'})
                assert active['lease_count'] > 0, active
                report['leases_when_stop_requested'] = active['lease_count']
                if args.scenario == 'drain':
                    os.killpg(supervisor.pid, signal.SIGTERM)
                else:
                    worker = next(m for m in registry['models'] if m['id'] == model)
                    reply = rpc(worker['endpoint'], dict(type='shutdown', mode='cancel'), timeout_ms=20000)
                    assert reply['gpu_synchronized']
                    report['cancel_acknowledged_after_gpu_sync'] = True
                remaining = response.read()
                if args.scenario == 'drain':
                    assert b'[DONE]' in remaining, remaining[-500:]
                    report['active_http_stream_completed'] = True
        code = supervisor.wait(timeout=90)
        shutdown = json.loads((runtime / 'shutdown.json').read_text())
        report.update(supervisor_exit=code, shutdown=shutdown)
        if args.scenario == 'drain':
            assert code in (0, 130) and shutdown['graceful'], report
            events = [e for e in shutdown['events'] if e['event'] != 'shutdown_phase']
            assert events[0]['event'] == 'api_drain_requested'
            assert events[-1]['event'] == 'owner_stopped'
            arena = next(e for e in events if e['event'] == 'arena_before_exit')
            assert arena['lease_count'] == 0 and arena['free_bytes'] == arena['arena_bytes']
            assert len([e for e in events if e['event'] == 'worker_stopped']) == 2
        else:
            assert code != 0 and not shutdown['graceful'], report
        pids = [m['worker_pid'] for m in registry['models']] + [registry['arena_owner_pid']]
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise AssertionError(f'GPU process survived domain stop: {pid}')
        report['all_gpu_processes_exited'] = True
        (directory / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGTERM)
            try:
                supervisor.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(supervisor.pid, signal.SIGKILL)
                supervisor.wait()
                if registry:
                    for pid in [m['worker_pid'] for m in registry['models']] + [registry['arena_owner_pid']]:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        log.close()


if __name__ == '__main__':
    main()
