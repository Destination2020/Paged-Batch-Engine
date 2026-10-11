"""Ordered shutdown phases for a domain sharing CUDA IPC allocations.

Transport and MPS control are injected so each fault path can be tested without
GPU processes. A live importer always prevents exporter reclamation.
"""
import enum
import json
import os
import signal
import subprocess
import time


class ShutdownPhase(enum.IntEnum):
    INSPECT = 0
    API = 1
    WORKERS = 2
    AUXILIARY = 3
    OWNER = 4
    COMPLETE = 5


class ShutdownCoordinator:
    def __init__(self, domain, rpc, terminate_mps_client):
        self.directory = domain.directory
        self.env = domain.env
        self.children = domain.children
        self.logs = domain.logs
        self.service = domain.service
        self.timeout = self.service['timeout']
        self.rpc = rpc
        self.terminate_mps_client = terminate_mps_client
        self.report = dict(backend=self.service['backend'], events=[], graceful=True,
                           mps_recovery_required=False)
        self.phase = ShutdownPhase.INSPECT

    def record(self, event, **fields):
        self.report['events'].append(dict(event=event, monotonic_ns=time.monotonic_ns(), **fields))

    def fault(self, reason):
        self.report['graceful'] = False
        if self.service['backend'] == 'mps_process':
            self.report['mps_recovery_required'] = True
        self.record('fault', reason=str(reason))

    @staticmethod
    def send_signal(child, sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def advance(self, phase):
        if phase.value != self.phase.value + 1:
            raise RuntimeError('invalid shutdown phase transition')
        self.phase = phase
        self.record('shutdown_phase', phase=phase.name.lower())

    def run(self):
        self.inspect()
        for phase, operation in ((ShutdownPhase.API, self.stop_api),
                                 (ShutdownPhase.WORKERS, self.stop_workers),
                                 (ShutdownPhase.AUXILIARY, self.stop_auxiliary),
                                 (ShutdownPhase.OWNER, self.stop_owner)):
            self.advance(phase)
            operation()
        self.advance(ShutdownPhase.COMPLETE)
        for log in self.logs:
            log.close()
        (self.directory / 'shutdown.json').write_text(json.dumps(self.report, indent=2) + '\n')
        return self.report

    def force(self, name, child, gpu):
        if child.poll() is not None:
            return
        if gpu and self.service['backend'] == 'mps_process':
            try:
                server = self.terminate_mps_client(self.service['binary'], child.pid, self.env)
                self.record('mps_client_terminated', name=name, pid=child.pid, server=server)
            except Exception as error:
                self.fault(error)
                self.record('gpu_termination_unconfirmed', name=name, pid=child.pid)
        # Forced exits are always marked non-graceful. No orphan lease is
        # recycled for surviving clients; the entire domain is discarded.
        self.fault(f'forced process exit: {name}')
        if child.poll() is None:
            self.send_signal(child, signal.SIGKILL)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.fault(f'{name} survived SIGKILL; retain exporter for live importers')
            return
        self.record('forced_exit', name=name, pid=child.pid)

    def inspect(self):
        if self.service.get('failure'):
            self.fault(self.service['failure'])
        self.failed = [(name, child) for name, child in self.children if child.poll() is not None]
        for name, child in self.failed:
            self.fault(f'{name} exited unexpectedly ({child.returncode})')

    def stop_api(self):
        api = self.service['api']
        if api is not None and api.poll() is None:
            if self.failed:
                self.force('api', api, False)
            else:
                self.record('api_drain_requested', pid=api.pid)
                self.send_signal(api, signal.SIGTERM)
                try:
                    if api.wait(timeout=self.timeout) != 0:
                        raise RuntimeError('API drain failed')
                    self.record('api_drained', pid=api.pid)
                except Exception as error:
                    self.fault(error)
                    self.force('api', api, False)

    def stop_workers(self):
        for name, child, endpoint in self.service['workers']:
            if child.poll() is not None:
                self.fault(f'{name} exited before shutdown acknowledgement ({child.returncode})')
                continue
            try:
                mode = 'drain' if self.report['graceful'] else 'cancel'
                reply = self.rpc(endpoint, dict(type='shutdown', mode=mode), timeout_ms=int(self.timeout * 1000))
                if not reply.get('gpu_synchronized'):
                    raise RuntimeError('worker did not confirm CUDA synchronization')
                if child.wait(timeout=self.timeout) != 0:
                    raise RuntimeError('worker failed after synchronization')
                self.record('worker_stopped', name=name, pid=child.pid, mode=mode, gpu_synchronized=True)
            except Exception as error:
                self.fault(error)
                self.force(name, child, True)

    def stop_auxiliary(self):
        # Includes an owner or worker whose startup failed before registration.
        known = {child.pid for _, child, _ in self.service['workers']}
        for name, child in self.children:
            if name not in ('api', 'kv-owner', 'ipc-event-owner') and child.pid not in known:
                self.force(name, child, True)
        for name, child in self.children:
            if name == 'ipc-event-owner' and child.poll() is None:
                if any(n not in ('api', 'kv-owner', 'ipc-event-owner') and p.poll() is None
                       for n, p in self.children):
                    self.fault('IPC event importer survived; retain exporter')
                    self.report['owner_quarantined'] = True
                else:
                    self.force(name, child, True)

    def stop_owner(self):
        for name, owner in self.children:
            if name != 'kv-owner':
                continue
            if owner.poll() is not None:
                self.fault(f'owner exited before importer cleanup ({owner.returncode})')
                continue
            live_importers = [child.pid for child_name, child in self.children
                              if child_name not in ('api', 'kv-owner') and child.poll() is None]
            if live_importers:
                self.fault('GPU importers still alive; owner must remain quarantined')
                self.report['owner_quarantined'] = True
                self.record('owner_quarantined', pid=owner.pid, live_importers=live_importers)
                continue
            try:
                metrics = self.rpc(self.service['owner_endpoint'], {'type': 'metrics'})
                self.record('arena_before_exit', free_bytes=metrics['free_bytes'],
                       arena_bytes=metrics['bytes'], lease_count=metrics['lease_count'])
                reply = self.rpc(self.service['owner_endpoint'], dict(type='shutdown', epoch=metrics['epoch']))
                if not reply.get('arena_freed') or not reply.get('gpu_synchronized'):
                    raise RuntimeError('owner did not confirm arena release')
                if owner.wait(timeout=self.timeout) != 0:
                    raise RuntimeError('owner shutdown failed')
                self.record('owner_stopped', pid=owner.pid, arena_freed=True)
            except Exception as error:
                self.fault(error)
                self.force(name, owner, True)
