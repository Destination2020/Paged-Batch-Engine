import copy
import json
import subprocess
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location("deploy", Path(__file__).with_name("deploy_multi_model.py"))
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(arena_mib=512, models=[
            dict(id="small", private_memory_mib=1536, kv_hard_limit_mib=512,
                 weights="pending-small.bin", tokenizer="pending-small.json"),
            dict(id="large", private_memory_mib=3584, kv_hard_limit_mib=512,
                 weights="pending-large.bin", tokenizer="pending-large.json")])

    def test_environment_does_not_inherit_client_resource_caps(self):
        source = {'PATH': '/bin', 'CUDA_MPS_PIPE_DIRECTORY': '/external',
                  'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': '1',
                  'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT': '0=1M'}
        env = deploy.deployment_environment(source, 'GPU-test', Path('/domain'), 'mps_process')
        self.assertEqual(env['CUDA_MPS_PIPE_DIRECTORY'], '/external')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        self.assertNotIn('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', env)
        self.assertNotIn('CUDA_MPS_PINNED_DEVICE_MEM_LIMIT', env)
        env = deploy.deployment_environment(source, 'GPU-test', Path('/domain'), 'process')
        self.assertEqual(env['CUDA_MPS_PIPE_DIRECTORY'], '/domain/mps-disabled')

    def test_arena_is_counted_once_not_for_each_importer(self):
        self.assertEqual(deploy.validate_config(self.config, require_files=False), 5632)

    def test_duplicate_identity_and_invalid_quota(self):
        for change in (dict(id="small"), dict(kv_hard_limit_mib=513),
                       dict(mps_active_thread_percentage=0), dict(private_memory_mib=-1)):
            config = copy.deepcopy(self.config)
            config["models"][1].update(change)
            with self.assertRaises(ValueError):
                deploy.validate_config(config, require_files=False)

    def test_mps_requires_all_clients_on_one_server(self):
        replies = {"get_server_list": "123\n456\n", "get_client_list 123": "11\n12\n",
                   "get_client_list 456": "13\n14\n"}
        with patch.object(deploy, "control", side_effect=lambda _, cmd, env: replies[cmd]):
            self.assertEqual(deploy.registered_server("mps", [11, 12], {}), 123)
            self.assertIsNone(deploy.registered_server("mps", [11, 13], {}))
            self.assertIsNone(deploy.registered_server("mps", [11, 12, 13], {}))

    def test_mps_termination_requires_explicit_cuda_success(self):
        with patch.object(deploy, 'registered_server', return_value=123):
            with patch.object(deploy, 'control', return_value='0') as command:
                self.assertEqual(deploy.terminate_mps_client('mps', 11, {}), 123)
                command.assert_called_once_with('mps', 'terminate_client 123 11', {})
            for response in ('', '1', 'Server not found'):
                with patch.object(deploy, 'control', return_value=response):
                    with self.assertRaises(RuntimeError):
                        deploy.terminate_mps_client('mps', 11, {})

    def test_shutdown_timeout_validation(self):
        for timeout in (0, -1, 301, float('nan'), '30'):
            config = copy.deepcopy(self.config)
            config['shutdown_timeout_s'] = timeout
            with self.assertRaises(ValueError):
                deploy.validate_config(config, require_files=False)

    def test_service_shutdown_order_and_fault_termination(self):
        for fail_worker in (False, True):
            with self.subTest(fail_worker=fail_worker), tempfile.TemporaryDirectory() as path:
                events = []
                def process(pid):
                    child = Mock(pid=pid, returncode=None)
                    child.poll.side_effect = lambda: child.returncode
                    def wait(**kwargs):
                        child.returncode = 0
                        events.append(('wait', pid))
                        return 0
                    child.wait.side_effect = wait
                    return child
                owner, worker, api = [process(pid) for pid in (11, 12, 13)]
                domain = deploy.Domain(path, {})
                domain.children = [('kv-owner', owner), ('model', worker), ('api', api)]
                domain.manage_service(dict(backend='mps_process', shutdown_timeout_s=1), 'mps', 'owner')
                domain.service.update(workers=[('model', worker, 'worker')], api=api)
                def rpc(endpoint, body, **kwargs):
                    events.append((body['type'], endpoint))
                    if endpoint == 'worker' and fail_worker:
                        raise TimeoutError('injected worker stall')
                    if body['type'] == 'metrics':
                        return dict(bytes=1024, free_bytes=1024, lease_count=0, epoch=1)
                    return dict(gpu_synchronized=True, arena_freed=True)
                def terminate(binary, pid, env):
                    events.append(('terminate_gpu', pid))
                    return 123
                with patch.object(deploy, 'rpc', side_effect=rpc), \
                     patch.object(deploy, 'terminate_mps_client', side_effect=terminate), \
                     patch.object(deploy.os, 'killpg', side_effect=lambda pid, sig: events.append(('signal', pid))):
                    report = domain.stop()
                    domain.stop() # cleanup is idempotent
                self.assertEqual(report['graceful'], not fail_worker)
                self.assertEqual(report['mps_recovery_required'], fail_worker)
                self.assertLess(events.index(('wait', 13)), events.index(('shutdown', 'worker')))
                self.assertLess(events.index(('wait', 12)), events.index(('shutdown', 'owner')))
                if fail_worker:
                    self.assertLess(events.index(('terminate_gpu', 12)), events.index(('signal', 12)))
                self.assertTrue(json.loads((Path(path) / 'shutdown.json').read_text())['events'])

    def test_exporter_is_quarantined_if_importer_cannot_exit(self):
        with tempfile.TemporaryDirectory() as path:
            owner = Mock(pid=11, returncode=None)
            owner.poll.return_value = None
            worker = Mock(pid=12, returncode=None)
            worker.poll.return_value = None
            worker.wait.side_effect = subprocess.TimeoutExpired('worker', 10)
            domain = deploy.Domain(path, {})
            domain.children = [('kv-owner', owner), ('model', worker)]
            domain.manage_service(dict(backend='mps_process', shutdown_timeout_s=1), 'mps', 'owner')
            domain.service['workers'] = [('model', worker, 'worker')]
            with patch.object(deploy, 'rpc', side_effect=TimeoutError('stalled')) as rpc, \
                 patch.object(deploy, 'terminate_mps_client', side_effect=RuntimeError('MPS unresponsive')), \
                 patch.object(deploy.os, 'killpg') as kill:
                report = domain.stop()
            self.assertTrue(report['owner_quarantined'])
            self.assertTrue(report['mps_recovery_required'])
            self.assertFalse(report['graceful'])
            self.assertTrue(any(e['event'] == 'gpu_termination_unconfirmed' for e in report['events']))
            self.assertEqual([call.args[0] for call in rpc.call_args_list], ['worker'])
            self.assertEqual([call.args[0] for call in kill.call_args_list], [12])
            owner.wait.assert_not_called()

    def test_weight_shape_preflight_rejects_wrong_dtype_context_and_kv_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(struct.pack("<9i", 896, 4864, 24, 14, 2, 151936,
                                         32768, 0x44545950, 4))
            config = copy.deepcopy(self.config)
            for model in config["models"]:
                model["weights"] = str(path)
            self.assertEqual(deploy.validate_config(config, require_files=False), 5632)
            for change in (dict(max_prompt_tokens=32768), dict(kv_hard_limit_mib=1)):
                invalid = copy.deepcopy(config)
                invalid["models"][0].update(change)
                with self.assertRaises(ValueError):
                    deploy.validate_config(invalid, require_files=False)
            path.write_bytes(struct.pack("<9i", 896, 4864, 24, 14, 2, 151936,
                                         32768, 0x44545950, 0))
            with self.assertRaises(ValueError):
                deploy.validate_config(config, require_files=False)

if __name__ == "__main__":
    unittest.main()
