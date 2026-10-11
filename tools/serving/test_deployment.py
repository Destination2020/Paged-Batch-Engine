import copy
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
