import json
from pathlib import Path
import struct
import tempfile
import unittest

from deploy_split_pd import validate


class SplitPDConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config = json.loads(
            (Path(__file__).resolve().parents[2] / "configs/qwen25-1.5b-2p1d.json").read_text()
        )
        weights = root / "model.bin"
        weights.write_bytes(struct.pack("<9i", 1536, 8960, 28, 12, 2, 151936, 32768, 0x44545950, 4))
        tokenizer = root / "tokenizer.json"
        tokenizer.write_text("{}")
        self.config.update(weights=str(weights), tokenizer=str(tokenizer))

    def test_real_geometry_budget(self):
        self.assertEqual(validate(self.config)["kv_bytes_per_token"], 28672)

    def test_decode_promises_cannot_exceed_physical_quota(self):
        self.config["short_kv_tokens"] = 65536
        with self.assertRaisesRegex(ValueError, "token promises"):
            validate(self.config)

    def test_owner_must_cover_all_live_role_quotas(self):
        self.config["arena_mib"] = 2048
        with self.assertRaisesRegex(ValueError, "alignment slack"):
            validate(self.config)

    def test_context_and_lane_bounds(self):
        self.config["max_output_tokens"] = 32768
        with self.assertRaisesRegex(ValueError, "model context"):
            validate(self.config)

    def test_no_silent_bool_or_negative_limits(self):
        for value in (True, -1, 0):
            self.config["short_decode_slots"] = value
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate(self.config)


if __name__ == "__main__":
    unittest.main()
