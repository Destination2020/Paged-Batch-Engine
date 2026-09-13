import fcntl
import multiprocessing
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from pbe_roles.vision.identity import cache_identity, singleflight_lock_path


def lock_then_crash(path: str, ready):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready.send(True)
    os._exit(17)


class VisionIdentityTest(unittest.TestCase):
    def make_model(self, root: Path, marker: str):
        (root / "config.json").write_text('{"marker":"%s"}' % marker)
        (root / "preprocessor_config.json").write_text('{"size":224}')
        (root / "model.safetensors.index.json").write_text('{"weight_map":{"visual.x":"v.safetensors"}}')
        (root / "v.safetensors").write_bytes(marker.encode())

    def test_model_or_processor_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory);self.make_model(model,"a")
            first=cache_identity(b"image",224,model)[:2]
            (model/"preprocessor_config.json").write_text('{"size":336}')
            second=cache_identity(b"image",224,model)[:2]
            self.assertNotEqual(first,second)

    def test_service_incarnation_separates_locks_and_crash_releases_flock(self):
        content=b"x"*32
        first=singleflight_lock_path("/tmp/service.sock",1,content)
        second=singleflight_lock_path("/tmp/service.sock",2,content)
        self.assertNotEqual(first,second)
        receiver,sender=multiprocessing.Pipe(False)
        process=multiprocessing.Process(target=lock_then_crash,args=(str(first),sender));process.start()
        self.assertTrue(receiver.recv());process.join(2);self.assertEqual(process.exitcode,17)
        fd=os.open(first,os.O_CREAT|os.O_RDWR,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(fd);first.unlink(missing_ok=True);second.unlink(missing_ok=True)


if __name__ == "__main__": unittest.main()
