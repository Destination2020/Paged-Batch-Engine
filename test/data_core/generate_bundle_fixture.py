"""Generate the Python wire fixture consumed by the C++ cross-language test."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from pbe_data_client.tensor_bundle import encode


def main():
    payload = encode([
        ('image_features', 'float32', (1, 4), struct.pack('<4f', 1, 2, 3, 4)),
        ('image_grid_thw', 'int64', (1, 3), struct.pack('<3q', 1, 2, 2)),
    ])
    Path(sys.argv[1]).write_bytes(payload)


if __name__ == '__main__':
    main()
