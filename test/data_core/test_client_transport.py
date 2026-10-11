"""Client bounds and deadline tests, without a GPU or running data service."""
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from pbe_data_client.client import DataClient, DataServiceError, DataError, DataKind, HEADER, MAGIC, VERSION


class DataClientTransportTest(unittest.TestCase):
    def run_server(self, respond):
        directory = tempfile.TemporaryDirectory()
        endpoint = str(Path(directory.name) / 'data.sock')
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(endpoint)
        server.listen(1)
        def serve():
            with server:
                connection, _ = server.accept()
                with connection:
                    header = bytearray()
                    while len(header) < HEADER.size:
                        header.extend(connection.recv(HEADER.size - len(header)))
                    respond(connection, HEADER.unpack(header))
        thread = threading.Thread(target=serve)
        thread.start()
        self.addCleanup(directory.cleanup)
        self.addCleanup(lambda: thread.join(2))
        return endpoint

    def test_stalled_response_expires(self):
        endpoint = self.run_server(lambda connection, header: time.sleep(.1))
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            DataClient(endpoint, timeout_s=.03).ping()
        self.assertLess(time.monotonic() - start, .5)

    def test_oversized_response_rejected_before_reading_payload(self):
        def respond(connection, header):
            connection.sendall(HEADER.pack(MAGIC, VERSION, header[2], 0, 0, header[5], 1025, 0))
        endpoint = self.run_server(respond)
        with self.assertRaisesRegex(ValueError, 'payload too large'):
            DataClient(endpoint, max_payload_bytes=1024).ping()

    def test_remote_failure_has_a_stable_error_code(self):
        def respond(connection, header):
            connection.sendall(HEADER.pack(MAGIC, VERSION, header[2],
                DataError.CAPACITY_EXHAUSTED, 0, header[5], 0, 0))
        with self.assertRaises(DataServiceError) as failure:
            DataClient(self.run_server(respond)).ping()
        self.assertEqual(failure.exception.code, DataError.CAPACITY_EXHAUSTED)

    def test_short_lease_is_a_protocol_error(self):
        from unittest.mock import patch
        client = DataClient('/unused')
        with patch.object(client, '_call', return_value=b'short'):
            with self.assertRaisesRegex(ValueError, 'short lease descriptor'):
                client.acquire(DataKind.KV_PAGE, bytes(32), bytes(32))

    def test_fragmented_response_is_reassembled(self):
        def respond(connection, header):
            wire = HEADER.pack(MAGIC, VERSION, header[2], 0, 0, header[5], 8, 0) + struct.pack('<Q', 42)
            for byte in wire:
                connection.sendall(bytes([byte]))
        self.assertEqual(DataClient(self.run_server(respond)).ping(), 42)


if __name__ == '__main__':
    unittest.main()
