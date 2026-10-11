"""Exercise the real API against a deterministic CPU-only ZeroMQ worker."""
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest

import zmq

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('PBE_SERVING_BINARY', ROOT / 'build-mps/demo/serving_qwen'))


@unittest.skipUnless(BINARY.is_file(), 'build serving_qwen before integration tests')
class HttpIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='pbe-http-test-')
        cls.path = Path(cls.directory.name)
        cls.endpoint = 'ipc://' + str(cls.path / 'worker.sock')
        cls.stop = threading.Event()
        cls.ready = threading.Event()
        cls.cancelled = []
        cls.worker = threading.Thread(target=cls.serve_worker)
        cls.worker.start()
        assert cls.ready.wait(5)
        registry = cls.path / 'registry.json'
        registry.write_text(json.dumps(dict(models=[dict(id='test', endpoint=cls.endpoint)], default_model='test')))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.log = open(cls.path / 'api.log', 'w')
        cls.process = subprocess.Popen([str(BINARY), 'unused', 'unused', '--online-server=1',
            '--online-process-role=zmq-http-api', '--model-registry=' + str(registry),
            '--listen-port=' + str(cls.port), '--http-worker-threads=4',
            '--http-max-pending-connections=4', '--http-io-timeout-ms=200',
            '--engine-zmq-timeout-ms=200'], cwd=cls.path, stdout=cls.log, stderr=cls.log)
        try:
            for _ in range(100):
                if cls.process.poll() is not None:
                    raise RuntimeError((cls.path / 'api.log').read_text())
                try:
                    if cls.call('GET', '/health')[0] == 200:
                        return
                except OSError:
                    time.sleep(.03)
            raise TimeoutError('API not ready')
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def serve_worker(cls):
        # The socket never crosses a thread boundary.
        with zmq.Context() as context:
            with context.socket(zmq.REP) as worker:
                worker.setsockopt(zmq.RCVTIMEO, 100)
                worker.bind(cls.endpoint)
                cls.ready.set()
                requests = {}
                sequence = 0
                while not cls.stop.is_set():
                    try:
                        request = worker.recv_json()
                    except zmq.Again:
                        continue
                    kind = request['type']
                    response = dict(ok=True, model_id='test')
                    if kind == 'generate':
                        sequence += 1
                        requests[sequence] = request['request']['prompt']
                        response['request_id'] = sequence
                    elif kind == 'token':
                        rid = request['request_id']
                        if requests.get(rid) == 'stall':
                            response['events'] = [dict(type='none')]
                        else:
                            response['events'] = [dict(type='token', text='hello'),
                                                  dict(type='final', text='hello', failed=False)]
                            requests.pop(rid, None)
                    elif kind == 'cancel':
                        cls.cancelled.append(request['request_id'])
                        requests.pop(request['request_id'], None)
                    elif kind == 'metrics':
                        response['metrics'] = dict(remote_requests=len(requests))
                    worker.send_json(response)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'process'):
            cls.process.terminate()
            try:
                cls.process.wait(5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait()
        cls.stop.set()
        cls.worker.join(5)
        cls.log.close()
        cls.directory.cleanup()

    @classmethod
    def call(cls, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', cls.port, timeout=3)
        try:
            connection.request(method, path, body)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def generate(self, **kwargs):
        return self.call('POST', '/v1/completions', json.dumps(dict(prompt='hello', **kwargs)))

    def test_invalid_fields_are_request_errors(self):
        for payload in ({'prompt': 123}, [], {'prompt': 'hi', 'stream': []}, {'messages': [1]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.call('POST', '/v1/completions', json.dumps(payload))[0], 400)
                self.assertEqual(self.call('GET', '/health')[0], 200)

    def test_invalid_content_length(self):
        with socket.create_connection(('127.0.0.1', self.port), 2) as client:
            client.sendall(b'POST /v1/completions HTTP/1.1\r\nContent-Length: x\r\n\r\n')
            self.assertIn(b'400 Bad Request', client.recv(4096))
        self.assertEqual(self.call('GET', '/health')[0], 200)

    def test_stream_deadline_cancels_worker(self):
        before = len(self.cancelled)
        start = time.monotonic()
        status, result = self.call('POST', '/v1/completions', json.dumps(
            dict(prompt='stall', stream=True, timeout_ms=100)))
        self.assertEqual(status, 200)
        self.assertIn(b'request_timeout', result)
        self.assertLess(time.monotonic() - start, 1)
        self.assertGreater(len(self.cancelled), before)

    def test_nonstream_deadline_cancels_worker(self):
        status, result = self.call('POST', '/v1/completions', json.dumps(
            dict(prompt='stall', timeout_ms=100)))
        self.assertEqual(status, 408)
        self.assertIn(b'request_timeout', result)

    def test_completed_requests_do_not_accumulate_threads_or_handles(self):
        for _ in range(100):
            status, response = self.generate()
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(response)['choices'][0]['text'], 'hello')
        status, response = self.call('GET', '/metrics')
        metrics = json.loads(response)['models']['test']
        self.assertEqual(metrics['engine_pool']['pending_remote_requests'], 0)
        self.assertEqual(metrics['engine_pool']['polling_threads'], 1)

    def test_concurrent_streams_preserve_results(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.generate(stream=True), range(16)))
        for status, body in results:
            self.assertEqual(status, 200)
            self.assertIn(b'hello', body)
            self.assertIn(b'[DONE]', body)

    def test_incomplete_headers_expire(self):
        clients = [socket.create_connection(('127.0.0.1', self.port), 2) for _ in range(4)]
        try:
            for client in clients:
                client.sendall(b'GET /health HTTP/1.1\r\n')
            time.sleep(.05)
            self.assertEqual(self.call('GET', '/health')[0], 200)
            for client in clients:
                self.assertIn(b'408 Request Timeout', client.recv(4096))
        finally:
            for client in clients:
                client.close()


    def test_overload_rejects_excess_connections(self):
        clients = [socket.create_connection(('127.0.0.1', self.port), 2) for _ in range(12)]
        try:
            replies = []
            for client in clients:
                replies.append(client.recv(4096))
            self.assertTrue(any(b'503 Service Unavailable' in reply for reply in replies))
            self.assertEqual(self.call('GET', '/health')[0], 200)
        finally:
            for client in clients:
                client.close()

    def test_shutdown_with_incomplete_header_is_bounded(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [arg for arg in self.process.args if not arg.startswith('--listen-port=')]
        command.append('--listen-port=' + str(port))
        with open(self.path / 'shutdown.log', 'w') as log:
            process = subprocess.Popen(command, cwd=self.path, stdout=log, stderr=log)
            client = None
            try:
                for _ in range(100):
                    try:
                        client = socket.create_connection(('127.0.0.1', port), .1)
                        break
                    except OSError:
                        time.sleep(.02)
                self.assertIsNotNone(client)
                client.sendall(b'GET /health HTTP/1.1\r\n')
                time.sleep(.03)
                process.terminate()
                self.assertEqual(process.wait(2), 0)
            finally:
                if client is not None:
                    client.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
