"""Dependency-free PBE data-service protocol client."""
from __future__ import annotations
import enum
import math
import os
import socket
import struct
import time
from dataclasses import dataclass

MAGIC = 0x53534250
VERSION = 3
HEADER = struct.Struct('<IHHHHQQI')
REF_MAGIC = 0x52444250
REF_SIZE = 100
UINT64_MASK = (1 << 64) - 1


class DataError(enum.IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    UNSUPPORTED_VERSION = 2
    UNKNOWN_OBJECT = 3
    STALE_GENERATION = 4
    NOT_READY = 5
    ALREADY_SEALED = 6
    CAPACITY_EXHAUSTED = 7
    COVERAGE_MISMATCH = 8
    CHECKSUM_MISMATCH = 9
    OWNER_RESTARTED = 10


class DataServiceError(RuntimeError):
    """A remote protocol failure with a stable code for callers to inspect."""

    def __init__(self, code: int):
        self.code = DataError(code)
        super().__init__(self.code)


class DataKind(enum.IntEnum):
    KV_PAGE = 1
    TENSOR_BUNDLE = 2
    CHECKPOINT = 3


@dataclass(frozen=True)

class LeaseToken:
    service_incarnation: int
    consumer_incarnation: int
    lease_id: int


@dataclass

class Lease:
    token: LeaseToken
    ref: bytes
    payload: bytes

    @property
    def lease_id(self) -> int:
        return self.token.lease_id


def checksum256(value: bytes) -> bytes:
    mask = (1 << 64) - 1
    h = [1469598103934665603, 1099511628211, 11400714819323198485, 15485907386658061715]
    for byte in value:
        for j in range(4):
            h[j] ^= byte + j * 41
            h[j] = h[j] * (1099511628211 + j * 2) & mask
            h[j] ^= h[j] >> 29
    return struct.pack('<QQQQ', *h)


class Operation(enum.IntEnum):
    PING = 1
    RESERVE = 2
    SEAL = 3
    ACQUIRE = 4
    RELEASE = 5
    RELEASE_PRODUCER = 7
    STATS = 8
    SHUTDOWN = 9



class DataClient:
    """One bounded transaction per call; leases require explicit release.

    ACQUIRE alone retries once with the same operation identity, so the server
    can deduplicate it. Other mutations are never replayed after a lost reply.
    """

    def __init__(self, endpoint: str, timeout_s: float = 30.0,
                 max_payload_bytes: int = 1 << 30):
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_payload_bytes <= 0:
            raise ValueError('positive finite timeout and payload limit required')
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_payload_bytes = max_payload_bytes
        self._seq = 0
        self.client_incarnation = (os.getpid() << 32) ^ time.monotonic_ns()

    def _call(self, op: int, payload: bytes = b'') -> bytes:
        if len(payload) > self.max_payload_bytes:
            raise ValueError('request payload too large')
        self._seq += 1
        rid = (self.client_incarnation + self._seq) & UINT64_MASK
        deadline = time.monotonic() + self.timeout_s
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:

            def remaining():
                duration = deadline - time.monotonic()
                if duration <= 0:
                    raise TimeoutError('data service deadline exceeded')
                connection.settimeout(duration)

            def receive(size):
                result = bytearray()
                while len(result) < size:
                    remaining()
                    part = connection.recv(min(size - len(result), 1 << 20))
                    if not part:
                        raise ConnectionError('short data service response')
                    result.extend(part)
                return bytes(result)
            remaining()
            connection.connect(self.endpoint)
            remaining()
            connection.sendall(HEADER.pack(MAGIC, VERSION, op, 0, 0, rid, len(payload), 0))
            if payload:
                remaining()
                connection.sendall(payload)
            (magic, version, rop, error, reserved, rrid, size, pad) = HEADER.unpack(receive(HEADER.size))
            if (magic, version, rop, reserved, rrid, pad) != (MAGIC, VERSION, op, 0, rid, 0):
                raise ValueError('invalid response header')
            if size > self.max_payload_bytes:
                raise ValueError('response payload too large')
            result = receive(size)
            if error:
                raise DataServiceError(error)
            return result

    def ping(self) -> int:
        return struct.unpack('<Q', self._call(Operation.PING))[0]

    def reserve(self, kind: DataKind, content: bytes, representation: bytes,
                size: int, operation: int | None = None) -> tuple[int, int, int]:
        if len(content) != 32 or len(representation) != 32 or size <= 0:
            raise ValueError('invalid reservation')
        owner = self.ping()
        seq = operation or ((self.client_incarnation + self._seq + 1) & UINT64_MASK)
        request = (struct.pack('<QQB', owner, seq, int(kind)) + content
                   + representation + struct.pack('<Q', size))
        response = self._call(Operation.RESERVE, request)
        return struct.unpack('<QQI', response)

    def seal(self, handle: tuple[int, int, int], payload: bytes) -> bytes:
        request = (struct.pack('<QQI', *handle) + checksum256(payload)
                   + struct.pack('<Q', len(payload)) + payload)
        response = self._call(Operation.SEAL, request)
        if len(response) != REF_SIZE:
            raise ValueError('invalid DataRef')
        return response

    def acquire(self, kind: DataKind, content: bytes, representation: bytes,
                lease_kind: int = 1) -> Lease:
        self._seq += 1
        operation = (self.client_incarnation, self._seq)
        request = (struct.pack('<QQ', *operation) + bytes([kind]) + content
                   + representation + bytes([lease_kind]))
        try:
            response = self._call(Operation.ACQUIRE, request)
        except (ConnectionError, OSError):
            response = self._call(Operation.ACQUIRE, request)
        if len(response) < 40 + REF_SIZE:
            raise ValueError('short lease descriptor')
        (service, consumer, lease_id, ref_size) = struct.unpack_from('<QQQQ', response)
        if ref_size != REF_SIZE:
            raise ValueError('invalid DataRef size')
        ref = response[32:32 + ref_size]
        size = struct.unpack_from('<Q', response, 32 + ref_size)[0]
        payload = response[40 + ref_size:]
        if len(payload) != size:
            raise ValueError('invalid lease payload')
        return Lease(LeaseToken(service, consumer, lease_id), ref, payload)

    def release(self, token: LeaseToken) -> None:
        request = struct.pack('<QQQ', token.service_incarnation,
                              token.consumer_incarnation, token.lease_id)
        self._call(Operation.RELEASE, request)

    def release_producer(self, handle: tuple[int, int, int]) -> None:
        self._call(Operation.RELEASE_PRODUCER, struct.pack('<QQI', *handle))

    def stats(self) -> tuple[int, ...]:
        return struct.unpack('<QQQQQQQ', self._call(Operation.STATS))

    def shutdown(self) -> None:
        self._call(Operation.SHUTDOWN)
