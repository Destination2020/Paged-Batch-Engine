#!/usr/bin/env python3
"""Small JSON-line client for the persistent Vision role."""
from __future__ import annotations

import argparse
import json
import socket


def call(endpoint: str, request: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(endpoint)
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = bytearray()
        while b"\n" not in response:
            part = connection.recv(65536)
            if not part:
                break
            response.extend(part)
    return json.loads(bytes(response).split(b"\n", 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", required=True)
    parser.add_argument("--op", choices=("encode", "cancel", "shutdown",
                                         "demote_features", "restore_features",
                                         "feature_status"),
                        default="encode")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--image")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--text", default="Describe the image briefly.")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()
    request = {"op": args.op}
    if args.op in {"encode", "cancel"}:
        request.update(request_id=args.request_id, generation=args.generation)
    if args.op == "encode":
        request.update(timeout_ms=args.timeout_ms, parts=[
            {"type": "image", "path": args.image, "size": args.size},
            {"type": "text", "text": args.text},
        ])
    response = call(args.listen, request)
    print(json.dumps(response, sort_keys=True))
    if not response.get("ok", False) and args.op != "cancel":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
