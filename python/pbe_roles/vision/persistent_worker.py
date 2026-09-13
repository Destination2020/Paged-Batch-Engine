#!/usr/bin/env python3
"""Persistent typed Qwen2.5-VL vision role with bounded dynamic batching."""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from pbe_data_client import DataClient, DataKind
from pbe_data_client.client import checksum256
from pbe_data_client.tensor_bundle import decode, encode
from pbe_roles.vision.identity import cache_identity
from pbe_roles.vision.worker import load_visual, sequence_metadata


def _missing(error: RuntimeError) -> bool:
    return str(error) in {"3", "5"}


@dataclass
class Work:
    request_id: str
    generation: int
    image: Path
    size: int
    text: str
    deadline: float
    feature_cache: bool = True
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class VisionRole:
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoProcessor

        self.args = args
        if os.environ.get("PBE_DETERMINISTIC_VISION") == "1":
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.client = DataClient(args.endpoint)
        self.processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, use_fast=False)
        self.vision_config = AutoConfig.from_pretrained(
            args.model, local_files_only=True).vision_config
        self.visual = load_visual(args.model, args.device)
        self.torch = torch
        self.pending: queue.Queue[Work | None] = queue.Queue(args.max_queue)
        self.active: dict[tuple[str, int], Work] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.forward_count = 0
        self.batch_count = 0
        self.feature_lock = threading.Lock()
        self.gpu_features: dict[bytes, dict] = {}
        self.demoted_features: dict[bytes, dict] = {}
        self.gpu_feature_bytes = 0
        self.gpu_feature_capacity = args.gpu_feature_cache_bytes
        self.gpu_feature_restores = 0
        _, self.device_total_bytes = torch.cuda.mem_get_info(args.device)
        self.device_admission_limit = max(0, self.device_total_bytes - (256 << 20))
        self.peak_process_allocated_bytes = torch.cuda.memory_allocated(args.device)
        self.thread = threading.Thread(target=self._run, name="vision-batcher")
        self.thread.start()

    def _gpu_feature(self, content: bytes, representation: bytes,
                     shape: tuple[int, int]):
        with self.feature_lock:
            entry = self.gpu_features.get(content)
            if (entry is None or entry["representation"] != representation or
                    entry["shape"] != shape):
                return None
            return entry["tensor"]

    def _install_gpu_feature(self, content: bytes, representation: bytes,
                             shape: tuple[int, int], tensor) -> bool:
        feature_bytes = tensor.numel() * tensor.element_size()
        free_bytes, _ = self.torch.cuda.mem_get_info(self.args.device)
        if free_bytes < feature_bytes + (64 << 20):
            return False
        with self.feature_lock:
            if content in self.gpu_features:
                return True
            if self.gpu_feature_bytes + feature_bytes > self.gpu_feature_capacity:
                return False
            self.gpu_features[content] = {
                "representation": representation, "shape": shape,
                "tensor": tensor.detach().contiguous(), "bytes": feature_bytes,
            }
            self.demoted_features[content] = {
                "representation": representation, "shape": shape,
                "bytes": feature_bytes,
            }
            self.gpu_feature_bytes += feature_bytes
            self.peak_process_allocated_bytes = max(
                self.peak_process_allocated_bytes,
                self.torch.cuda.memory_allocated(self.args.device))
        return True

    def demote_features(self) -> dict:
        self.torch.cuda.synchronize(self.args.device)
        with self.feature_lock:
            objects = len(self.gpu_features)
            feature_bytes = self.gpu_feature_bytes
            self.gpu_features.clear()
            self.gpu_feature_bytes = 0
        return {"ok": True, "event": "gpu_features_demoted_to_host",
                "demoted_objects": objects, "demoted_bytes": feature_bytes,
                "host_recipes": len(self.demoted_features)}

    def restore_features(self) -> dict:
        restored_objects = 0
        restored_bytes = 0
        for content, recipe in list(self.demoted_features.items()):
            with self.feature_lock:
                if content in self.gpu_features:
                    continue
                if self.gpu_feature_bytes + recipe["bytes"] > self.gpu_feature_capacity:
                    return {"ok": False, "error": "gpu_feature_cache_full",
                            "restored_objects": restored_objects,
                            "restored_bytes": restored_bytes}
            lease = self.client.acquire(DataKind.TENSOR_BUNDLE, content,
                                        recipe["representation"])
            try:
                components = {name: (shape, value)
                              for name, _, shape, value in decode(lease.payload)}
                shape, value = components["image_features"]
                if tuple(shape) != recipe["shape"] or len(value) != recipe["bytes"]:
                    raise ValueError("host feature recovery shape mismatch")
                tensor = self.torch.frombuffer(bytearray(value), dtype=self.torch.uint16)
                tensor = tensor.view(*shape).view(self.torch.bfloat16).to(self.args.device)
                if not self._install_gpu_feature(content, recipe["representation"],
                                                 tuple(shape), tensor):
                    raise RuntimeError("gpu_feature_cache_full")
                restored_objects += 1
                restored_bytes += len(value)
            finally:
                self.client.release(lease.token)
        self.torch.cuda.synchronize(self.args.device)
        self.gpu_feature_restores += restored_objects
        return {"ok": True, "event": "host_features_restored_to_gpu",
                "restored_objects": restored_objects, "restored_bytes": restored_bytes}

    def feature_status(self) -> dict:
        with self.feature_lock:
            return {"ok": True, "gpu_feature_objects": len(self.gpu_features),
                    "gpu_feature_bytes": self.gpu_feature_bytes,
                    "gpu_feature_capacity": self.gpu_feature_capacity,
                    "host_recovery_recipes": len(self.demoted_features),
                    "gpu_feature_restores": self.gpu_feature_restores,
                    "device_total_bytes": self.device_total_bytes,
                    "device_admission_limit": self.device_admission_limit,
                    "process_allocated_bytes": self.torch.cuda.memory_allocated(
                        self.args.device),
                    "peak_process_allocated_bytes": self.peak_process_allocated_bytes}

    def submit(self, request: dict) -> Work:
        request_id = str(request.get("request_id", ""))
        generation = int(request.get("generation", 0))
        timeout_ms = int(request.get("timeout_ms", 30000))
        parts = request.get("parts")
        if not request_id or generation <= 0 or timeout_ms <= 0 or not isinstance(parts, list):
            raise ValueError("invalid request identity, generation, timeout, or parts")
        images = [part for part in parts if part.get("type") == "image"]
        texts = [part for part in parts if part.get("type") == "text"]
        if len(images) != 1 or len(texts) != 1 or len(parts) != 2:
            raise ValueError("V4 accepts exactly one image part and one text part")
        image = Path(images[0].get("path", ""))
        size = int(images[0].get("size", 224))
        text = str(texts[0].get("text", ""))
        if not image.is_file() or image.stat().st_size > 64 << 20:
            raise ValueError("invalid image or encoded image exceeds 64 MiB")
        if not 28 <= size <= 1024 or not text:
            raise ValueError("invalid image size or empty text")
        work = Work(request_id, generation, image, size, text,
                    time.monotonic() + timeout_ms / 1000.0,
                    bool(images[0].get("cache", True)))
        key = (request_id, generation)
        with self.lock:
            if key in self.active:
                raise ValueError("duplicate active request generation")
            self.active[key] = work
        try:
            self.pending.put_nowait(work)
        except queue.Full:
            with self.lock:
                self.active.pop(key, None)
            raise RuntimeError("vision_queue_full")
        return work

    def cancel(self, request_id: str, generation: int) -> bool:
        with self.lock:
            work = self.active.get((request_id, generation))
        if work is None:
            return False
        work.cancelled.set()
        return True

    def close(self):
        self.stopping.set()
        self.pending.put(None)
        self.thread.join()

    def _finish(self, work: Work, result: dict):
        work.result = result
        with self.lock:
            self.active.pop((work.request_id, work.generation), None)
        work.done.set()

    def _run(self):
        while True:
            first = self.pending.get()
            if first is None:
                return
            batch = [first]
            end = time.monotonic() + self.args.batch_window_ms / 1000.0
            while len(batch) < self.args.max_batch:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.pending.put(None)
                    break
                batch.append(item)
            live = []
            for work in batch:
                if work.cancelled.is_set():
                    self._finish(work, {"ok": False, "error": "cancelled", "stage": "encode"})
                elif time.monotonic() >= work.deadline:
                    self._finish(work, {"ok": False, "error": "deadline_exceeded", "stage": "encode"})
                else:
                    live.append(work)
            if live:
                self._process(live)

    def _inputs(self, work: Work):
        from PIL import Image

        image = Image.open(work.image).convert("RGB").resize(
            (work.size, work.size), Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": work.text},
        ]}]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")

    def _process(self, batch: list[Work]):
        import numpy as np

        torch = self.torch
        prepared = []
        reservations = []
        try:
            for work in batch:
                inputs = self._inputs(work)
                grid = inputs["image_grid_thw"].cpu()
                ids, positions, delta = sequence_metadata(inputs, self.vision_config)
                merge = int(getattr(self.vision_config, "spatial_merge_size", 2))
                tokens = int(grid.prod(-1).sum().item()) // (merge * merge)
                hidden = int(getattr(self.vision_config, "out_hidden_size",
                                     getattr(self.vision_config, "hidden_size", 0)))
                raw = work.image.read_bytes()
                # The semantic media identity is content-derived and must remain
                # stable when only the feature-cache policy changes.  A
                # request-private lookup key used to leak out as feature_content,
                # which unintentionally disabled semantic KV reuse as well.
                feature_id, feature_rep, _ = cache_identity(raw, work.size, self.args.model)
                feature_cache_enabled = (not self.args.disable_feature_cache and
                                         work.feature_cache)
                feature = None
                cache_hit = False
                cache_tier = "miss"
                gpu_feature = (self._gpu_feature(feature_id, feature_rep, (tokens, hidden))
                               if feature_cache_enabled else None)
                if gpu_feature is not None:
                    feature = gpu_feature.to("cpu").contiguous().view(
                        torch.uint16).numpy().astype("<u2", copy=False).tobytes()
                    cache_hit = True
                    cache_tier = "gpu"
                elif feature_cache_enabled:
                    try:
                        lease = self.client.acquire(DataKind.TENSOR_BUNDLE,
                                                    feature_id, feature_rep)
                        components = {name: (shape, value)
                                      for name, _, shape, value in decode(lease.payload)}
                        self.client.release(lease.token)
                        shape, value = components["image_features"]
                        if tuple(shape) != (tokens, hidden):
                            raise ValueError("cached feature shape mismatch")
                        feature = value
                        cache_hit = True
                        cache_tier = "host"
                    except RuntimeError as error:
                        if not _missing(error):
                            raise
                grid_bytes = grid.numpy().astype("<i8", copy=False).tobytes()
                feature_template = encode([
                    ("image_features", "bfloat16", (tokens, hidden),
                     b"\0" * (tokens * hidden * 2)),
                    ("image_grid_thw", "int64", tuple(grid.shape), grid_bytes),
                ])
                feature_handle = None
                if feature_cache_enabled and not cache_hit:
                    feature_handle = self.client.reserve(
                        DataKind.TENSOR_BUNDLE, feature_id, feature_rep,
                        len(feature_template))
                    reservations.append(feature_handle)
                metadata = [
                    ("input_ids", "int32", tuple(ids.shape),
                     ids.numpy().astype("<i4", copy=False).tobytes()),
                    ("position_ids", "int32", tuple(positions.shape),
                     positions.numpy().astype("<i4", copy=False).tobytes()),
                    ("rope_delta", "int64", (1,),
                     int(delta).to_bytes(8, "little", signed=True)),
                ]
                request_id = checksum256(feature_id + work.text.encode() + metadata[0][3])
                request_rep = checksum256(b"qwen25-vl-request-bundle-v3")
                request_template = encode([
                    ("image_features", "bfloat16", (tokens, hidden),
                     b"\0" * (tokens * hidden * 2)),
                    ("image_grid_thw", "int64", tuple(grid.shape), grid_bytes),
                    *metadata,
                ])
                request_handle = self.client.reserve(
                    DataKind.TENSOR_BUNDLE, request_id, request_rep,
                    len(request_template))
                reservations.append(request_handle)
                prepared.append(dict(work=work, inputs=inputs, grid=grid,
                                     tokens=tokens, hidden=hidden, feature=feature,
                                     feature_id=feature_id, feature_rep=feature_rep,
                                     feature_cache_enabled=feature_cache_enabled,
                                     feature_handle=feature_handle,
                                     request_id=request_id, request_rep=request_rep,
                                     request_handle=request_handle, metadata=metadata,
                                     grid_bytes=grid_bytes, cache_hit=cache_hit))
                prepared[-1]["cache_tier"] = cache_tier
                prepared[-1]["gpu_feature"] = gpu_feature

            misses = [item for item in prepared if item["feature"] is None]
            compute_misses = misses
            duplicate_sources = {}
            if misses and not self.args.disable_singleflight:
                representatives = {}
                compute_misses = []
                for item in misses:
                    key = item["feature_id"]
                    if key in representatives:
                        duplicate_sources[id(item)] = representatives[key]
                    else:
                        representatives[key] = item
                        compute_misses.append(item)
            if compute_misses:
                pixels = torch.cat([item["inputs"]["pixel_values"] for item in compute_misses])
                grids = torch.cat([item["grid"] for item in compute_misses])
                started = time.perf_counter()
                with torch.inference_mode():
                    packed_gpu = self.visual(
                        pixels.to(self.args.device, dtype=torch.bfloat16),
                        grid_thw=grids.to(self.args.device),
                        return_dict=True).pooler_output.to(torch.bfloat16)
                    packed = packed_gpu.cpu()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.forward_count += 1
                splits = [item["tokens"] for item in compute_misses]
                for item, feature, gpu_feature in zip(
                        compute_misses, torch.split(packed, splits),
                        torch.split(packed_gpu, splits)):
                    item["feature"] = feature.contiguous().view(torch.uint16).numpy().astype(
                        "<u2", copy=False).tobytes()
                    item["gpu_feature"] = gpu_feature
                for item in misses:
                    if id(item) in duplicate_sources:
                        item["feature"] = duplicate_sources[id(item)]["feature"]
                        item["gpu_feature"] = duplicate_sources[id(item)]["gpu_feature"]
            else:
                elapsed_ms = 0.0

            self.batch_count += 1
            for item in prepared:
                work = item["work"]
                feature_bundle = encode([
                    ("image_features", "bfloat16",
                     (item["tokens"], item["hidden"]), item["feature"]),
                    ("image_grid_thw", "int64", tuple(item["grid"].shape),
                     item["grid_bytes"]),
                ])
                if item["feature_handle"] is not None:
                    self.client.seal(item["feature_handle"], feature_bundle)
                    self.client.release_producer(item["feature_handle"])
                    reservations.remove(item["feature_handle"])
                if (item["gpu_feature"] is not None and
                        item["feature_cache_enabled"]):
                    self._install_gpu_feature(
                        item["feature_id"], item["feature_rep"],
                        (item["tokens"], item["hidden"]), item["gpu_feature"])
                if work.cancelled.is_set() or time.monotonic() >= work.deadline:
                    self.client.release_producer(item["request_handle"])
                    reservations.remove(item["request_handle"])
                    error = "cancelled" if work.cancelled.is_set() else "deadline_exceeded"
                    self._finish(work, {"ok": False, "error": error,
                                        "stage": "encode"})
                    continue
                request_bundle = encode([
                    ("image_features", "bfloat16",
                     (item["tokens"], item["hidden"]), item["feature"]),
                    ("image_grid_thw", "int64", tuple(item["grid"].shape),
                     item["grid_bytes"]),
                    *item["metadata"],
                ])
                self.client.seal(item["request_handle"], request_bundle)
                self.client.release_producer(item["request_handle"])
                reservations.remove(item["request_handle"])
                self._finish(work, {
                    "ok": True, "request_id": work.request_id,
                    "generation": work.generation,
                    "content": item["request_id"].hex(),
                    "representation": item["request_rep"].hex(),
                    "feature_content": item["feature_id"].hex(),
                    "feature_representation": item["feature_rep"].hex(),
                    "feature_cache_enabled": item["feature_cache_enabled"],
                    "feature_cache_hit": item["cache_hit"],
                    "feature_cache_tier": item["cache_tier"],
                    "feature_gpu_cache_hit": item["cache_tier"] == "gpu",
                    "feature_rows": item["tokens"],
                    "feature_payload_sha256": checksum256(item["feature"]).hex(),
                    "bundle_bytes": len(request_bundle),
                    "batch_size": len(batch), "forward_batch_size": len(misses),
                    "physical_forward_batch_size": len(compute_misses),
                    "singleflight_saved": len(misses) - len(compute_misses),
                    "forward_ms": elapsed_ms,
                    "worker_pid": os.getpid(),
                    "forward_count": self.forward_count,
                    "batch_count": self.batch_count,
                })
        except Exception as error:
            for handle in list(reservations):
                try:
                    self.client.release_producer(handle)
                except Exception:
                    pass
            for work in batch:
                if not work.done.is_set():
                    self._finish(work, {"ok": False, "error": str(error),
                                        "stage": "encode"})


def _read_request(connection: socket.socket) -> dict:
    connection.settimeout(5.0)
    data = bytearray()
    while b"\n" not in data:
        part = connection.recv(65536)
        if not part:
            break
        data.extend(part)
        if len(data) > 1 << 20:
            raise ValueError("request exceeds 1 MiB")
    return json.loads(bytes(data).split(b"\n", 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-window-ms", type=int, default=20)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-queue", type=int, default=64)
    parser.add_argument("--disable-feature-cache", action="store_true",
                        help="use request-private feature identities for cache-off experiments")
    parser.add_argument("--disable-singleflight", action="store_true",
                        help="compute duplicate cold identities independently within a batch")
    parser.add_argument("--gpu-feature-cache-bytes", type=int, default=0,
                        help="bounded optional GPU feature tier; Host cache remains canonical")
    args = parser.parse_args()
    if (args.batch_window_ms < 0 or args.max_batch <= 0 or args.max_queue <= 0 or
            args.gpu_feature_cache_bytes < 0):
        raise ValueError("invalid batching limits")

    listen = Path(args.listen)
    listen.unlink(missing_ok=True)
    role = VisionRole(args)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(listen))
    server.listen(128)
    server.settimeout(0.2)
    print(json.dumps({"event": "ready", "pid": os.getpid(),
                      "listen": str(listen), "device": args.device}), flush=True)

    def serve(connection: socket.socket):
        try:
            request = _read_request(connection)
            op = request.get("op", "encode")
            if op == "encode":
                work = role.submit(request)
                wait = max(0.0, work.deadline - time.monotonic()) + 1.0
                work.done.wait(wait)
                response = work.result or {"ok": False, "error": "deadline_exceeded"}
            elif op == "cancel":
                response = {"ok": role.cancel(str(request.get("request_id", "")),
                                               int(request.get("generation", 0)))}
            elif op == "shutdown":
                role.stopping.set()
                response = {"ok": True}
            elif op == "demote_features":
                response = role.demote_features()
            elif op == "restore_features":
                response = role.restore_features()
            elif op == "feature_status":
                response = role.feature_status()
            else:
                response = {"ok": False, "error": "unknown_op"}
        except Exception as error:
            response = {"ok": False, "error": str(error)}
        try:
            connection.sendall(json.dumps(response, sort_keys=True).encode() + b"\n")
        finally:
            connection.close()

    threads = []
    try:
        while not role.stopping.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=serve, args=(connection,))
            thread.start()
            threads.append(thread)
    finally:
        server.close()
        for thread in threads:
            thread.join()
        role.close()
        listen.unlink(missing_ok=True)
        print(json.dumps({"event": "stopped", "pid": os.getpid(),
                          "forward_count": role.forward_count,
                          "batch_count": role.batch_count}), flush=True)


if __name__ == "__main__":
    main()
