#!/usr/bin/env python3
"""Reproduce the V4 M0 baseline and write machine-readable evidence.

The runner deliberately uses existing local dependency/model caches.  It does not
download packages or mutate the source tree outside the evidence directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/data_flow_evidence/v4/M0"
LOGS = OUT / "logs"

# Captured before this runner was added.  This identifies the inherited V3
# worktree that M0 was asked to preserve.
INHERITED = {
    "head": "f3597a763b6cd5cf60b826baa2dc94395c3d02fd",
    "tracked_binary_diff_sha256": "71942cb3b21367fdd3171139f86d9c84b8befcc44df59ccc9f68c41e0477532b",
    "index_binary_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "source_tree_listing_sha256": "13bdac6c6cf0e7b6aebe6482824900177a8f6ad2cbbeb1b44eb70421e02a828c",
    "porcelain_status_sha256": "432c7f6623b603393976cd18e3996a8ad1e15d55307eefd3f7e6c41e2ec1a251",
}


def sha256_file(path: pathlib.Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def capture(argv: list[str], *, env: dict[str, str] | None = None) -> dict[str, object]:
    completed = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True)
    return {
        "command": argv,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run(name: str, argv: list[str], *, env: dict[str, str] | None = None) -> dict[str, object]:
    started = dt.datetime.now(dt.timezone.utc)
    begin = time.monotonic()
    completed = subprocess.run(argv, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - begin
    log_path = LOGS / f"{name}.log"
    header = (
        f"started_utc: {started.isoformat()}\n"
        f"cwd: {ROOT}\n"
        f"command: {shlex.join(argv)}\n"
        f"exit_code: {completed.returncode}\n"
        f"elapsed_seconds: {elapsed:.6f}\n\n"
    )
    log_path.write_text(header + completed.stdout, encoding="utf-8")
    print(f"[{name}] exit={completed.returncode} elapsed={elapsed:.2f}s log={log_path}", flush=True)
    return {
        "name": name,
        "command": argv,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "log": str(log_path.relative_to(ROOT)),
    }


def source_files() -> list[dict[str, object]]:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "infMain", "test", "demo", "tools", "cmake", "CMakeLists.txt"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return [sha256_file(ROOT / path) for path in sorted(set(paths)) if (ROOT / path).is_file()]


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "transformers", "accelerate", "safetensors", "tokenizers", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    model_root = pathlib.Path("/tmp/Paged-Batch-Engine-models")
    model_dir = model_root / "Qwen2-0.5B-Instruct"
    model_files = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
        model_dir / "generation_config.json",
        model_dir / "model.safetensors",
        model_root / "Qwen2-0.5B-Instruct.bf16.bin",
    ]
    manifest = {
        "schema_version": 1,
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inherited_worktree": INHERITED,
        "head": capture(["git", "rev-parse", "HEAD"]),
        "status": capture(["git", "status", "--short", "--branch"]),
        "tracked_binary_diff_sha256": subprocess.check_output(
            "git diff --binary | sha256sum", cwd=ROOT, shell=True, text=True).split()[0],
        "source_files": source_files(),
        "model_revision": "Upstream revision unavailable in local snapshot; exact bytes frozen by SHA-256",
        "model_files": [sha256_file(path) for path in model_files if path.is_file()],
        "platform": platform.platform(),
        "python": sys.version,
        "python_packages": package_versions(),
        "compiler": capture(["c++", "--version"]),
        "cmake": capture(["cmake", "--version"]),
        "cuda": capture(["nvcc", "--version"]),
        "gpu": capture(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
                        "--format=csv,noheader"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    deps = pathlib.Path("/tmp/Paged-Batch-Engine-build-p1/_deps")
    cpu = pathlib.Path("/tmp/Paged-Batch-Engine-v4-m0-cpu")
    asan = pathlib.Path("/tmp/Paged-Batch-Engine-v4-m0-asan")
    qwen_build = pathlib.Path("/tmp/Paged-Batch-Engine-build-qwen05")
    checks: list[dict[str, object]] = []

    common = [
        "-DKUIPER_TEST_CPU_OWNERSHIP=ON",
        f"-Dglog_DIR={deps / 'glog-build'}",
        f"-DGTest_DIR={deps / 'googletest-build'}",
        f"-DARMADILLO_INCLUDE_DIR={deps / 'armadillo-src/include'}",
        f"-DARMADILLO_LIBRARY={deps / 'armadillo-build/libarmadillo.so'}",
    ]
    checks.append(run("cpu_configure", ["cmake", "-S", "test/cache_core", "-B", str(cpu), *common,
                                         "-DCMAKE_BUILD_TYPE=RelWithDebInfo"]))
    checks.append(run("cpu_build", ["cmake", "--build", str(cpu), "-j16"]))
    checks.append(run("cpu_tests", ["ctest", "--test-dir", str(cpu), "--output-on-failure"]))

    sanitizer_libdir = "/tmp/Paged-Batch-Engine-v4-m0-sanitizer-runtime/root/usr/lib64"
    sanitizer_flags = (
        "-fsanitize=address,undefined -fno-omit-frame-pointer "
        f"-L{sanitizer_libdir}"
    )
    sanitizer_env = os.environ.copy()
    sanitizer_env["LIBRARY_PATH"] = sanitizer_libdir + os.pathsep + sanitizer_env.get("LIBRARY_PATH", "")
    sanitizer_env["LD_LIBRARY_PATH"] = sanitizer_libdir + os.pathsep + sanitizer_env.get("LD_LIBRARY_PATH", "")
    sanitizer_env["LD_PRELOAD"] = str(pathlib.Path(sanitizer_libdir) / "libasan.so.6.0.0")
    sanitizer_env.update({"ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
                          "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"})
    checks.append(run("sanitizer_configure", ["cmake", "-S", "test/cache_core", "-B", str(asan),
                                               *common, "-DCMAKE_BUILD_TYPE=Debug",
                                               f"-DCMAKE_CXX_FLAGS={sanitizer_flags}",
                                               f"-DCMAKE_EXE_LINKER_FLAGS={sanitizer_flags}"],
                      env=sanitizer_env))
    checks.append(run("sanitizer_build", ["cmake", "--build", str(asan), "-j16"],
                      env=sanitizer_env))
    sanitizer_test_env = sanitizer_env.copy()
    sanitizer_test_env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    checks.append(run("sanitizer_tests", ["ctest", "--test-dir", str(asan), "--output-on-failure"],
                      env=sanitizer_test_env))

    checks.append(run("cuda_configure", [
        "cmake", "-S", ".", "-B", "build-v3", "-DUSE_CPM=ON",
        "-DKUIPER_BUILD_DEMOS=OFF", "-DKUIPER_BUILD_TESTS=ON",
        "-DQWEN2_SUPPORT=OFF", "-DKUIPER_ENABLE_NCCL=OFF",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DCMAKE_CUDA_ARCHITECTURES=90",
    ]))
    checks.append(run("cuda_build", ["cmake", "--build", "build-v3", "--target", "test_llm", "-j16"]))
    cuda_filter = (
        "TransferSchedulerTest.*:SchedulerCheckpointTest.*:KVCacheManagerTest.*:"
        "RequestCheckpointTest.*:PDHandoffTest.*:PageSchemaTest.*:CacheTransferPlanTest.*:"
        "CacheLayoutTest.*:PageDirectoryTest.*:PageMigrationTest.*:PageMigrationCudaTest.*:"
        "HostStoreTest.*:HostStoreCudaTest.*"
    )
    checks.append(run("cuda_data_flow_tests", ["./build-v3/test/test_llm", f"--gtest_filter={cuda_filter}"]))
    checks.append(run("cuda_full_tests", [
        "./build-v3/test/test_llm", "--gtest_filter=-test_load.*",
    ]))

    checks.append(run("qwen_build", ["cmake", "--build", str(qwen_build), "--target", "serving_qwen",
                                      "qwen_instruct_infer", "test_llm", "-j16"]))
    smoke_env = os.environ.copy()
    smoke_env["CUDA_VISIBLE_DEVICES"] = "0"
    smoke_env["PBE_QWEN2_TEST_MODEL"] = str(model_root / "Qwen2-0.5B-Instruct.bf16.bin")
    smoke_env["PBE_QWEN2_TEST_TOKENIZER"] = str(model_dir / "tokenizer.json")
    checks.append(run("qwen_greedy_smoke", [
        str(qwen_build / "demo/serving_qwen"),
        str(model_root / "Qwen2-0.5B-Instruct.bf16.bin"),
        str(model_dir / "tokenizer.json"),
        "Explain KV cache in one sentence.", "What is continuous batching?",
        "--max-new-tokens=24", "--max-batched-tokens=128",
        "--kv-cache-memory-utilization=0.02", "--warmup-rounds=1", "--quiet=1",
        "--final-summary=1",
    ], env=smoke_env))
    model_test_env = smoke_env.copy()
    model_test_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    checks.append(run("qwen_model_regressions", [
        str(qwen_build / "test/test_llm"),
        "--gtest_filter=Qwen2DeviceInitTest.DualGpuP2PMatchesSingleModelGreedyTokens:"
        "Qwen2DeviceInitTest.CheckpointResumeMatchesUninterruptedSampledTokens",
    ], env=model_test_env))

    summary = {
        "schema_version": 1,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checks": checks,
        "passed": sum(check["exit_code"] == 0 for check in checks),
        "failed": sum(check["exit_code"] != 0 for check in checks),
        "m0_complete": all(check["exit_code"] == 0 for check in checks),
        "notes": [
            "NCCL-only tests remain inapplicable because the baseline build has KUIPER_ENABLE_NCCL=OFF.",
            "test_load.* is excluded because this checkout has no external llama fixture at ../stories15M.bin.",
            "The legacy hex-JSON/ZMQ handoff remains a compatibility/oracle path, not the V4 transport.",
        ],
    }
    (OUT / "checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["m0_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
