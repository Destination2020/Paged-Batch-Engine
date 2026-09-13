#!/usr/bin/env python3
"""Hash final V4 source and acceptance evidence without recording secrets."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hasher.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    listed = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard",
        "CMakeLists.txt", "demo", "infMain", "python", "test", "tools/bench/data_flow",
        "tools/launch", "tools/models"], cwd=root, text=True).splitlines()
    sources = [root / item for item in sorted(set(listed))
               if (root / item).is_file() and not item.startswith("build-")]
    evidence_names = [
        "docs/V4_FINAL_COMPLETION_20260913.md",
        "docs/PBE_MULTI_ROLE_MULTIMODAL_EXECUTION_PLAN_V4_20260912.md",
        "docs/data_flow_evidence/v4/M5_numerics_final/validation_final.json",
        "docs/data_flow_evidence/v4/M5_online_lifecycle_final/validation.json",
        "docs/data_flow_evidence/v4/E1/checks.json",
        "docs/data_flow_evidence/v4/E2/checks.json",
        "docs/data_flow_evidence/v4/E3/checks.json",
        "docs/data_flow_evidence/v4/M9/final_results.json",
        "docs/data_flow_evidence/v4/final_regression/checks.json",
    ]
    evidence = [root / item for item in evidence_names]
    result = {
        "schema": "pbe.v4.final-manifest.v1",
        "date": "2026-09-13",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_files": [digest(path) | {"path": str(path.relative_to(root))}
                         for path in sources],
        "acceptance_files": [digest(path) | {"path": str(path.relative_to(root))}
                             for path in evidence],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"ok": True, "source_files": len(sources),
                      "acceptance_files": len(evidence)}, sort_keys=True))


if __name__ == "__main__":
    main()
