#!/usr/bin/env python3
"""Validate E4 build/unit/full-suite/plain-text regression logs."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = (args.directory / "build.log").read_text(errors="replace")
    unit = (args.directory / "shared_weight_unit.log").read_text(errors="replace")
    full = (args.directory / "full_test.log").read_text(errors="replace")
    text = (args.directory / "plain_text.log").read_text(errors="replace")
    unit_count = re.findall(r"\[  PASSED  \]\s+(\d+) tests?", unit)
    full_count = re.findall(r"\[  PASSED  \]\s+(\d+) tests?", full)
    checks = {
        "build_targets": all(marker in build for marker in
            ("Built target pbe_data_service", "Built target pbe_vlm_language_role",
             "Built target test_llm")),
        "shared_weight_unit_tests": bool(unit_count) and int(unit_count[-1]) >= 4 and
            "FAILED" not in unit,
        "full_regression": bool(full_count) and int(full_count[-1]) >= 232 and
            "FAILED" not in full,
        "plain_text_interface": "PBE_MULTI_ROLE_TEXT_OK" in text,
    }
    result = {"schema": "pbe-e4-regression-v1", "ok": all(checks.values()),
              "checks": checks,
              "shared_weight_unit_tests": int(unit_count[-1]) if unit_count else 0,
              "full_tests": int(full_count[-1]) if full_count else 0}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]: raise SystemExit(1)


if __name__ == "__main__": main()
