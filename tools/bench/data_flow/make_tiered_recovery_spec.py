#!/usr/bin/env python3
"""Build the real VLM E3 host-demotion plus checkpoint/restore fixture."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def request(name: str, tokens: int, mode: str) -> dict:
        return {
            "request_id": name, "generation": 1,
            "parts": [
                {"type": "image", "path": args.image, "size": 224},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
            "max_new_tokens": tokens,
            "diagnostic_key": "tiered-recovery",
            "diagnostic_mode": mode,
        }
    producer = request("gpu-prefix-producer", 16, "producer")
    recovered = request("host-prefix-checkpoint-recovery", 16, "recovered")
    recovered["demote_prefix_before"] = True
    recovered["external_checkpoint_after_ms"] = 120
    json.dump({"rounds": [[producer], [recovered]]},
              open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
