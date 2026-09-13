#!/usr/bin/env python3
"""Create the fixed 20-case persistent VLM mixed/single correctness trace."""
import argparse
import json


def request(case: int, suffix: str, image: str) -> dict:
    sizes = (224, 252, 280, 308)
    return {
        "request_id": f"case-{case:02d}-{suffix}",
        "generation": 1,
        "parts": [
            {"type": "image", "path": image, "size": sizes[case % len(sizes)]},
            {"type": "text", "text": f"Describe this image for fixed case {case:02d}."},
        ],
        "max_new_tokens": 8,
        "diagnostic_key": f"case-{case:02d}",
        "diagnostic_mode": suffix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mixed = [[request(case, "mixed", args.image) for case in range(begin, begin + 4)]
             for begin in range(0, 20, 4)]
    single = [[request(case, "single", args.image)] for case in range(20)]
    with open(args.output, "w") as stream:
        json.dump({"rounds": mixed + single}, stream)


if __name__ == "__main__":
    main()
