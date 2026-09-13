#!/usr/bin/env python3
"""Create randomized cache off/on rounds for persistent multimodal serving."""
import argparse
import json
import random


def item(request_id: str, image: str, cache: bool) -> dict:
    return {
        "request_id": request_id, "generation": 1,
        "parts": [
            {"type": "image", "path": image, "size": 252, "cache": cache},
            {"type": "text", "text": "Describe the image in one short sentence."},
        ],
        "max_new_tokens": 16,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rounds = [[item("warmup", args.image, True)]]
    order = [(mode, repetition) for repetition in range(5) for mode in ("off", "on")]
    random.Random(20260913).shuffle(order)
    rounds.extend([item(f"cache-{mode}-{repetition}", args.image, mode == "on")]
                  for mode, repetition in order)
    with open(args.output, "w") as stream:
        json.dump({"seed": 20260913, "rounds": rounds}, stream)


if __name__ == "__main__":
    main()
