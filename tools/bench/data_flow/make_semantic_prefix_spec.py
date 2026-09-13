#!/usr/bin/env python3
"""Build the real-VLM E1 semantic prefix positive/negative fixture."""
import argparse
import json


def request(request_id: str, image: str, size: int, question: str) -> dict:
    result = {
        "request_id": request_id,
        "generation": 1,
        "parts": [
            {"type": "image", "path": image, "size": size},
            {"type": "text", "text": question},
        ],
        "max_new_tokens": 8,
    }
    if request_id in {"producer", "exact-replay"}:
        result["diagnostic_key"] = "e1-exact-replay"
        result["diagnostic_mode"] = request_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--different-image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    common = "Describe this image in one sentence."
    rounds = [
        [request("producer", args.image, 224, common)],
        [request("exact-replay", args.image, 224, common)],
        [request("same-image-different-question", args.image, 224,
                 "What colors are most prominent in this image?")],
        [request("different-processor", args.image, 252, common)],
        [request("different-image", args.different_image, 224, common)],
    ]
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump({"rounds": rounds}, stream, indent=2)


if __name__ == "__main__":
    main()
