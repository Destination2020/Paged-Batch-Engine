#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    requests = [{"request_id": f"duplicate-{index}", "generation": 1,
                 "parts": [{"type": "image", "path": args.image, "size": 280},
                           {"type": "text", "text": f"Describe duplicate {index}."}],
                 "max_new_tokens": 4} for index in range(8)]
    json.dump({"rounds": [requests]}, open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
