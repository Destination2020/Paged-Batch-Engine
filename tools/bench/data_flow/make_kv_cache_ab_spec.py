#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def request(name):
        return {"request_id": name, "generation": 1,
                "parts": [{"type": "image", "path": args.image, "size": 224},
                          {"type": "text", "text": "Describe this image in one sentence."}],
                "max_new_tokens": 16}
    json.dump({"rounds": [[request("producer")], [request("replay")]]},
              open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
