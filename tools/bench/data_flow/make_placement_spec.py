#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def request(name, question, tokens=8, **extra):
        value = {"request_id": name, "generation": 1,
                 "parts": [{"type": "image", "path": args.image, "size": 224},
                           {"type": "text", "text": question}],
                 "max_new_tokens": tokens, "timeout_ms": 120000}
        value.update(extra)
        return value
    hot = "Describe this image in one sentence."
    rounds = [
        [request("warm-prefix", hot, force_worker=0)],
        [request("long-queue", "Explain every visible component in detail.", 64,
                 force_worker=0),
         request("cache-versus-queue", hot)],
        [request("fresh-cache-choice", hot)],
        [request("stale-stat-choice", hot, stale_worker=0)],
        [request("budget-fallback", hot, unavailable_worker=0)],
        [request("worker-restart", hot, restart_worker_before=1)],
        [request("random-a", "Name the main diagram components.", shuffle_dispatch=True),
         request("random-b", "Summarize the arrows in the diagram."),
         request("random-c", "What is the apparent topic of the image?"),
         request("random-d", "Give a short caption for the image.")],
    ]
    json.dump({"rounds": rounds}, open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
