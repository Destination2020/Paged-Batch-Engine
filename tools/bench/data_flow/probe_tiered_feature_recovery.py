#!/usr/bin/env python3
"""Executable E3 gate for real Vision feature GPU/Host demote and restore."""
import argparse
import json

from pbe_roles.vision.client import call


def encode(endpoint: str, image: str, request_id: str, text: str) -> dict:
    result = call(endpoint, {
        "op": "encode", "request_id": request_id, "generation": 1,
        "timeout_ms": 30000,
        "parts": [{"type": "image", "path": image, "size": 224},
                  {"type": "text", "text": text}],
    })
    assert result.get("ok"), result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    producer = encode(args.endpoint, args.image, "feature-producer",
                      "Describe this image in one sentence.")
    before = call(args.endpoint, {"op": "feature_status"})
    demotion = call(args.endpoint, {"op": "demote_features"})
    demoted = call(args.endpoint, {"op": "feature_status"})
    restoration = call(args.endpoint, {"op": "restore_features"})
    restored = call(args.endpoint, {"op": "feature_status"})
    consumer = encode(args.endpoint, args.image, "feature-consumer",
                      "What are the most prominent colors?")

    assert producer["feature_cache_tier"] == "miss"
    assert before["gpu_feature_objects"] == 1 and before["gpu_feature_bytes"] > 0
    assert demotion["demoted_objects"] == 1 and demotion["demoted_bytes"] > 0
    assert demoted["gpu_feature_objects"] == 0 and demoted["gpu_feature_bytes"] == 0
    assert restoration["restored_objects"] == 1
    assert restoration["restored_bytes"] == demotion["demoted_bytes"]
    assert restored["gpu_feature_objects"] == 1
    assert restored["peak_process_allocated_bytes"] < restored["device_admission_limit"]
    assert consumer["feature_gpu_cache_hit"] and consumer["forward_ms"] == 0.0
    assert consumer["feature_content"] == producer["feature_content"]
    assert consumer["feature_representation"] == producer["feature_representation"]
    output = {"ok": True, "producer": producer, "before": before,
              "demotion": demotion, "demoted": demoted,
              "restoration": restoration, "restored": restored,
              "consumer": consumer}
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"ok": True, "demoted_bytes": demotion["demoted_bytes"],
                      "restored_bytes": restoration["restored_bytes"],
                      "consumer_tier": consumer["feature_cache_tier"]}, sort_keys=True))


if __name__ == "__main__":
    main()
