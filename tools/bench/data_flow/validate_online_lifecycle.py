#!/usr/bin/env python3
"""Executable M5/M7 gate for asynchronous cancel, deadline and device budget."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("trace")
    parser.add_argument("--deadline-ms", type=float, required=True)
    args = parser.parse_args()

    with open(args.result, encoding="utf-8") as stream:
        result = json.load(stream)
    with open(args.trace, encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]

    assert result["ok"] and len(result["rounds"]) == 1
    round_result = result["rounds"][0]
    outputs = {item["request_id"]: item for item in round_result["outputs"]}
    cancelled = outputs["external-cancel"]
    deadline = outputs["cumulative-deadline"]
    normal = outputs["normal"]

    assert cancelled["failed"] and cancelled["finish_reason"] == "cancelled_by_client"
    assert 0 < len(cancelled["tokens"]) < 8, "cancel must interrupt active decode"
    assert deadline["failed"] and deadline["finish_reason"] == "deadline_exceeded"
    assert not normal["failed"] and len(normal["tokens"]) == 8
    assert round_result["encode_join_ms"] > args.deadline_ms / 2
    assert args.deadline_ms <= round_result["round_latency_ms"] < args.deadline_ms + 1500
    assert round_result["worker_pid"] == result["status"]["worker_pid"]

    lifecycle = [item for item in trace
                 if item.get("stage") == "external_lifecycle_control"]
    assert any(any(ack.get("event") == "cancel_ack" and
                       ack.get("request_id") == "external-cancel" and ack.get("ok")
                       for ack in item.get("results", []))
               for item in lifecycle)

    status = result["status"]
    memory = status["device_memory"]
    assert status["budget_invariant"] and memory["within_admission_limit"]
    assert memory["model_process_allocation_bytes"] > memory["model_file_bytes"] > 0
    assert memory["workspace_reserved_bytes"] > 0
    assert memory["preexisting_used_bytes"] > 0
    assert memory["external_kv_owner_bytes"] > 0
    accounted = (memory["preexisting_used_bytes"] +
                 memory["model_process_allocation_bytes"] +
                 memory["external_kv_owner_bytes"])
    assert memory["peak_observed_used_bytes"] >= accounted
    assert status["pool"]["active_grants"] == 1
    for name in ("kv", "bundles", "staging"):
        assert status["budget"][name]["used"] == 0

    print(json.dumps({
        "ok": True,
        "cancelled_after_tokens": len(cancelled["tokens"]),
        "encode_join_ms": round_result["encode_join_ms"],
        "deadline_round_ms": round_result["round_latency_ms"],
        "worker_pid": round_result["worker_pid"],
        "peak_observed_used_bytes": memory["peak_observed_used_bytes"],
        "admission_limit_bytes": memory["admission_limit_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
