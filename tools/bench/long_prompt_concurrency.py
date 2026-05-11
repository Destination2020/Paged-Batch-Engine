#!/usr/bin/env python3
"""Concurrent long-prompt load generator for the online serving path.

This script intentionally uses only the Python standard library so it can run
on bare benchmark machines. It targets the /generate endpoint by default.
"""

import argparse
import concurrent.futures
import http.client
import json
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


AIRPORT_PARAGRAPH = (
    "Airport operations incident log segment {segment:04d} for request {request:04d}. "
    "Gate assignment changed after late inbound arrival. Ground crew recorded baggage "
    "cart staging, fueling confirmation, catering delay, runway inspection note, "
    "weather update, crew duty limit check, passenger rebooking queue, and tower "
    "handoff timing. The dispatcher must summarize dependencies, detect conflicts, "
    "rank operational risks, and produce a concise recovery plan without omitting "
    "tail numbers, gate identifiers, timestamps, or resource constraints. "
)


LEGAL_PARAGRAPH = (
    "Document review section {segment:04d} for request {request:04d}. The record "
    "contains contract clauses, exception notes, amendment history, counterparty "
    "obligations, audit findings, approval chains, and unresolved comments. The "
    "assistant must preserve chronology, identify contradictions, and cite the "
    "relevant section identifiers while preparing a risk-oriented summary. "
)


CODE_PARAGRAPH = (
    "Repository analysis block {segment:04d} for request {request:04d}. The module "
    "contains scheduler state transitions, memory allocation hints, queue admission "
    "rules, GPU worker calls, error propagation notes, and benchmark observations. "
    "Explain the control flow, isolate potential failure points, and propose a "
    "minimal patch plan with validation steps. "
)


TEMPLATES = {
    "airport": AIRPORT_PARAGRAPH,
    "legal": LEGAL_PARAGRAPH,
    "code": CODE_PARAGRAPH,
}


@dataclass
class RequestResult:
    index: int
    ok: bool
    status: Optional[int]
    elapsed_s: float
    body_bytes: int
    text_len: int
    error: str


def build_prompt(target_chars: int, request_index: int, template_name: str,
                 unique_prompts: bool) -> str:
    paragraph = TEMPLATES[template_name]
    header = (
        "You are analyzing a very long operational context. "
        "Return only the final answer after reading all records.\n\n"
    )
    if unique_prompts:
        header += (
            f"Unique request marker: request-{request_index:04d}. "
            "This marker appears throughout the prompt to avoid full prefix reuse.\n\n"
        )

    parts = [header]
    segment = 0
    while sum(len(part) for part in parts) < target_chars:
        marker_request = request_index if unique_prompts else 0
        parts.append(paragraph.format(segment=segment, request=marker_request))
        if segment % 8 == 7:
            parts.append("\n")
        segment += 1

    prompt = "".join(parts)
    return prompt[:target_chars]


def post_json(host: str, port: int, path: str, payload: Dict,
              timeout_s: float) -> Tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        data = response.read()
        return response.status, data
    finally:
        conn.close()


def get_metrics(host: str, port: int, timeout_s: float) -> Optional[str]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        data = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            return f"metrics_status={response.status} body={data}"
        return data
    except Exception as exc:  # pylint: disable=broad-except
        return f"metrics_error={type(exc).__name__}: {exc}"
    finally:
        conn.close()


def run_one(index: int, args: argparse.Namespace) -> RequestResult:
    prompt = build_prompt(
        args.prompt_chars,
        index,
        args.template,
        not args.shared_prompt,
    )
    payload = {
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_eos": args.ignore_eos,
        "stream": False,
        "timeout_ms": args.request_timeout_ms,
    }
    start = time.perf_counter()
    try:
        status, body = post_json(args.host, args.port, args.path, payload,
                                 args.http_timeout_s)
        elapsed = time.perf_counter() - start
        text_len = 0
        error = ""
        try:
            decoded = json.loads(body.decode("utf-8", errors="replace"))
            if isinstance(decoded, dict):
                text_len = len(decoded.get("text", ""))
                if "error" in decoded:
                    error = str(decoded["error"])
        except json.JSONDecodeError:
            error = body[:200].decode("utf-8", errors="replace")
        return RequestResult(
            index=index,
            ok=200 <= status < 300,
            status=status,
            elapsed_s=elapsed,
            body_bytes=len(body),
            text_len=text_len,
            error=error,
        )
    except Exception as exc:  # pylint: disable=broad-except
        elapsed = time.perf_counter() - start
        return RequestResult(
            index=index,
            ok=False,
            status=None,
            elapsed_s=elapsed,
            body_bytes=0,
            text_len=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(rank, len(ordered) - 1))]


def print_summary(results: List[RequestResult], wall_s: float) -> None:
    ok = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    latencies = [item.elapsed_s for item in results]
    status_counts: Dict[str, int] = {}
    for item in results:
        key = str(item.status) if item.status is not None else "exception"
        status_counts[key] = status_counts.get(key, 0) + 1

    print("\n=== Summary ===")
    print(f"requests={len(results)} ok={len(ok)} failed={len(failed)} wall_s={wall_s:.3f}")
    print(f"status_counts={json.dumps(status_counts, sort_keys=True)}")
    if latencies:
        print(
            "latency_s "
            f"avg={statistics.mean(latencies):.3f} "
            f"p50={percentile(latencies, 50):.3f} "
            f"p95={percentile(latencies, 95):.3f} "
            f"max={max(latencies):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send concurrent long-prompt /generate requests."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18181)
    parser.add_argument("--path", default="/generate")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-chars", type=int, default=6000)
    parser.add_argument("--template", choices=sorted(TEMPLATES), default="airport")
    parser.add_argument("--shared-prompt", action="store_true",
                        help="Use identical prompts. Default makes each prompt unique.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--request-timeout-ms", type=int, default=300000)
    parser.add_argument("--http-timeout-s", type=float, default=360.0)
    parser.add_argument("--stagger-ms", type=int, default=0,
                        help="Delay between submissions from the driver.")
    parser.add_argument("--metrics", action="store_true",
                        help="Fetch /metrics after the run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.prompt_chars <= 0:
        raise SystemExit("requests, concurrency, and prompt-chars must be positive")

    sample_prompt = build_prompt(args.prompt_chars, 0, args.template,
    
                                 not args.shared_prompt)
    print("=== Long Prompt Concurrency Load ===")
    print(
        f"target=http://{args.host}:{args.port}{args.path} "
        f"requests={args.requests} concurrency={args.concurrency} "
        f"prompt_chars={len(sample_prompt)} template={args.template} "
        f"shared_prompt={args.shared_prompt} max_new_tokens={args.max_new_tokens}"
    )

    start = time.perf_counter()
    results: List[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for index in range(args.requests):
            futures.append(pool.submit(run_one, index, args))
            if args.stagger_ms > 0:
                time.sleep(args.stagger_ms / 1000.0)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = result.status if result.status is not None else "exception"
            print(
                f"request={result.index:04d} ok={int(result.ok)} "
                f"status={status} elapsed_s={result.elapsed_s:.3f} "
                f"text_len={result.text_len} body_bytes={result.body_bytes} "
                f"error={result.error[:180]}"
            )

    wall_s = time.perf_counter() - start
    results.sort(key=lambda item: item.index)
    print_summary(results, wall_s)

    if args.metrics:
        print("\n=== Metrics ===")
        print(get_metrics(args.host, args.port, args.http_timeout_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
