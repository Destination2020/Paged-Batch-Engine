#!/usr/bin/env python3
"""Plot measured HTTP request lifetimes, not GPU kernel occupancy (matplotlib)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    mixed = report["mixed_load"]
    rows = mixed["extreme"] + mixed["short"]
    origin = min(row["start_s"] for row in rows)
    labels = ["Long input A", "Long input B", "Short input / long output"] + [
        f"Short request {i + 1}" for i in range(len(mixed["short"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (row, label) in enumerate(zip(rows, labels)):
        start = row["start_s"] - origin
        duration = row["end_s"] - row["start_s"]
        color = "#4872a8" if i < 3 else "#169b82"
        axes[0].barh(i, duration, left=start, height=0.58, color=color)
        axes[0].plot(start + row["first_text_ms"] / 1000, i, "|", color="#172b4d", ms=10)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Seconds since mixed load began")
    axes[0].set_title("Measured HTTP request lifetimes")
    short_origin = min(row["start_s"] for row in mixed["short"])
    for i, row in enumerate(mixed["short"]):
        start = (row["start_s"] - short_origin) * 1000
        axes[1].barh(i, row["total_ms"], left=start, height=0.58, color="#169b82")
        axes[1].plot(start + row["first_text_ms"], i, "|", color="#172b4d", ms=10)
        axes[1].text(
            start + row["total_ms"] + 3, i, f'{row["total_ms"]:.1f} ms', va="center", fontsize=9
        )
    axes[1].set_yticks(range(len(mixed["short"])), ["Short 1", "Short 2", "Short 3"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Milliseconds since short burst began")
    axes[1].set_title("Short requests during long-output load")
    axes[1].margins(x=0.3)
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    tokens = mixed["extreme"][0]["requested_output_tokens"]
    fig.suptitle(f"Qwen2.5-1.5B · one GPU · 2 Prefill + 1 Decode · 3 × {tokens}-token outputs")
    fig.text(
        0.5,
        0.01,
        "Markers: first text received. Host timestamps; not a GPU kernel timeline or a latency guarantee.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
