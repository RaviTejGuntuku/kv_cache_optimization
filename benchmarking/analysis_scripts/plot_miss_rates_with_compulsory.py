#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LRU, Belady, and compulsory block miss rates from synced traces."
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def trace_stats(trace_path: Path) -> dict[str, float | int]:
    total_blocks = 0
    unique_blocks: set[int] = set()
    missed_blocks = 0
    matched_blocks = 0

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "request_lookup":
                blocks = [int(block) for block in event.get("block_hashes", [])]
                total_blocks += len(blocks)
                unique_blocks.update(blocks)
            elif event.get("event") == "match_result":
                missed_blocks += int(event.get("missed_blocks", 0))
                matched_blocks += int(event.get("matched_blocks", 0))

    match_total = matched_blocks + missed_blocks
    denominator = match_total or total_blocks
    return {
        "total_blocks": denominator,
        "missed_blocks": missed_blocks,
        "miss_rate": missed_blocks / denominator if denominator else 0.0,
        "compulsory_misses": len(unique_blocks),
        "compulsory_miss_rate": len(unique_blocks) / denominator if denominator else 0.0,
    }


def run_metadata(run_root: Path) -> tuple[str, int]:
    metadata = json.loads((run_root / "run_metadata.json").read_text(encoding="utf-8"))
    args = metadata.get("args", {})
    workload = Path(args["dataset_path"]).stem
    page_size = int(args.get("page_size", 0))
    return workload, page_size


def collect_rows(experiment_root: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for run_root in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        lru_trace = run_root / "traces" / "lru.jsonl"
        belady_trace = run_root / "traces" / "belady.jsonl"
        if not lru_trace.exists() or not belady_trace.exists():
            continue

        workload, page_size = run_metadata(run_root)
        lru = trace_stats(lru_trace)
        belady = trace_stats(belady_trace)
        rows.append(
            {
                "run": run_root.name,
                "workload": workload,
                "page_size": page_size,
                "lru_total_blocks": lru["total_blocks"],
                "lru_missed_blocks": lru["missed_blocks"],
                "lru_miss_rate": lru["miss_rate"],
                "belady_total_blocks": belady["total_blocks"],
                "belady_missed_blocks": belady["missed_blocks"],
                "belady_miss_rate": belady["miss_rate"],
                "compulsory_misses_lru_trace": lru["compulsory_misses"],
                "compulsory_miss_rate_lru_trace": lru["compulsory_miss_rate"],
                "compulsory_misses_belady_trace": belady["compulsory_misses"],
                "compulsory_miss_rate_belady_trace": belady["compulsory_miss_rate"],
                "compulsory_misses": min(
                    int(lru["compulsory_misses"]), int(belady["compulsory_misses"])
                ),
                "compulsory_miss_rate": min(
                    float(lru["compulsory_miss_rate"]),
                    float(belady["compulsory_miss_rate"]),
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_workload(rows: list[dict[str, float | int | str]], workload: str, output_dir: Path) -> None:
    subset = sorted(
        [row for row in rows if row["workload"] == workload],
        key=lambda row: int(row["page_size"]),
    )
    if not subset:
        return

    labels = [str(row["page_size"]) for row in subset]
    x = list(range(len(subset)))
    width = 0.25

    def bar(
        lru_key: str,
        belady_key: str,
        compulsory_key: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            [item - width for item in x],
            [float(row[lru_key]) for row in subset],
            width=width,
            label="LRU",
        )
        ax.bar(
            x,
            [float(row[belady_key]) for row in subset],
            width=width,
            label="Belady",
        )
        ax.bar(
            [item + width for item in x],
            [float(row[compulsory_key]) for row in subset],
            width=width,
            label="Compulsory",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Page Size (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    safe_name = workload.replace("_", "-")
    bar(
        "lru_miss_rate",
        "belady_miss_rate",
        "compulsory_miss_rate",
        "Block Miss Rate",
        f"{workload}: LRU vs Belady vs Compulsory Miss Rate",
        f"{safe_name}__miss_rate_with_compulsory.png",
    )
    bar(
        "lru_missed_blocks",
        "belady_missed_blocks",
        "compulsory_misses",
        "Block Miss Count",
        f"{workload}: LRU vs Belady vs Compulsory Miss Count",
        f"{safe_name}__miss_count_with_compulsory.png",
    )


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(experiment_root)
    if not rows:
        raise SystemExit(f"No completed traces found under {experiment_root}")

    write_csv(rows, output_dir / "miss_rates_with_compulsory.csv")
    (output_dir / "miss_rates_with_compulsory.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    for workload in sorted({str(row["workload"]) for row in rows}):
        plot_workload(rows, workload, output_dir)


if __name__ == "__main__":
    main()
