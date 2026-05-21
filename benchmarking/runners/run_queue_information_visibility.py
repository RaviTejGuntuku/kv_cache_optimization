#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from headroom_common import ROOT, WorkloadRef, profile_payload, resolve_timestamped_output_root, write_manifest


WORKLOADS = [
    WorkloadRef(
        label="natural_tenant_rotation_gap",
        path="datasets/synthetic/adversarial_fcfs/natural_tenant_rotation_gap.jsonl",
    ),
    WorkloadRef(
        label="natural_periodic_refinement_gap",
        path="datasets/synthetic/adversarial_fcfs/natural_periodic_refinement_gap.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [16],
        "queue_sizes": [1, 4, 16, "full"],
        "max_rows": 64,
        "estimated_runtime_hours": 0.02,
    },
    "full": {
        "page_sizes": [16],
        "queue_sizes": [1] + list(range(10, 801, 10)) + ["full"],
        "max_rows": None,
        "estimated_runtime_hours": 0.08,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline queue-information visibility study."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--page-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--queue-sizes", nargs="+", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, max_rows: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def shared_blocks(row: dict, page_size: int) -> int:
    meta = row.get("metadata", {})
    shared_prefix_tokens = int(meta.get("shared_prefix_tokens", 0) or 0)
    if shared_prefix_tokens <= 0:
        return 0
    return (shared_prefix_tokens + page_size - 1) // page_size


def family_of(row: dict) -> str | None:
    meta = row.get("metadata", {})
    if meta.get("kind") == "unique":
        return None
    family = meta.get("family")
    return family if family else None


def compute_visibility_rows(rows: list[dict], *, workload: str, page_size: int, queue_sizes: list[str | int]) -> list[dict]:
    family_positions: dict[str, list[int]] = defaultdict(list)
    block_counts: list[int] = []
    families: list[str | None] = []

    for idx, row in enumerate(rows):
        fam = family_of(row)
        families.append(fam)
        blocks = shared_blocks(row, page_size)
        block_counts.append(blocks)
        if fam is not None and blocks > 0:
            family_positions[fam].append(idx)

    total_reuse_events = 0
    total_reuse_blocks = 0
    total_next_reuses = 0

    per_q = {
        q: {
            "visible_reuse_events": 0,
            "visible_reuse_blocks": 0,
            "visible_next_reuses": 0,
        }
        for q in queue_sizes
    }

    for idx, fam in enumerate(families):
        if fam is None:
            continue
        blocks = block_counts[idx]
        if blocks <= 0:
            continue
        future_positions = [pos for pos in family_positions[fam] if pos > idx]
        if not future_positions:
            continue

        total_reuse_events += len(future_positions)
        total_reuse_blocks += blocks * len(future_positions)
        total_next_reuses += 1

        for q in queue_sizes:
            horizon_end = len(rows) - 1 if q == "full" else min(len(rows) - 1, idx + int(q))
            visible_count = sum(1 for pos in future_positions if pos <= horizon_end)
            per_q[q]["visible_reuse_events"] += visible_count
            per_q[q]["visible_reuse_blocks"] += blocks * visible_count
            if future_positions[0] <= horizon_end:
                per_q[q]["visible_next_reuses"] += 1

    out_rows: list[dict] = []
    for q in queue_sizes:
        q_numeric = len(rows) if q == "full" else int(q)
        data = per_q[q]
        out_rows.append(
            {
                "workload": workload,
                "page_size": page_size,
                "queue_size_q": str(q),
                "queue_size_numeric": q_numeric,
                "total_reuse_events": total_reuse_events,
                "visible_reuse_events": data["visible_reuse_events"],
                "reuse_event_visibility_fraction": (
                    data["visible_reuse_events"] / total_reuse_events if total_reuse_events else 0.0
                ),
                "total_reuse_blocks": total_reuse_blocks,
                "visible_reuse_blocks": data["visible_reuse_blocks"],
                "reuse_block_visibility_fraction": (
                    data["visible_reuse_blocks"] / total_reuse_blocks if total_reuse_blocks else 0.0
                ),
                "total_next_reuses": total_next_reuses,
                "visible_next_reuses": data["visible_next_reuses"],
                "next_reuse_visibility_fraction": (
                    data["visible_next_reuses"] / total_next_reuses if total_next_reuses else 0.0
                ),
            }
        )
    return out_rows


def threshold_rows(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    thresholds = [0.50, 0.75, 0.90, 0.95]
    metrics = [
        "reuse_event_visibility_fraction",
        "reuse_block_visibility_fraction",
        "next_reuse_visibility_fraction",
    ]
    for workload in sorted({row["workload"] for row in rows}):
        wrows = sorted(
            [row for row in rows if row["workload"] == workload],
            key=lambda row: float(row["queue_size_numeric"]),
        )
        for metric in metrics:
            for threshold in thresholds:
                match = next((row for row in wrows if float(row[metric]) >= threshold), None)
                output.append(
                    {
                        "workload": workload,
                        "metric": metric,
                        "threshold": threshold,
                        "smallest_queue_size_q": match["queue_size_q"] if match else None,
                        "smallest_queue_size_numeric": match["queue_size_numeric"] if match else None,
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    page_sizes = args.page_sizes or profile["page_sizes"]
    queue_sizes = args.queue_sizes or profile["queue_sizes"]
    queue_sizes = [int(q) if str(q).isdigit() else str(q) for q in queue_sizes]
    max_rows = args.max_rows if args.max_rows is not None else profile["max_rows"]
    output_root = resolve_timestamped_output_root(args.output_root, no_timestamp=args.no_timestamp)
    manifest = profile_payload(
        "queue_information_visibility",
        args.mode,
        {
            "invoked_at_utc": output_root.name.rsplit("__", 1)[-1] if "__" in output_root.name else None,
            "resolved_output_root": str(output_root),
            "workloads": [workload.path for workload in WORKLOADS],
            "page_sizes": page_sizes,
            "queue_sizes": [str(q) for q in queue_sizes],
            "max_rows": max_rows,
            "estimated_runtime_hours": profile["estimated_runtime_hours"],
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    metrics_rows: list[dict] = []
    inputs_dir = output_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    for workload in WORKLOADS:
        src = ROOT / workload.path
        rows = load_rows(src, max_rows=max_rows)
        sampled_path = inputs_dir / f"{workload.label}.jsonl"
        sampled_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        for page_size in page_sizes:
            metrics_rows.extend(
                compute_visibility_rows(
                    rows,
                    workload=workload.label,
                    page_size=page_size,
                    queue_sizes=queue_sizes,
                )
            )

    threshold_summary = threshold_rows(metrics_rows)
    metrics_dir = output_root / "metrics"
    write_csv(metrics_dir / "visibility_metrics.csv", metrics_rows)
    write_manifest(metrics_dir / "visibility_metrics.json", {"rows": metrics_rows})
    write_csv(metrics_dir / "threshold_summary.csv", threshold_summary)
    write_manifest(metrics_dir / "threshold_summary.json", {"rows": threshold_summary})

    if args.dry_run:
        return

    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarking" / "analysis_scripts" / "plot_queue_information_visibility.py"),
                "--experiment-root",
                str(output_root),
            ],
            check=True,
            cwd=str(ROOT),
        )
    except subprocess.CalledProcessError as exc:
        write_manifest(
            output_root / "plotting_status.json",
            {
                "status": "failed",
                "reason": str(exc),
                "note": "Metrics were generated successfully, but plotting failed. Install matplotlib in the active Python environment to materialize graphs.",
            },
        )


if __name__ == "__main__":
    main()
