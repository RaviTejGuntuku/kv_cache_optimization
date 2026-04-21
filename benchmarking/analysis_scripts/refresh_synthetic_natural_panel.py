#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh local plots for the synthetic natural panel from any completed runs "
            "already synced into the local experiment directory."
        )
    )
    parser.add_argument(
        "--experiment-root",
        default="runs/experiments/synthetic-natural-panel",
        help="Local experiment root containing workload subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots/synthetic-natural-panel-live",
        help="Directory where aggregated metrics and plots are written.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Refresh interval when --watch is enabled.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously rebuild plots as new completed runs arrive.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_status(experiment_root: Path) -> dict[str, Any]:
    plan_path = experiment_root / "experiment_plan.json"
    plan = load_json(plan_path) if plan_path.exists() else {}
    planned_runs = plan.get("runs", [])

    completed_runs = []
    partial_runs = []
    run_roots = sorted(experiment_root.glob("workloads/*/sched-*/mc-*"))
    for run_root in run_roots:
        comparison_path = run_root / "reports" / "comparison.json"
        metadata_path = run_root / "run_metadata.json"
        if comparison_path.exists():
            completed_runs.append(str(run_root))
        elif metadata_path.exists():
            partial_runs.append(str(run_root))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(experiment_root),
        "planned_run_count": len(planned_runs),
        "completed_run_count": len(completed_runs),
        "partial_run_count": len(partial_runs),
        "completed_run_roots": completed_runs,
        "partial_run_roots": partial_runs,
    }


def refresh_once(experiment_root: Path, output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = collect_status(experiment_root)
    status_path = output_dir / "live_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    if status["completed_run_count"] == 0:
        print(f"[refresh] no completed runs under {experiment_root}", flush=True)
        return 0, status["partial_run_count"]

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("summarize_synthetic_workload_matrix.py")),
        "--experiment-root",
        str(experiment_root),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    print(
        (
            f"[refresh] wrote plots to {output_dir} "
            f"(completed={status['completed_run_count']} partial={status['partial_run_count']})"
        ),
        flush=True,
    )
    return status["completed_run_count"], status["partial_run_count"]


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)

    if not args.watch:
        refresh_once(experiment_root, output_dir)
        return

    last_signature: tuple[int, int] | None = None
    while True:
        try:
            signature = refresh_once(experiment_root, output_dir)
            last_signature = signature
        except Exception as exc:
            print(f"[refresh] failed: {exc}", file=sys.stderr, flush=True)
            if last_signature is not None:
                print(
                    f"[refresh] last successful state: completed={last_signature[0]} partial={last_signature[1]}",
                    file=sys.stderr,
                    flush=True,
                )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
