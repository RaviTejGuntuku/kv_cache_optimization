#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process completed two-pass run directories under an experiment root. "
            "This runs trace analysis, comparison, and memory-pressure estimation locally."
        )
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip run roots that already have reports/comparison.json.",
    )
    parser.add_argument(
        "--workload-filter",
        nargs="*",
        default=None,
        help="Optional substrings used to restrict which run roots are processed.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of run roots to post-process in parallel.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=str(cwd))


def should_include(run_root: Path, filters: list[str] | None) -> bool:
    if not filters:
        return True
    lowered = str(run_root).lower()
    return any(fragment in lowered for fragment in filters)


def discover_run_roots(experiment_root: Path, only_missing: bool, filters: list[str] | None) -> list[Path]:
    run_roots: list[Path] = []
    for run_root in sorted(experiment_root.glob("workloads/*/sched-*/mc-*")):
        if not should_include(run_root, filters):
            continue
        required = [
            run_root / "traces" / "lru.jsonl",
            run_root / "traces" / "belady.jsonl",
            run_root / "benchmarks" / "lru.jsonl",
            run_root / "benchmarks" / "belady.jsonl",
        ]
        if not all(path.exists() for path in required):
            continue
        comparison = run_root / "reports" / "comparison.json"
        if only_missing and comparison.exists():
            continue
        run_roots.append(run_root)
    return run_roots


def process_run_root(run_root_str: str) -> str:
    root = Path(__file__).resolve().parents[2]
    run_root = Path(run_root_str)
    metadata = load_json(run_root / "run_metadata.json")
    run_args = metadata.get("args", {})
    page_size = int(run_args.get("page_size", 16))
    block_capacity = int(run_args.get("gpu_kv_capacity_blocks", 20000))
    dataset_path = run_args["dataset_path"]
    max_concurrency = int(run_args.get("max_concurrency") or run_args.get("num_prompts"))

    traces_dir = run_root / "traces"
    analysis_dir = run_root / "analysis"
    reports_dir = run_root / "reports"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lru_trace = traces_dir / "lru.jsonl"
    belady_trace = traces_dir / "belady.jsonl"
    lru_analysis = analysis_dir / "lru"
    belady_analysis = analysis_dir / "belady"
    lru_bench = run_root / "benchmarks" / "lru.jsonl"
    belady_bench = run_root / "benchmarks" / "belady.jsonl"
    final_report = reports_dir / "comparison.json"
    pressure_report = reports_dir / "memory_pressure.json"

    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "analyze_kv_trace.py"),
            "--trace",
            str(lru_trace),
            "--output-dir",
            str(lru_analysis),
            "--block-capacity",
            str(block_capacity),
        ],
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "analyze_kv_trace.py"),
            "--trace",
            str(belady_trace),
            "--output-dir",
            str(belady_analysis),
            "--block-capacity",
            str(block_capacity),
        ],
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "compare_benchmark_runs.py"),
            "--lru-bench",
            str(lru_bench),
            "--belady-bench",
            str(belady_bench),
            "--lru-trace-summary",
            str(lru_analysis / "summary.json"),
            "--belady-trace-summary",
            str(belady_analysis / "summary.json"),
            "--output",
            str(final_report),
            "--page-size",
            str(page_size),
        ],
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "benchmarking" / "analysis_scripts" / "estimate_memory_pressure.py"),
            "--dataset",
            dataset_path,
            "--gpu-kv-capacity-blocks",
            str(block_capacity),
            "--page-size",
            str(page_size),
            "--concurrency",
            str(max_concurrency),
            "--output",
            str(pressure_report),
        ],
        cwd=root,
    )
    return str(run_root)


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    run_roots = discover_run_roots(
        experiment_root,
        only_missing=args.only_missing,
        filters=[item.lower() for item in args.workload_filter] if args.workload_filter else None,
    )
    if not run_roots:
        print("No run roots selected for post-processing.")
        return

    if args.jobs <= 1:
        for run_root in run_roots:
            completed = process_run_root(str(run_root))
            print(f"Post-processed {completed}")
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(process_run_root, str(run_root)): run_root for run_root in run_roots
        }
        for future in concurrent.futures.as_completed(future_map):
            run_root = future_map[future]
            completed = future.result()
            print(f"Post-processed {completed}")


if __name__ == "__main__":
    main()
