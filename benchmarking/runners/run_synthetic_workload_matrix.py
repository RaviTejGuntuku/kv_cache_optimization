#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkloadStats:
    dataset_path: Path
    workload_name: str
    num_requests: int
    mean_prompt_len: float
    mean_output_len: float
    pressure_by_concurrency: list[dict[str, float]]
    selected_concurrencies: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic workload matrix: every generated workload, both FCFS and "
            "prefix-coverage scheduling, and five concurrency points per workload."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Experiment root. Runs are nested by workload/scheduler/concurrency underneath it.",
    )
    parser.add_argument(
        "--data-root",
        default="data/synthetic",
        help="Synthetic workload root. All *.jsonl files under this tree are included.",
    )
    parser.add_argument(
        "--schedule-policies",
        nargs="+",
        default=["fcfs", "prefix-coverage"],
        help="Scheduler policies to test. Defaults to fcfs and prefix-coverage.",
    )
    parser.add_argument(
        "--request-rate",
        default="inf",
        help="Arrival rate passed into the serving benchmark. Default keeps the queue backlogged.",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", default="0.7")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--candidate-concurrencies", type=int, nargs="+", default=[
        4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512
    ])
    parser.add_argument(
        "--target-pressures",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 1.00, 1.25, 1.50],
        help="Target full-request pressure points used to pick five concurrency levels.",
    )
    parser.add_argument(
        "--num-concurrency-points",
        type=int,
        default=1,
        help="How many concurrency levels to select per workload.",
    )
    parser.add_argument(
        "--single-concurrency-strategy",
        choices=["max-pressure", "closest-to-one"],
        default="max-pressure",
        help=(
            "When --num-concurrency-points=1, either choose the highest-pressure point "
            "or the point whose full-request pressure is closest to 1.0."
        ),
    )
    parser.add_argument(
        "--num-prompts-override",
        type=int,
        default=None,
        help="Optional cap on prompts per run. Defaults to the full workload size.",
    )
    parser.add_argument(
        "--workload-filter",
        nargs="*",
        default=None,
        help="Optional substrings. If provided, only workloads whose path contains one match are used.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose reports/comparison.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize the experiment plan and print runtime estimates.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the run matrix across shards for multi-GPU execution.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index. Only runs assigned to this shard are executed.",
    )
    parser.add_argument(
        "--assumed-output-throughput-toks-per-s",
        type=float,
        default=90.0,
        help="Used only for planning. Default is calibrated from existing Qwen-7B synthetic runs.",
    )
    parser.add_argument(
        "--assumed-server-startup-seconds",
        type=float,
        default=90.0,
        help="Used only for planning. Applied once per pass, so twice per run.",
    )
    parser.add_argument(
        "--assumed-analysis-seconds",
        type=float,
        default=20.0,
        help="Used only for planning. Flat post-processing overhead per two-pass run.",
    )
    parser.add_argument(
        "--postprocess-mode",
        choices=["inline", "skip"],
        default="inline",
        help=(
            "Whether each run performs trace analysis/comparison inline on the GPU node "
            "or stops after writing traces/benchmarks for later local post-processing."
        ),
    )
    return parser.parse_args()


def coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        for item in value:
            candidate = coerce_text(item)
            if candidate:
                return candidate
    if isinstance(value, dict):
        for key in ("content", "value", "text"):
            if key in value:
                candidate = coerce_text(value[key])
                if candidate:
                    return candidate
    return None


def load_lengths(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_no} in {path}") from exc

            conversation = record.get("conversations", [])
            prompt = coerce_text(conversation[0]) if len(conversation) > 0 else ""
            answer = coerce_text(conversation[1]) if len(conversation) > 1 else ""
            prompt_len = record.get("prompt_len")
            if prompt_len is None:
                prompt_len = max(1, math.ceil(len(prompt.split()) * 1.3)) if prompt else 0
            output_len = record.get("output_len")
            if output_len is None:
                output_len = max(1, math.ceil(len(answer.split()) * 1.3)) if answer else 256
            rows.append({"prompt_len": int(prompt_len), "output_len": int(output_len)})
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def workload_slug(path: Path, data_root: Path) -> str:
    relative = path.relative_to(data_root)
    parts = list(relative.parts[:-1]) + [path.stem]
    cleaned = []
    for part in parts:
        cleaned.append(part.replace("_", "-"))
    return "__".join(cleaned)


def infer_pressure(rows: list[dict[str, int]], page_size: int, capacity_blocks: int, candidates: list[int]) -> list[dict[str, float]]:
    prompt_blocks = [ceil_div(row["prompt_len"], page_size) for row in rows]
    full_blocks = [ceil_div(row["prompt_len"] + row["output_len"], page_size) for row in rows]
    sorted_prompt = sorted(prompt_blocks, reverse=True)
    sorted_full = sorted(full_blocks, reverse=True)

    pressure = []
    for concurrency in candidates:
        limited = min(concurrency, len(rows))
        worst_prompt_blocks = sum(sorted_prompt[:limited])
        worst_full_blocks = sum(sorted_full[:limited])
        pressure.append(
            {
                "concurrency": float(concurrency),
                "prompt_only_blocks": float(worst_prompt_blocks),
                "prompt_only_pressure": worst_prompt_blocks / capacity_blocks,
                "full_request_blocks": float(worst_full_blocks),
                "full_request_pressure": worst_full_blocks / capacity_blocks,
            }
        )
    return pressure


def choose_concurrency_points(
    pressure_rows: list[dict[str, float]],
    *,
    target_pressures: list[float],
    num_points: int,
    single_concurrency_strategy: str,
) -> list[int]:
    if not pressure_rows:
        return []
    if num_points == 1:
        if single_concurrency_strategy == "closest-to-one":
            best = min(
                pressure_rows,
                key=lambda item: (
                    abs(item["full_request_pressure"] - 1.0),
                    -item["full_request_pressure"],
                ),
            )
        else:
            best = max(
                pressure_rows,
                key=lambda item: (item["full_request_pressure"], item["concurrency"]),
            )
        return [int(best["concurrency"])]

    by_concurrency = {
        int(item["concurrency"]): item for item in sorted(pressure_rows, key=lambda item: item["concurrency"])
    }
    selected: list[int] = []
    for target in target_pressures:
        best = min(
            by_concurrency.values(),
            key=lambda item: (
                abs(item["full_request_pressure"] - target),
                abs(item["concurrency"] - target * 100),
            ),
        )
        concurrency = int(best["concurrency"])
        if concurrency not in selected:
            selected.append(concurrency)
        if len(selected) == num_points:
            return sorted(selected)

    ordered = list(by_concurrency.keys())
    if len(selected) < num_points:
        quantile_indices = [
            round(i * (len(ordered) - 1) / max(1, num_points - 1)) for i in range(num_points)
        ]
        for idx in quantile_indices:
            concurrency = ordered[idx]
            if concurrency not in selected:
                selected.append(concurrency)
            if len(selected) == num_points:
                break

    if len(selected) < num_points:
        for concurrency in reversed(ordered):
            if concurrency not in selected:
                selected.append(concurrency)
            if len(selected) == num_points:
                break

    return sorted(selected)


def discover_workloads(args: argparse.Namespace, root: Path) -> list[WorkloadStats]:
    data_root = (root / args.data_root).resolve()
    jsonl_paths = sorted(data_root.rglob("*.jsonl"))
    if args.workload_filter:
        filters = [item.lower() for item in args.workload_filter]
        jsonl_paths = [
            path for path in jsonl_paths if any(item in str(path).lower() for item in filters)
        ]
    if not jsonl_paths:
        raise ValueError(f"No workloads found under {data_root}")

    workloads: list[WorkloadStats] = []
    for dataset_path in jsonl_paths:
        rows = load_lengths(dataset_path)
        max_requests = len(rows) if args.num_prompts_override is None else min(len(rows), args.num_prompts_override)
        base_candidates = sorted(set(args.candidate_concurrencies + [max_requests]))
        candidates = [value for value in base_candidates if value <= max_requests]
        if not candidates:
            candidates = [max_requests]

        pressure = infer_pressure(
            rows[:max_requests],
            page_size=args.page_size,
            capacity_blocks=args.gpu_kv_capacity_blocks,
            candidates=candidates,
        )
        selected = choose_concurrency_points(
            pressure,
            target_pressures=args.target_pressures,
            num_points=min(args.num_concurrency_points, len(candidates)),
            single_concurrency_strategy=args.single_concurrency_strategy,
        )
        workloads.append(
            WorkloadStats(
                dataset_path=dataset_path,
                workload_name=workload_slug(dataset_path, data_root),
                num_requests=max_requests,
                mean_prompt_len=sum(row["prompt_len"] for row in rows[:max_requests]) / max_requests,
                mean_output_len=sum(row["output_len"] for row in rows[:max_requests]) / max_requests,
                pressure_by_concurrency=pressure,
                selected_concurrencies=selected,
            )
        )
    return workloads


def estimate_run_seconds(
    workload: WorkloadStats,
    *,
    output_throughput_toks_per_s: float,
    startup_seconds: float,
    analysis_seconds: float,
) -> float:
    total_generated_tokens = workload.num_requests * workload.mean_output_len * 2.0
    generation_seconds = total_generated_tokens / output_throughput_toks_per_s
    return generation_seconds + (2.0 * startup_seconds) + analysis_seconds


def build_run_plan(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    workloads = discover_workloads(args, root)
    run_plan: list[dict[str, Any]] = []
    for workload in workloads:
        pressure_lookup = {
            int(item["concurrency"]): item for item in workload.pressure_by_concurrency
        }
        estimated_seconds = estimate_run_seconds(
            workload,
            output_throughput_toks_per_s=args.assumed_output_throughput_toks_per_s,
            startup_seconds=args.assumed_server_startup_seconds,
            analysis_seconds=args.assumed_analysis_seconds,
        )
        for schedule_policy in args.schedule_policies:
            for max_concurrency in workload.selected_concurrencies:
                run_plan.append(
                    {
                        "workload_name": workload.workload_name,
                        "dataset_path": str(workload.dataset_path.relative_to(root)),
                        "schedule_policy": schedule_policy,
                        "request_rate": args.request_rate,
                        "max_concurrency": max_concurrency,
                        "num_prompts": workload.num_requests,
                        "pressure_estimate": pressure_lookup[max_concurrency],
                        "mean_prompt_len": workload.mean_prompt_len,
                        "mean_output_len": workload.mean_output_len,
                        "estimated_run_seconds": estimated_seconds,
                    }
                )
    return run_plan


def write_plan(args: argparse.Namespace, output_root: Path, run_plan: list[dict[str, Any]]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_runs = [
        item for idx, item in enumerate(run_plan) if idx % args.num_shards == args.shard_index
    ]
    payload = {
        "args": vars(args),
        "num_total_runs": len(run_plan),
        "num_selected_runs": len(selected_runs),
        "selected_workloads": sorted({item["workload_name"] for item in selected_runs}),
        "estimated_serial_seconds_all_runs": sum(item["estimated_run_seconds"] for item in run_plan),
        "estimated_serial_seconds_selected_runs": sum(
            item["estimated_run_seconds"] for item in selected_runs
        ),
        "runs": selected_runs,
    }
    plan_path = output_root / "experiment_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path


def run_matrix(args: argparse.Namespace, root: Path, output_root: Path, run_plan: list[dict[str, Any]]) -> None:
    selected_runs = [
        item for idx, item in enumerate(run_plan) if idx % args.num_shards == args.shard_index
    ]
    for item in selected_runs:
        run_root = (
            output_root
            / "workloads"
            / item["workload_name"]
            / f"sched-{item['schedule_policy']}"
            / f"mc-{item['max_concurrency']}"
        )
        comparison_report = run_root / "reports" / "comparison.json"
        lru_trace = run_root / "traces" / "lru.jsonl"
        belady_trace = run_root / "traces" / "belady.jsonl"
        lru_bench = run_root / "benchmarks" / "lru.jsonl"
        belady_bench = run_root / "benchmarks" / "belady.jsonl"
        belady_plan = run_root / "plans" / "belady_plan.json"
        run_complete = (
            comparison_report.exists()
            if args.postprocess_mode == "inline"
            else all(
                path.exists()
                for path in (lru_trace, belady_trace, lru_bench, belady_bench, belady_plan)
            )
        )
        if args.resume and run_complete:
            continue

        command = [
            sys.executable,
            str(root / "benchmarking" / "runners" / "run_two_pass_benchmark.py"),
            "--model-path",
            args.model_path,
            "--dataset-path",
            item["dataset_path"],
            "--output-root",
            str(run_root),
            "--page-size",
            str(args.page_size),
            "--num-prompts",
            str(item["num_prompts"]),
            "--request-rate",
            str(item["request_rate"]),
            "--max-concurrency",
            str(item["max_concurrency"]),
            "--schedule-policy",
            str(item["schedule_policy"]),
            "--bench-seed",
            str(args.bench_seed),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--gpu-kv-capacity-blocks",
            str(args.gpu_kv_capacity_blocks),
        ]
        if args.postprocess_mode == "skip":
            command.append("--skip-analysis")
        if args.server_extra_args:
            command.extend(["--server-extra-args", args.server_extra_args])
        subprocess.run(command, check=True, cwd=str(root))


def print_summary(plan_path: Path) -> None:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    selected_runs = payload["num_selected_runs"]
    total_hours = payload["estimated_serial_seconds_selected_runs"] / 3600.0
    print(f"Experiment plan: {plan_path}")
    print(f"Selected runs: {selected_runs}")
    print(f"Estimated serial runtime for selected runs: {total_hours:.1f} hours")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    run_plan = build_run_plan(args, root)
    plan_path = write_plan(args, output_root, run_plan)
    print_summary(plan_path)

    if args.dry_run:
        return

    run_matrix(args, root, output_root, run_plan)


if __name__ == "__main__":
    main()
