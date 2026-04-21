#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FCFS delayed-reuse workload matrix over multiple page sizes."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workloads", nargs="+", required=True)
    parser.add_argument("--page-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=128)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", default="0.24")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=16000)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--auto-version", action="store_true")
    return parser.parse_args()


def normalize(path: str) -> str:
    return (
        Path(path)
        .stem.replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    if args.auto_version:
        suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_root = output_root.parent / f"{output_root.name}__{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "invoked_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        "runs": [],
    }

    for workload in args.workloads:
        for page_size in args.page_sizes:
            run_name = f"{normalize(workload)}__ps-{page_size}"
            run_output = output_root / run_name
            dataset_path = Path(workload)
            num_prompts = args.num_prompts if args.num_prompts is not None else count_jsonl_rows(dataset_path)
            cmd = [
                sys.executable,
                str(root / "benchmarking" / "runners" / "run_two_pass_benchmark.py"),
                "--model-path",
                args.model_path,
                "--dataset-path",
                workload,
                "--output-root",
                str(run_output),
                "--page-size",
                str(page_size),
                "--request-rate",
                str(args.request_rate),
                "--max-concurrency",
                str(args.max_concurrency),
                "--schedule-policy",
                str(args.schedule_policy),
                "--bench-seed",
                str(args.bench_seed),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--gpu-kv-capacity-blocks",
                str(args.gpu_kv_capacity_blocks),
            ]
            cmd.extend(["--num-prompts", str(num_prompts)])
            if args.server_extra_args:
                cmd.extend(["--server-extra-args", args.server_extra_args])
            if args.skip_analysis:
                cmd.append("--skip-analysis")
            subprocess.run(cmd, check=True, cwd=str(root))
            manifest["runs"].append(
                {
                    "workload": workload,
                    "num_prompts": num_prompts,
                    "page_size": page_size,
                    "output_root": str(run_output),
                }
            )

    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
