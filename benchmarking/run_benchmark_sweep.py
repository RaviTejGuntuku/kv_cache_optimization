#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep of two-pass benchmark settings.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-rates", nargs="+", required=True)
    parser.add_argument("--max-concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--server-extra-args", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for request_rate in args.request_rates:
        for max_concurrency in args.max_concurrencies:
            run_name = f"rr_{request_rate}_mc_{max_concurrency}".replace("/", "_")
            run_output = output_root / run_name
            cmd = [
                sys.executable,
                str(root / "benchmarking" / "run_two_pass_benchmark.py"),
                "--model-path",
                args.model_path,
                "--dataset-path",
                args.dataset_path,
                "--output-root",
                str(run_output),
                "--page-size",
                str(args.page_size),
                "--num-prompts",
                str(args.num_prompts),
                "--request-rate",
                str(request_rate),
                "--max-concurrency",
                str(max_concurrency),
                "--bench-seed",
                str(args.bench_seed),
                "--gpu-kv-capacity-blocks",
                str(args.gpu_kv_capacity_blocks),
            ]
            if args.server_extra_args:
                cmd.extend(["--server-extra-args", args.server_extra_args])
            subprocess.run(cmd, check=True, cwd=str(root))
            manifest.append(
                {
                    "run_name": run_name,
                    "request_rate": request_rate,
                    "max_concurrency": max_concurrency,
                    "output_root": str(run_output),
                    "comparison_report": str(run_output / "reports" / "comparison.json"),
                }
            )

    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
