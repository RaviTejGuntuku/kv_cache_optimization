#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def normalize_component(value: object) -> str:
    return str(value).replace("/", "-").replace("_", "-").replace(" ", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep of two-pass benchmark settings.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-rates", nargs="+", required=True)
    parser.add_argument("--max-concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument(
        "--schedule-policy",
        default="fcfs",
        help="SGLang schedule policy, e.g. fcfs or lpm.",
    )
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--mem-fraction-static", default="0.7")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument(
        "--sweep-tag",
        default=None,
        help="Optional suffix appended to the sweep output-root directory name.",
    )
    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="Append a UTC timestamp suffix to output-root so repeated sweeps are preserved.",
    )
    return parser.parse_args()


def resolve_output_root(output_root: Path, *, sweep_tag: str | None, auto_version: bool) -> Path:
    suffix = sweep_tag
    if auto_version and not suffix:
        suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if suffix:
        resolved = output_root.parent / f"{output_root.name}__{suffix}"
    else:
        resolved = output_root

    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Sweep output directory {resolved} already exists and is not empty. "
            "Pass --sweep-tag or --auto-version to preserve previous sweeps."
        )
    return resolved


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_root = resolve_output_root(
        Path(args.output_root), sweep_tag=args.sweep_tag, auto_version=args.auto_version
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    (output_root / "sweep_metadata.json").write_text(
        json.dumps(
            {
                "invoked_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "resolved_output_root": str(output_root),
                "args": vars(args),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for request_rate in args.request_rates:
        for max_concurrency in args.max_concurrencies:
            run_name = (
                f"policy-{normalize_component(args.schedule_policy)}"
                f"__rr-{normalize_component(request_rate)}"
                f"__mc-{normalize_component(max_concurrency)}"
            )
            run_output = output_root / run_name
            cmd = [
                sys.executable,
                str(root / "benchmarking" / "runners" / "run_two_pass_benchmark.py"),
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
                "--schedule-policy",
                str(args.schedule_policy),
                "--request-rate",
                str(request_rate),
                "--max-concurrency",
                str(max_concurrency),
                "--bench-seed",
                str(args.bench_seed),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--gpu-kv-capacity-blocks",
                str(args.gpu_kv_capacity_blocks),
            ]
            if args.server_extra_args:
                cmd.extend(["--server-extra-args", args.server_extra_args])
            subprocess.run(cmd, check=True, cwd=str(root))
            manifest.append(
                {
                    "run_name": run_name,
                    "schedule_policy": args.schedule_policy,
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
