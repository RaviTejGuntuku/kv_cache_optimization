#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from headroom_common import (
    ROOT,
    WorkloadRef,
    call_two_pass,
    count_jsonl_rows,
    profile_payload,
    run_subprocess,
    write_manifest,
)


WORKLOADS = [
    WorkloadRef(
        label="optimistic",
        path="data/synthetic/headroom_studies/opt_with_incoming_line/incoming_suffix_pollution.jsonl",
    ),
    WorkloadRef(
        label="near_real",
        path="data/processed/headroom_studies/opt_with_incoming_line/opt_with_incoming_line__realworld_sequence.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [32],
        "cache_fractions": [0.6],
        "num_prompts": 24,
        "max_concurrency": 16,
        "request_rate": "8",
        "mem_fraction_static": 0.24,
        "estimated_runtime_hours": 0.08,
    },
    "full": {
        "page_sizes": [32],
        "cache_fractions": [0.4, 0.5, 0.6, 0.7, 0.8],
        "num_prompts": 192,
        "max_concurrency": 96,
        "request_rate": "16",
        "mem_fraction_static": 0.24,
        "estimated_runtime_hours": 2.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the incoming-line-aware OPT+bypass partition sweep."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=16000)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    server_extra_args = args.server_extra_args
    if args.mode == "pilot" and not server_extra_args:
        server_extra_args = "--disable-cuda-graph --disable-piecewise-cuda-graph"
    output_root = Path(args.output_root)
    manifest = profile_payload(
        "opt_with_incoming_line",
        args.mode,
        {"workloads": [workload.path for workload in WORKLOADS], **profile},
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    for workload in WORKLOADS:
        dataset_path = ROOT / workload.path
        num_prompts = min(profile["num_prompts"], count_jsonl_rows(dataset_path))
        for page_size in profile["page_sizes"]:
            bypass_roots: list[Path] = []
            for fraction in profile["cache_fractions"]:
                run_root = output_root / (
                    f"{workload.label}__ps{page_size}__n{fraction:.1f}".replace(".", "")
                )
                bypass_roots.append(run_root)
                call_two_pass(
                    model_path=args.model_path,
                    dataset_path=str(dataset_path),
                    output_root=run_root,
                    page_size=page_size,
                    num_prompts=num_prompts,
                    request_rate=profile["request_rate"],
                    max_concurrency=profile["max_concurrency"],
                    mem_fraction_static=profile["mem_fraction_static"],
                    gpu_kv_capacity_blocks=args.gpu_kv_capacity_blocks,
                    schedule_policy=args.schedule_policy,
                    bench_seed=args.bench_seed,
                    second_policy="belady_bypass",
                    second_policy_cache_fraction=fraction,
                    server_extra_args=server_extra_args,
                    skip_analysis=args.skip_analysis,
                    dry_run=args.dry_run,
                )
            if not args.skip_analysis:
                command = [
                    sys.executable,
                    str(ROOT / "benchmarking" / "analysis_scripts" / "select_best_opt_bypass_fraction.py"),
                    "--output",
                    str(output_root / f"{workload.label}__best_fraction.json"),
                    "--metric",
                    "output_throughput",
                    "--run-roots",
                    *[str(path) for path in bypass_roots],
                ]
                run_subprocess(command, dry_run=args.dry_run, cwd=ROOT)

            baseline_root = output_root / f"{workload.label}__ps{page_size}__opt_baseline"
            call_two_pass(
                model_path=args.model_path,
                dataset_path=str(dataset_path),
                output_root=baseline_root,
                page_size=page_size,
                num_prompts=num_prompts,
                request_rate=profile["request_rate"],
                max_concurrency=profile["max_concurrency"],
                mem_fraction_static=profile["mem_fraction_static"],
                gpu_kv_capacity_blocks=args.gpu_kv_capacity_blocks,
                schedule_policy=args.schedule_policy,
                bench_seed=args.bench_seed,
                second_policy="belady",
                second_policy_cache_fraction=None,
                server_extra_args=server_extra_args,
                skip_analysis=args.skip_analysis,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
