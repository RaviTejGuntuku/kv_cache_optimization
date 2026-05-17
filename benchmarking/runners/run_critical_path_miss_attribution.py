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
    scaled_block_capacity,
    run_subprocess,
    write_manifest,
)


WORKLOADS = [
    WorkloadRef(
        label="optimistic",
        path="data/synthetic/headroom_studies/critical_path_miss_attribution/critical_path_serial_resume.jsonl",
    ),
    WorkloadRef(
        label="near_real",
        path="data/processed/headroom_studies/critical_path_miss_attribution/critical_path_miss_attribution__realworld_sequence.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [32],
        "mem_fractions": [0.24],
        "num_prompts": 16,
        "max_concurrency": 8,
        "request_rate": "4",
        "estimated_runtime_hours": 0.08,
    },
    "full": {
        "page_sizes": [32],
        "mem_fractions": [0.20, 0.28, 0.36, 0.44, 0.52, 0.60],
        "num_prompts": 192,
        "max_concurrency": 48,
        "request_rate": "8",
        "estimated_runtime_hours": 1.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the critical-path miss attribution study in pilot or full mode."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=16000)
    parser.add_argument("--reference-mem-fraction-static", type=float, default=0.24)
    parser.add_argument("--bypass-cache-fraction", type=float, default=0.6)
    parser.add_argument("--mem-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.mode]
    mem_fractions = args.mem_fractions or profile["mem_fractions"]
    server_extra_args = args.server_extra_args
    if args.mode == "pilot" and not server_extra_args:
        server_extra_args = "--disable-cuda-graph --disable-piecewise-cuda-graph"
    output_root = Path(args.output_root)
    manifest = profile_payload(
        "critical_path_miss_attribution",
        args.mode,
        {
            "workloads": [workload.path for workload in WORKLOADS],
            **{**profile, "mem_fractions": mem_fractions},
            "bypass_cache_fraction": args.bypass_cache_fraction,
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)

    for workload in WORKLOADS:
        dataset_path = ROOT / workload.path
        num_prompts = min(profile["num_prompts"], count_jsonl_rows(dataset_path))
        for page_size in profile["page_sizes"]:
            for mem_fraction in mem_fractions:
                block_capacity = scaled_block_capacity(
                    base_capacity_blocks=args.gpu_kv_capacity_blocks,
                    mem_fraction_static=mem_fraction,
                    reference_mem_fraction_static=args.reference_mem_fraction_static,
                )
                for second_policy, cache_fraction in (
                    ("belady", None),
                    ("belady_bypass", args.bypass_cache_fraction),
                ):
                    run_name = (
                        f"{workload.label}__ps{page_size}__mem{mem_fraction:.2f}__{second_policy}"
                    ).replace(".", "")
                    run_root = output_root / run_name
                    call_two_pass(
                        model_path=args.model_path,
                        dataset_path=str(dataset_path),
                        output_root=run_root,
                        page_size=page_size,
                        num_prompts=num_prompts,
                        request_rate=profile["request_rate"],
                        max_concurrency=profile["max_concurrency"],
                        mem_fraction_static=mem_fraction,
                        gpu_kv_capacity_blocks=block_capacity,
                        schedule_policy=args.schedule_policy,
                        bench_seed=args.bench_seed,
                        second_policy=second_policy,
                        second_policy_cache_fraction=cache_fraction,
                        server_extra_args=server_extra_args,
                        skip_analysis=args.skip_analysis,
                        dry_run=args.dry_run,
                    )
                    if args.skip_analysis:
                        continue
                    for label, trace_path, bench_path, output_name in (
                        ("lru", run_root / "traces" / "lru.jsonl", run_root / "benchmarks" / "lru.jsonl", "lru_critical_path.json"),
                        (second_policy, run_root / "traces" / "belady.jsonl", run_root / "benchmarks" / "belady.jsonl", f"{second_policy}_critical_path.json"),
                    ):
                        command = [
                            sys.executable,
                            str(ROOT / "benchmarking" / "analysis_scripts" / "analyze_critical_path_misses.py"),
                            "--trace",
                            str(trace_path),
                            "--bench",
                            str(bench_path),
                            "--output",
                            str(run_root / "reports" / output_name),
                            "--policy-label",
                            label,
                        ]
                        run_subprocess(command, dry_run=args.dry_run, cwd=ROOT)


if __name__ == "__main__":
    main()
