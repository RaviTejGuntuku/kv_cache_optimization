#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from headroom_common import (
    ROOT,
    WorkloadRef,
    call_two_pass,
    count_jsonl_rows,
    profile_payload,
    scaled_block_capacity,
    write_manifest,
)


WORKLOADS = [
    WorkloadRef(
        label="optimistic",
        path="datasets/synthetic/headroom_studies/effective_residency_sweep/residency_hotset_capacity_ladder.jsonl",
    ),
    WorkloadRef(
        label="near_real",
        path="datasets/processed/headroom_studies/effective_residency_sweep/effective_residency_sweep__realworld_sequence.jsonl",
    ),
]

PROFILES = {
    "pilot": {
        "page_sizes": [32],
        "mem_fractions": [0.24],
        "num_prompts": 16,
        "max_concurrency": 16,
        "request_rate": "8",
        "estimated_runtime_hours": 0.08,
    },
    "full": {
        "page_sizes": [32],
        "mem_fractions": [0.20, 0.24, 0.28, 0.32, 0.40, 0.48, 0.56, 0.64],
        "num_prompts": 256,
        "max_concurrency": 96,
        "request_rate": "16",
        "estimated_runtime_hours": 1.75,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the effective-residency headroom sweep in pilot or full mode."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--schedule-policy", default="fcfs")
    parser.add_argument("--bench-seed", type=int, default=1)
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=16000)
    parser.add_argument("--reference-mem-fraction-static", type=float, default=0.24)
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
        "effective_residency_sweep",
        args.mode,
        {
            "workloads": [workload.path for workload in WORKLOADS],
            **{**profile, "mem_fractions": mem_fractions},
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
                for second_policy in ("belady",):
                    run_name = (
                        f"{workload.label}__ps{page_size}__mem{mem_fraction:.2f}__{second_policy}"
                    ).replace(".", "")
                    call_two_pass(
                        model_path=args.model_path,
                        dataset_path=str(dataset_path),
                        output_root=output_root / run_name,
                        page_size=page_size,
                        num_prompts=num_prompts,
                        request_rate=profile["request_rate"],
                        max_concurrency=profile["max_concurrency"],
                        mem_fraction_static=mem_fraction,
                        gpu_kv_capacity_blocks=block_capacity,
                        schedule_policy=args.schedule_policy,
                        bench_seed=args.bench_seed,
                        second_policy=second_policy,
                        server_extra_args=server_extra_args,
                        skip_analysis=args.skip_analysis,
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    main()
