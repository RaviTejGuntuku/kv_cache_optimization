#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from generate_recency_trap_workloads import FAMILY_BUILDERS


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    family: str
    description: str
    num_families: int
    hot_set_size: int
    branches_per_family: int
    rounds: int
    interference_per_round: int
    prefix_tokens: int
    suffix_tokens: int
    output_len: int


RECOMMENDED_SPECS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        name="natural_bursty_return_hbm",
        family="bursty-return",
        description=(
            "A hot set of reusable families is active, displaced by substantial other work, "
            "and then becomes hot again. This models burst-return traffic with large shared templates."
        ),
        num_families=20,
        hot_set_size=4,
        branches_per_family=16,
        rounds=1,
        interference_per_round=0,
        prefix_tokens=3072,
        suffix_tokens=256,
        output_len=256,
    ),
    WorkloadSpec(
        name="natural_zipf_bursty_hbm",
        family="zipf-bursty",
        description=(
            "Skewed popularity with wave-like arrivals. This approximates heavy-head, long-tail "
            "traffic where popular reusable prefixes arrive in bursts."
        ),
        num_families=12,
        hot_set_size=3,
        branches_per_family=12,
        rounds=6,
        interference_per_round=0,
        prefix_tokens=3584,
        suffix_tokens=256,
        output_len=256,
    ),
    WorkloadSpec(
        name="natural_hotset_one_shot_hbm",
        family="hotset-one-shot",
        description=(
            "A reusable hot set competes with a steady stream of largely one-shot work. "
            "This models recurring templates mixed with singleton traffic."
        ),
        num_families=24,
        hot_set_size=6,
        branches_per_family=16,
        rounds=16,
        interference_per_round=20,
        prefix_tokens=3072,
        suffix_tokens=256,
        output_len=256,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate heavier natural-pattern synthetic workloads that saturate HBM at realistic "
            "concurrency levels."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic/natural_saturated",
        help="Directory where JSONL workloads and manifests are written.",
    )
    parser.add_argument(
        "--gpu-kv-capacity-blocks",
        type=int,
        default=16000,
        help="Used only for manifest pressure estimates.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Used only for manifest pressure estimates.",
    )
    parser.add_argument(
        "--candidate-concurrencies",
        type=int,
        nargs="+",
        default=[32, 48, 64, 72, 80, 96, 128, 160],
        help="Candidate concurrency points reported in the manifest.",
    )
    return parser.parse_args()


def block_count(spec: WorkloadSpec, page_size: int) -> int:
    tokens = spec.prefix_tokens + spec.suffix_tokens + spec.output_len
    return math.ceil(tokens / page_size)


def estimated_pressure(spec: WorkloadSpec, concurrency: int, page_size: int, capacity_blocks: int) -> float:
    return block_count(spec, page_size) * concurrency / capacity_blocks


def recommend_concurrency(spec: WorkloadSpec, candidates: list[int], page_size: int, capacity_blocks: int) -> int:
    return min(
        candidates,
        key=lambda concurrency: (
            abs(estimated_pressure(spec, concurrency, page_size, capacity_blocks) - 1.0),
            -estimated_pressure(spec, concurrency, page_size, capacity_blocks),
        ),
    )


def to_builder_args(spec: WorkloadSpec):
    class Args:
        pass

    args = Args()
    args.family = spec.family
    args.num_families = spec.num_families
    args.hot_set_size = spec.hot_set_size
    args.branches_per_family = spec.branches_per_family
    args.rounds = spec.rounds
    args.interference_per_round = spec.interference_per_round
    args.prefix_tokens = spec.prefix_tokens
    args.suffix_tokens = spec.suffix_tokens
    args.output_len = spec.output_len
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for spec in RECOMMENDED_SPECS:
        rows = FAMILY_BUILDERS[spec.family](to_builder_args(spec))

        jsonl_path = output_dir / f"{spec.name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        recommended_mc = recommend_concurrency(
            spec,
            args.candidate_concurrencies,
            args.page_size,
            args.gpu_kv_capacity_blocks,
        )
        pressure_table = [
            {
                "concurrency": concurrency,
                "estimated_full_request_pressure": estimated_pressure(
                    spec, concurrency, args.page_size, args.gpu_kv_capacity_blocks
                ),
            }
            for concurrency in args.candidate_concurrencies
        ]

        manifest = {
            "name": spec.name,
            "description": spec.description,
            "family": spec.family,
            "selected_requests": len(rows),
            "page_size": args.page_size,
            "gpu_kv_capacity_blocks": args.gpu_kv_capacity_blocks,
            "per_request_full_blocks_estimate": block_count(spec, args.page_size),
            "recommended_concurrency": recommended_mc,
            "recommended_concurrency_pressure": estimated_pressure(
                spec, recommended_mc, args.page_size, args.gpu_kv_capacity_blocks
            ),
            "pressure_by_concurrency": pressure_table,
            "spec": asdict(spec),
        }
        manifest_path = output_dir / f"{spec.name}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        index.append(
            {
                "name": spec.name,
                "path": str(jsonl_path),
                "manifest": str(manifest_path),
                "recommended_concurrency": recommended_mc,
                "recommended_concurrency_pressure": manifest["recommended_concurrency_pressure"],
            }
        )

    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "workloads": index,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
