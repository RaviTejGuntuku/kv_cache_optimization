#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


COMMON_VOCAB = (
    "analysis",
    "context",
    "memory",
    "request",
    "shared",
    "prefix",
    "cache",
    "branch",
    "token",
    "sequence",
    "inference",
    "trace",
    "reuse",
    "scheduler",
    "radix",
    "oracle",
)


def render_tokens(count: int, *, offset: int = 0) -> str:
    words = [COMMON_VOCAB[(offset + i) % len(COMMON_VOCAB)] for i in range(count)]
    return " ".join(words)


def make_shared_prefix(family: str, prefix_tokens: int, offset: int) -> str:
    header = (
        f"System directive for family {family}. "
        f"Requests in this family share a long reusable prefix. "
        f"Reuse this family context whenever possible and answer only the branch-local question.\n\n"
    )
    return header + render_tokens(prefix_tokens, offset=offset)


def make_unique_prompt(tag: str, suffix_tokens: int, offset: int) -> str:
    return (
        f"User request for one-shot workload item {tag}. "
        f"This prompt is intentionally unique and should not share cache state with others.\n\n"
        + render_tokens(suffix_tokens, offset=offset)
    )


def make_family_prompt(
    family: str,
    branch: int,
    *,
    prefix_tokens: int,
    suffix_tokens: int,
    offset: int,
    phase_label: str,
) -> str:
    return (
        make_shared_prefix(family, prefix_tokens, offset)
        + "\n\n"
        + f"Phase {phase_label}. Branch {branch} from family {family}. "
        + "Explain the impact of this branch-local evidence.\n\n"
        + render_tokens(suffix_tokens, offset=offset + branch + 1)
    )


def make_row(
    prompt: str,
    *,
    family: str,
    branch: int,
    sequence_id: int,
    phase: str,
    kind: str,
    prefix_tokens: int,
    suffix_tokens: int,
    output_len: int,
) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": f"Synthetic answer for {family} branch {branch} in phase {phase}.",
            },
        ],
        "prompt_len": prefix_tokens + suffix_tokens if kind != "unique" else suffix_tokens,
        "output_len": output_len,
        "metadata": {
            "family": family,
            "branch": branch,
            "sequence_id": sequence_id,
            "phase": phase,
            "kind": kind,
            "shared_prefix_tokens": prefix_tokens if kind != "unique" else 0,
            "branch_suffix_tokens": suffix_tokens,
        },
    }


def build_bursty_return(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    hot_families = [f"hot_{i:02d}" for i in range(args.hot_set_size)]
    cold_families = [f"cold_{i:02d}" for i in range(args.num_families - args.hot_set_size)]

    for family_idx, family in enumerate(hot_families):
        for branch in range(args.branches_per_family // 2):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        branch,
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        offset=family_idx * 17,
                        phase_label="warm",
                    ),
                    family=family,
                    branch=branch,
                    sequence_id=seq,
                    phase="warm",
                    kind="shared",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1

    for family_idx, family in enumerate(cold_families):
        for branch in range(args.branches_per_family):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        branch,
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        offset=500 + family_idx * 19,
                        phase_label="interference",
                    ),
                    family=family,
                    branch=branch,
                    sequence_id=seq,
                    phase="interference",
                    kind="shared",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1

    for family_idx, family in enumerate(hot_families):
        for branch in range(args.branches_per_family // 2, args.branches_per_family):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        branch,
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        offset=family_idx * 17,
                        phase_label="return",
                    ),
                    family=family,
                    branch=branch,
                    sequence_id=seq,
                    phase="return",
                    kind="shared",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1
    return rows


def build_hotset_with_one_shot_interference(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    hot_families = [f"hot_{i:02d}" for i in range(args.hot_set_size)]

    for round_id in range(args.rounds):
        for family_idx, family in enumerate(hot_families):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        round_id,
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        offset=family_idx * 23,
                        phase_label=f"round-{round_id}",
                    ),
                    family=family,
                    branch=round_id,
                    sequence_id=seq,
                    phase=f"round_{round_id}",
                    kind="shared",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1

        for one_shot in range(args.interference_per_round):
            tag = f"unique_r{round_id}_{one_shot:03d}"
            rows.append(
                make_row(
                    make_unique_prompt(tag, args.suffix_tokens + args.prefix_tokens, 1000 + seq * 3),
                    family=tag,
                    branch=0,
                    sequence_id=seq,
                    phase=f"interference_{round_id}",
                    kind="unique",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens + args.prefix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1
    return rows


def build_zipf_bursty(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    family_names = [f"zipf_{i:02d}" for i in range(args.num_families)]
    family_counts = [
        max(2, args.branches_per_family * max(1, args.num_families // (idx + 1)))
        for idx in range(args.num_families)
    ]

    for wave in range(args.rounds):
        ordered = sorted(
            range(args.num_families),
            key=lambda idx: ((idx + wave) % args.num_families, idx),
        )
        for idx in ordered:
            family = family_names[idx]
            burst = max(2, family_counts[idx] // args.num_families)
            for local_branch in range(burst):
                branch = wave * burst + local_branch
                rows.append(
                    make_row(
                        make_family_prompt(
                            family,
                            branch,
                            prefix_tokens=args.prefix_tokens,
                            suffix_tokens=args.suffix_tokens,
                            offset=idx * 29,
                            phase_label=f"wave-{wave}",
                        ),
                        family=family,
                        branch=branch,
                        sequence_id=seq,
                        phase=f"wave_{wave}",
                        kind="shared",
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        output_len=args.output_len,
                    )
                )
                seq += 1
    return rows


def build_adversarial_recency_trap(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    anchor_family = "anchor"

    for branch in range(args.branches_per_family):
        rows.append(
            make_row(
                make_family_prompt(
                    anchor_family,
                    branch,
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    offset=11,
                    phase_label="anchor-warm",
                ),
                family=anchor_family,
                branch=branch,
                sequence_id=seq,
                phase="anchor_warm",
                kind="shared",
                prefix_tokens=args.prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                output_len=args.output_len,
            )
        )
        seq += 1

    for trap_id in range(args.num_families * args.interference_per_round):
        family = f"trap_{trap_id:03d}"
        rows.append(
            make_row(
                make_family_prompt(
                    family,
                    0,
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    offset=300 + trap_id * 7,
                    phase_label="trap",
                ),
                family=family,
                branch=0,
                sequence_id=seq,
                phase="trap",
                kind="shared",
                prefix_tokens=args.prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                output_len=args.output_len,
            )
        )
        seq += 1

    for branch in range(args.branches_per_family, args.branches_per_family * 2):
        rows.append(
            make_row(
                make_family_prompt(
                    anchor_family,
                    branch,
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    offset=11,
                    phase_label="anchor-return",
                ),
                family=anchor_family,
                branch=branch,
                sequence_id=seq,
                phase="anchor_return",
                kind="shared",
                prefix_tokens=args.prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                output_len=args.output_len,
            )
        )
        seq += 1
    return rows


def build_grouped_baseline(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seq = 0
    for family_idx in range(args.num_families):
        family = f"grouped_{family_idx:02d}"
        for branch in range(args.branches_per_family):
            rows.append(
                make_row(
                    make_family_prompt(
                        family,
                        branch,
                        prefix_tokens=args.prefix_tokens,
                        suffix_tokens=args.suffix_tokens,
                        offset=family_idx * 13,
                        phase_label="grouped",
                    ),
                    family=family,
                    branch=branch,
                    sequence_id=seq,
                    phase="grouped",
                    kind="shared",
                    prefix_tokens=args.prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    output_len=args.output_len,
                )
            )
            seq += 1
    return rows


FAMILY_BUILDERS: dict[str, Callable[[argparse.Namespace], list[dict]]] = {
    "grouped-baseline": build_grouped_baseline,
    "bursty-return": build_bursty_return,
    "hotset-one-shot": build_hotset_with_one_shot_interference,
    "zipf-bursty": build_zipf_bursty,
    "adversarial-recency-trap": build_adversarial_recency_trap,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic workloads designed to expose LRU-vs-OPT gaps."
    )
    parser.add_argument("--family", choices=sorted(FAMILY_BUILDERS), required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--manifest", default=None, help="Optional manifest output path.")
    parser.add_argument("--num-families", type=int, default=12)
    parser.add_argument("--hot-set-size", type=int, default=3)
    parser.add_argument("--branches-per-family", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--interference-per-round", type=int, default=12)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = FAMILY_BUILDERS[args.family](args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    preview = []
    for row in rows[: min(16, len(rows))]:
        meta = row["metadata"]
        preview.append(
            {
                "sequence_id": meta["sequence_id"],
                "family": meta["family"],
                "branch": meta["branch"],
                "phase": meta["phase"],
                "kind": meta["kind"],
            }
        )
    for row in rows:
        meta = row["metadata"]
        family_counts[meta["family"]] = family_counts.get(meta["family"], 0) + 1
        phase_counts[meta["phase"]] = phase_counts.get(meta["phase"], 0) + 1

    manifest = {
        "family": args.family,
        "selected_requests": len(rows),
        "num_families": args.num_families,
        "hot_set_size": args.hot_set_size,
        "branches_per_family": args.branches_per_family,
        "rounds": args.rounds,
        "interference_per_round": args.interference_per_round,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "output_len": args.output_len,
        "phase_counts": phase_counts,
        "family_counts": family_counts,
        "preview_sequence": preview,
    }
    manifest_path = (
        Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
