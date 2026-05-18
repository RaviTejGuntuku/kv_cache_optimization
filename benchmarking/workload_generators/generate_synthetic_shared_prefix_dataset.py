#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic synthetic workload with long shared prefixes "
            "and many competing branches per prefix group."
        )
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults next to --output.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=64,
        help="Number of distinct shared-prefix groups.",
    )
    parser.add_argument(
        "--prompts-per-group",
        type=int,
        default=16,
        help="Number of branches per shared-prefix group.",
    )
    parser.add_argument(
        "--system-prompt-len",
        type=int,
        default=2048,
        help="Approximate token count of the shared prefix per group.",
    )
    parser.add_argument(
        "--question-len",
        type=int,
        default=256,
        help="Approximate token count of the branch-specific suffix.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Target completion length recorded in the dataset.",
    )
    parser.add_argument(
        "--group-interleave",
        action="store_true",
        help="Interleave requests round-robin across groups. Enabled by default.",
        default=True,
    )
    return parser.parse_args()


COMMON_VOCAB = (
    "the",
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
)


def render_tokens(count: int, *, offset: int = 0) -> str:
    words = [COMMON_VOCAB[(offset + i) % len(COMMON_VOCAB)] for i in range(count)]
    return " ".join(words)


def make_shared_prefix(group_idx: int, system_prompt_len: int) -> str:
    header = (
        f"System directive for workload group {group_idx}. "
        f"All requests in this group share the same long prefix. "
        f"Retain the context exactly and answer only the branch-specific question.\n\n"
    )
    body = render_tokens(system_prompt_len, offset=group_idx)
    return header + body


def make_branch_suffix(group_idx: int, branch_idx: int, question_len: int) -> str:
    branch_header = (
        f"\n\nBranch request {branch_idx} for group {group_idx}. "
        f"This branch diverges only after the shared prefix. "
        f"Summarize the implications of the following branch-local evidence.\n\n"
    )
    branch_body = render_tokens(question_len, offset=group_idx + branch_idx + 1)
    return branch_header + branch_body


def build_rows(args: argparse.Namespace) -> list[dict]:
    groups: list[list[dict]] = []
    for group_idx in range(args.num_groups):
        shared_prefix = make_shared_prefix(group_idx, args.system_prompt_len)
        group_rows: list[dict] = []
        for branch_idx in range(args.prompts_per_group):
            prompt = shared_prefix + make_branch_suffix(
                group_idx=group_idx,
                branch_idx=branch_idx,
                question_len=args.question_len,
            )
            prompt_len = args.system_prompt_len + args.question_len
            group_rows.append(
                {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": f"Synthetic reference answer for group {group_idx}, branch {branch_idx}.",
                        },
                    ],
                    "prompt_len": prompt_len,
                    "output_len": args.output_len,
                    "metadata": {
                        "group_id": f"group_{group_idx:03d}",
                        "branch_id": branch_idx,
                        "shared_prefix_tokens": args.system_prompt_len,
                        "branch_suffix_tokens": args.question_len,
                    },
                }
            )
        groups.append(group_rows)

    rows: list[dict] = []
    if args.group_interleave:
        for branch_idx in range(args.prompts_per_group):
            for group_rows in groups:
                rows.append(group_rows[branch_idx])
    else:
        for group_rows in groups:
            rows.extend(group_rows)
    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "num_groups": args.num_groups,
        "prompts_per_group": args.prompts_per_group,
        "selected_requests": len(rows),
        "group_interleave": args.group_interleave,
        "system_prompt_len": args.system_prompt_len,
        "question_len": args.question_len,
        "output_len": args.output_len,
        "prompt_len_per_request": args.system_prompt_len + args.question_len,
    }
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
