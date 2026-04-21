#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a ShareGPT-derived workload that maximizes long shared-prefix "
            "competition across active requests."
        )
    )
    parser.add_argument("--input", required=True, help="Path to raw ShareGPT JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL benchmark path.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path. Defaults next to --output.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional Hugging Face tokenizer id/path for token-length estimates.",
    )
    parser.add_argument(
        "--group-prefix-words",
        type=int,
        default=32,
        help="Number of normalized leading words used for grouping.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=8,
        help="Discard groups smaller than this.",
    )
    parser.add_argument(
        "--target-groups",
        type=int,
        default=8,
        help="Maximum number of prefix groups to keep.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=32,
        help="Maximum number of requests emitted from each selected group.",
    )
    parser.add_argument(
        "--min-prompt-len",
        type=int,
        default=256,
        help="Minimum prompt length for a request to be eligible.",
    )
    parser.add_argument(
        "--min-shared-prefix-chars",
        type=int,
        default=192,
        help="Minimum normalized shared-prefix length across a group.",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="Optional hard cap on prompt tokens.",
    )
    parser.add_argument(
        "--max-total-len",
        type=int,
        default=None,
        help="Optional hard cap on prompt + output tokens.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def first_words(text: str, n: int) -> str:
    words = normalize_text(text).split()
    return " ".join(words[:n])


def shared_prefix_len(strings: list[str]) -> int:
    if not strings:
        return 0
    prefix = strings[0]
    for value in strings[1:]:
        limit = min(len(prefix), len(value))
        i = 0
        while i < limit and prefix[i] == value[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return 0
    return len(prefix)


def maybe_load_tokenizer(tokenizer_name: Optional[str]):
    if tokenizer_name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def estimate_len(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, math.ceil(len(text.split()) * 1.3))


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_conversation_pair(record: dict[str, Any]) -> Optional[tuple[str, str]]:
    conversation = record.get("conversations", record.get("conversation", []))
    if not isinstance(conversation, list) or len(conversation) < 2:
        return None
    first, second = conversation[0], conversation[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    prompt = first.get("value") or first.get("content")
    answer = second.get("value") or second.get("content")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer:
        return None
    return prompt, answer


@dataclass
class Candidate:
    prompt: str
    answer: str
    prompt_len: int
    answer_len: int
    group_key: str
    normalized_prompt: str


def build_candidates(records: list[dict[str, Any]], tokenizer, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in records:
        pair = extract_conversation_pair(record)
        if pair is None:
            continue
        prompt, answer = pair
        prompt_len = estimate_len(prompt, tokenizer)
        answer_len = estimate_len(answer, tokenizer)
        if prompt_len < args.min_prompt_len:
            continue
        if args.max_prompt_len is not None and prompt_len > args.max_prompt_len:
            continue
        if args.max_total_len is not None and (prompt_len + answer_len) > args.max_total_len:
            continue
        normalized_prompt = normalize_text(prompt)
        group_key = first_words(prompt, args.group_prefix_words)
        if not group_key:
            continue
        candidates.append(
            Candidate(
                prompt=prompt,
                answer=answer,
                prompt_len=prompt_len,
                answer_len=answer_len,
                group_key=group_key,
                normalized_prompt=normalized_prompt,
            )
        )
    return candidates


def score_group(group: list[Candidate], lcp_chars: int) -> tuple[float, int, int]:
    avg_prompt_len = sum(item.prompt_len for item in group) / len(group)
    return (lcp_chars * len(group) * avg_prompt_len, lcp_chars, len(group))


def main() -> None:
    args = parse_args()
    tokenizer = maybe_load_tokenizer(args.tokenizer)
    records = load_records(Path(args.input))
    candidates = build_candidates(records, tokenizer, args)

    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.group_key, []).append(candidate)

    ranked_groups: list[tuple[str, list[Candidate], int]] = []
    for group_key, group in groups.items():
        if len(group) < args.min_group_size:
            continue
        lcp_chars = shared_prefix_len([item.normalized_prompt for item in group])
        if lcp_chars < args.min_shared_prefix_chars:
            continue
        ranked_groups.append((group_key, group, lcp_chars))

    ranked_groups.sort(
        key=lambda item: score_group(item[1], item[2]),
        reverse=True,
    )
    chosen_groups = ranked_groups[: args.target_groups]

    selected_groups: list[tuple[str, list[Candidate], int]] = []
    for group_idx, (group_key, group, lcp_chars) in enumerate(chosen_groups):
        sorted_group = sorted(
            group,
            key=lambda item: (item.prompt_len, item.answer_len, item.prompt),
            reverse=True,
        )[: args.samples_per_group]
        selected_groups.append((f"group_{group_idx:02d}", sorted_group, lcp_chars))

    interleaved_rows: list[dict[str, Any]] = []
    max_group_depth = max((len(group) for _, group, _ in selected_groups), default=0)
    for round_idx in range(max_group_depth):
        for group_id, group, lcp_chars in selected_groups:
            if round_idx >= len(group):
                continue
            candidate = group[round_idx]
            interleaved_rows.append(
                {
                    "conversations": [
                        {"role": "user", "content": candidate.prompt},
                        {"role": "assistant", "content": candidate.answer},
                    ],
                    "prompt_len": candidate.prompt_len,
                    "output_len": candidate.answer_len,
                    "metadata": {
                        "group_id": group_id,
                        "round_idx": round_idx,
                        "shared_prefix_chars": lcp_chars,
                        "group_key": candidate.group_key,
                    },
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in interleaved_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    manifest = {
        "input_candidates": len(candidates),
        "selected_requests": len(interleaved_rows),
        "selected_groups": len(selected_groups),
        "group_prefix_words": args.group_prefix_words,
        "selection_args": vars(args),
        "groups": [
            {
                "group_id": group_id,
                "group_key": group[0].group_key if group else "",
                "selected_count": len(group),
                "shared_prefix_chars": lcp_chars,
                "avg_prompt_len": (sum(item.prompt_len for item in group) / len(group)) if group else 0.0,
                "max_prompt_len": max((item.prompt_len for item in group), default=0),
                "avg_output_len": (sum(item.answer_len for item in group) / len(group)) if group else 0.0,
            }
            for group_id, group, lcp_chars in selected_groups
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
