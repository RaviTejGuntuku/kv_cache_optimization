#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a benchmark subset for LMSYS, ShareGPT, or LongBench-style "
            "datasets while preserving the properties that matter for KV-cache evaluation."
        )
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["lmsys", "sharegpt", "longbench"],
        help="Subset policy to apply.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path containing the selected benchmark subset.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON path for subset statistics. Defaults next to --output.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Approximate number of selected requests. Defaults depend on dataset type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional Hugging Face tokenizer id/path for better length estimates.",
    )
    parser.add_argument(
        "--prefix-chars",
        type=int,
        default=160,
        help="Number of normalized leading characters used for prefix clustering.",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="Optional hard cap on prompt tokens. Candidates above this are dropped.",
    )
    parser.add_argument(
        "--max-total-len",
        type=int,
        default=None,
        help="Optional hard cap on prompt_len + answer_len. Candidates above this are dropped.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


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


def coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, list):
        for item in value:
            coerced = coerce_text(item)
            if coerced:
                return coerced
    if isinstance(value, dict):
        for key in ("content", "value", "text", "answer"):
            if key in value:
                coerced = coerce_text(value[key])
                if coerced:
                    return coerced
    return None


def extract_conversation_pair(record: dict[str, Any]) -> Optional[tuple[str, str]]:
    conversation = record.get("conversations", record.get("conversation", []))
    if not isinstance(conversation, list) or len(conversation) < 2:
        return None
    prompt = coerce_text(conversation[0])
    answer = coerce_text(conversation[1])
    if prompt and answer:
        return prompt, answer
    return None


@dataclass
class Candidate:
    raw: dict[str, Any]
    prompt: str
    answer: str
    prompt_len: int
    answer_len: int
    prefix_key: str
    bucket: str


def make_custom_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "user", "content": candidate.prompt},
            {"role": "assistant", "content": candidate.answer},
        ],
        "prompt_len": candidate.prompt_len,
        "output_len": candidate.answer_len,
    }


def bucket_share_length(prompt_len: int) -> str:
    if prompt_len < 512:
        return "short"
    if prompt_len < 2048:
        return "medium"
    if prompt_len < 8192:
        return "long"
    return "xlong"


def bucket_longbench(prompt_len: int) -> str:
    if prompt_len < 8000:
        return "0_8k"
    if prompt_len < 32000:
        return "8k_32k"
    if prompt_len < 64000:
        return "32k_64k"
    return "64k_plus"


def default_target_size(dataset_type: str) -> int:
    return {
        "lmsys": 1500,
        "sharegpt": 1500,
        "longbench": 120,
    }[dataset_type]


def build_lmsys_candidates(records: Iterable[dict[str, Any]], tokenizer, prefix_chars: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in records:
        pair = extract_conversation_pair(record)
        if pair is None:
            continue
        prompt, answer = pair
        prompt_len = estimate_len(prompt, tokenizer)
        answer_len = estimate_len(answer, tokenizer)
        prefix_key = normalize_text(prompt)[:prefix_chars]
        candidates.append(
            Candidate(
                raw=record,
                prompt=prompt,
                answer=answer,
                prompt_len=prompt_len,
                answer_len=answer_len,
                prefix_key=prefix_key,
                bucket=bucket_share_length(prompt_len),
            )
        )
    return candidates


def build_sharegpt_candidates(records: Iterable[dict[str, Any]], tokenizer, prefix_chars: int) -> list[Candidate]:
    return build_lmsys_candidates(records, tokenizer, prefix_chars)


def build_longbench_candidates(records: Iterable[dict[str, Any]], tokenizer, prefix_chars: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for record in records:
        prompt = coerce_text(record.get("input"))
        answer = coerce_text(record.get("answers", record.get("output")))
        if not prompt or not answer:
            continue
        prompt_len = estimate_len(prompt, tokenizer)
        answer_len = estimate_len(answer, tokenizer)
        prefix_key = normalize_text(prompt)[:prefix_chars]
        candidates.append(
            Candidate(
                raw=record,
                prompt=prompt,
                answer=answer,
                prompt_len=prompt_len,
                answer_len=answer_len,
                prefix_key=prefix_key,
                bucket=bucket_longbench(prompt_len),
            )
        )
    return candidates


def sample_groupwise(
    groups: dict[str, list[Candidate]],
    *,
    target_size: int,
    rng: random.Random,
    score_fn,
) -> list[Candidate]:
    group_items = list(groups.items())
    group_items.sort(key=lambda item: score_fn(item[1]), reverse=True)
    selected: list[Candidate] = []

    for _key, group in group_items:
        if len(selected) >= target_size:
            break
        rng.shuffle(group)
        selected.extend(group[: min(len(group), max(1, target_size // max(1, len(group_items))))])

    if len(selected) < target_size:
        remaining = [item for _key, group in group_items for item in group if item not in selected]
        remaining.sort(key=lambda item: (item.prompt_len, item.answer_len), reverse=True)
        selected.extend(remaining[: target_size - len(selected)])

    return selected[:target_size]


def select_lmsys_subset(candidates: list[Candidate], target_size: int, rng: random.Random) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.prefix_key].append(candidate)

    shared_groups = {key: value for key, value in groups.items() if len(value) >= 2 and key}
    singleton_groups = {key: value for key, value in groups.items() if len(value) < 2 or not key}

    # Prioritize groups with real prefix reuse and longer prompts.
    selected = sample_groupwise(
        shared_groups,
        target_size=max(1, int(target_size * 0.75)),
        rng=rng,
        score_fn=lambda group: (len(group), max(item.prompt_len for item in group)),
    )

    if len(selected) < target_size:
        remaining_needed = target_size - len(selected)
        singleton_items = [item for items in singleton_groups.values() for item in items]
        singleton_items.sort(key=lambda item: item.prompt_len, reverse=True)
        rng.shuffle(singleton_items[: min(len(singleton_items), 64)])
        selected.extend(singleton_items[:remaining_needed])

    return selected[:target_size]


def select_sharegpt_subset(candidates: list[Candidate], target_size: int, rng: random.Random) -> list[Candidate]:
    by_bucket: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate.bucket].append(candidate)

    # Preserve the heavy tail by explicitly reserving budget for long and xlong prompts.
    target_by_bucket = {
        "short": int(target_size * 0.35),
        "medium": int(target_size * 0.35),
        "long": int(target_size * 0.20),
        "xlong": target_size,
    }
    target_by_bucket["xlong"] -= sum(target_by_bucket.values()) - target_size

    selected: list[Candidate] = []
    for bucket in ("xlong", "long", "medium", "short"):
        items = list(by_bucket.get(bucket, []))
        rng.shuffle(items)
        items.sort(key=lambda item: item.prompt_len, reverse=True)
        selected.extend(items[: target_by_bucket.get(bucket, 0)])

    if len(selected) < target_size:
        leftovers = [item for bucket_items in by_bucket.values() for item in bucket_items if item not in selected]
        leftovers.sort(key=lambda item: item.prompt_len, reverse=True)
        selected.extend(leftovers[: target_size - len(selected)])

    return selected[:target_size]


def select_longbench_subset(candidates: list[Candidate], target_size: int, rng: random.Random) -> list[Candidate]:
    by_bucket: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate.bucket].append(candidate)

    target_by_bucket = {
        "0_8k": int(target_size * 0.10),
        "8k_32k": int(target_size * 0.25),
        "32k_64k": int(target_size * 0.30),
        "64k_plus": target_size,
    }
    target_by_bucket["64k_plus"] -= sum(target_by_bucket.values()) - target_size

    selected: list[Candidate] = []
    for bucket in ("64k_plus", "32k_64k", "8k_32k", "0_8k"):
        items = list(by_bucket.get(bucket, []))
        items.sort(key=lambda item: item.prompt_len, reverse=True)
        rng.shuffle(items[: min(len(items), 32)])
        selected.extend(items[: target_by_bucket.get(bucket, 0)])

    if len(selected) < target_size:
        leftovers = [item for bucket_items in by_bucket.values() for item in bucket_items if item not in selected]
        leftovers.sort(key=lambda item: item.prompt_len, reverse=True)
        selected.extend(leftovers[: target_size - len(selected)])

    return selected[:target_size]


def summarize(candidates: list[Candidate], selected: list[Candidate], dataset_type: str, seed: int) -> dict[str, Any]:
    def length_stats(items: list[Candidate]) -> dict[str, float]:
        if not items:
            return {"count": 0, "avg_prompt_len": 0.0, "avg_answer_len": 0.0, "max_prompt_len": 0}
        return {
            "count": len(items),
            "avg_prompt_len": sum(item.prompt_len for item in items) / len(items),
            "avg_answer_len": sum(item.answer_len for item in items) / len(items),
            "max_prompt_len": max(item.prompt_len for item in items),
        }

    selected_prefix_counts = Counter(item.prefix_key for item in selected if item.prefix_key)
    candidate_prefix_counts = Counter(item.prefix_key for item in candidates if item.prefix_key)

    return {
        "dataset_type": dataset_type,
        "seed": seed,
        "input_size": len(candidates),
        "selected_size": len(selected),
        "selected_stats": length_stats(selected),
        "input_stats": length_stats(candidates),
        "selected_bucket_counts": Counter(item.bucket for item in selected),
        "input_bucket_counts": Counter(item.bucket for item in candidates),
        "selected_shared_prefix_groups": sum(1 for count in selected_prefix_counts.values() if count >= 2),
        "input_shared_prefix_groups": sum(1 for count in candidate_prefix_counts.values() if count >= 2),
        "top_selected_prefix_groups": selected_prefix_counts.most_common(20),
    }


def filter_candidates(
    candidates: list[Candidate],
    *,
    max_prompt_len: int | None,
    max_total_len: int | None,
) -> list[Candidate]:
    filtered: list[Candidate] = []
    for candidate in candidates:
        if max_prompt_len is not None and candidate.prompt_len > max_prompt_len:
            continue
        if max_total_len is not None and (candidate.prompt_len + candidate.answer_len) > max_total_len:
            continue
        filtered.append(candidate)
    return filtered


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")

    tokenizer = maybe_load_tokenizer(args.tokenizer)
    records = load_records(input_path)
    target_size = args.target_size or default_target_size(args.dataset_type)

    if args.dataset_type == "lmsys":
        candidates = build_lmsys_candidates(records, tokenizer, args.prefix_chars)
    elif args.dataset_type == "sharegpt":
        candidates = build_sharegpt_candidates(records, tokenizer, args.prefix_chars)
    else:
        candidates = build_longbench_candidates(records, tokenizer, args.prefix_chars)

    candidates = filter_candidates(
        candidates,
        max_prompt_len=args.max_prompt_len,
        max_total_len=args.max_total_len,
    )

    if args.dataset_type == "lmsys":
        selected = select_lmsys_subset(candidates, target_size, rng)
    elif args.dataset_type == "sharegpt":
        selected = select_sharegpt_subset(candidates, target_size, rng)
    else:
        selected = select_longbench_subset(candidates, target_size, rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for candidate in selected:
            fh.write(json.dumps(make_custom_row(candidate), ensure_ascii=False) + "\n")

    manifest = summarize(candidates, selected, args.dataset_type, args.seed)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
