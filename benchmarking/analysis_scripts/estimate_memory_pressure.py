#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate KV-cache memory pressure for a benchmark subset."
    )
    parser.add_argument("--dataset", required=True, help="Custom JSONL dataset path.")
    parser.add_argument("--gpu-kv-capacity-blocks", type=int, default=20000)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--concurrency", type=int, nargs="*", default=[32, 64, 128, 256, 512])
    parser.add_argument(
        "--assume-generated-tokens",
        type=int,
        default=256,
        help="Fallback generated-token count if output length is unavailable.",
    )
    return parser.parse_args()


def coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, list):
        for item in value:
            candidate = coerce_text(item)
            if candidate:
                return candidate
    if isinstance(value, dict):
        for key in ("content", "value", "text"):
            if key in value:
                candidate = coerce_text(value[key])
                if candidate:
                    return candidate
    return None


def load_lengths(path: Path, assume_generated_tokens: int) -> list[dict[str, int]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_no} in {path}"
                ) from exc
            conversation = record.get("conversations", [])
            prompt = coerce_text(conversation[0]) if len(conversation) > 0 else ""
            answer = coerce_text(conversation[1]) if len(conversation) > 1 else ""
            prompt_len = record.get("prompt_len")
            if prompt_len is None:
                prompt_len = max(1, math.ceil(len(prompt.split()) * 1.3)) if prompt else 0

            answer_len = record.get("output_len")
            if answer_len is None:
                answer_len = (
                    max(1, math.ceil(len(answer.split()) * 1.3))
                    if answer
                    else assume_generated_tokens
                )
            rows.append({"prompt_len": prompt_len, "answer_len": answer_len})
    return rows


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def main() -> None:
    args = parse_args()
    rows = load_lengths(Path(args.dataset), args.assume_generated_tokens)
    if not rows:
        raise ValueError("Dataset is empty.")

    prompt_blocks = [ceil_div(row["prompt_len"], args.page_size) for row in rows]
    full_blocks = [
        ceil_div(row["prompt_len"] + row["answer_len"], args.page_size) for row in rows
    ]
    sorted_prompt = sorted(prompt_blocks, reverse=True)
    sorted_full = sorted(full_blocks, reverse=True)

    pressure = []
    for concurrency in args.concurrency:
        worst_prompt_blocks = sum(sorted_prompt[: min(concurrency, len(sorted_prompt))])
        worst_full_blocks = sum(sorted_full[: min(concurrency, len(sorted_full))])
        pressure.append(
            {
                "concurrency": concurrency,
                "prompt_only_blocks": worst_prompt_blocks,
                "prompt_only_pressure": worst_prompt_blocks / args.gpu_kv_capacity_blocks,
                "full_request_blocks": worst_full_blocks,
                "full_request_pressure": worst_full_blocks / args.gpu_kv_capacity_blocks,
            }
        )

    payload = {
        "dataset": args.dataset,
        "num_requests": len(rows),
        "page_size": args.page_size,
        "gpu_kv_capacity_blocks": args.gpu_kv_capacity_blocks,
        "mean_prompt_blocks": sum(prompt_blocks) / len(prompt_blocks),
        "mean_full_blocks": sum(full_blocks) / len(full_blocks),
        "max_prompt_blocks": max(prompt_blocks),
        "max_full_blocks": max(full_blocks),
        "pressure_by_concurrency": pressure,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
