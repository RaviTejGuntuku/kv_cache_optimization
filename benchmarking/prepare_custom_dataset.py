#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local or Hugging Face data into SGLang custom JSONL format."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--source-path", help="Input JSON/JSONL path.")
    parser.add_argument("--hf-dataset", help="Optional Hugging Face dataset repo id.")
    parser.add_argument("--hf-split", default="train", help="Dataset split for --hf-dataset.")
    parser.add_argument("--conversation-field", default=None, help="Field containing a conversation list.")
    parser.add_argument("--input-field", default=None, help="Field containing the user prompt.")
    parser.add_argument("--output-field", default=None, help="Field containing the assistant answer/reference.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to write.")
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source_path:
        path = Path(args.source_path)
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return json.loads(path.read_text(encoding="utf-8"))

    if args.hf_dataset:
        from datasets import load_dataset

        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        return [dict(row) for row in dataset]

    raise ValueError("One of --source-path or --hf-dataset is required.")


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, list):
        for item in value:
            coerced = _coerce_text(item)
            if coerced:
                return coerced
    return None


def extract_pair(record: dict[str, Any], args: argparse.Namespace) -> tuple[str, str] | None:
    if args.conversation_field:
        conversation = record.get(args.conversation_field, [])
        if len(conversation) < 2:
            return None
        user_turn = conversation[0]
        assistant_turn = conversation[1]
        user_content = _coerce_text(user_turn.get("content", user_turn.get("value", "")))
        assistant_content = _coerce_text(
            assistant_turn.get("content", assistant_turn.get("value", ""))
        )
        if user_content and assistant_content:
            return user_content, assistant_content
        return None

    if args.input_field and args.output_field:
        prompt = _coerce_text(record.get(args.input_field))
        answer = _coerce_text(record.get(args.output_field))
        if prompt and answer:
            return prompt, answer
        return None

    if "conversations" in record:
        return extract_pair(
            record,
            argparse.Namespace(**{**vars(args), "conversation_field": "conversations"}),
        )
    if "conversation" in record:
        return extract_pair(
            record,
            argparse.Namespace(**{**vars(args), "conversation_field": "conversation"}),
        )
    return None


def iter_output_rows(records: Iterable[dict[str, Any]], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    emitted = 0
    for record in records:
        pair = extract_pair(record, args)
        if pair is None:
            continue
        prompt, answer = pair
        yield {
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
        emitted += 1
        if args.limit is not None and emitted >= args.limit:
            break


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(args)

    with output_path.open("w", encoding="utf-8") as fh:
        for row in iter_output_rows(records, args):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
