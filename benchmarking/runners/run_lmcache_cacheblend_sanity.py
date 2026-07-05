#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.adapters import AdapterConfig, build_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal LMCache CacheBlend sanity check."
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--min-cached-tokens", type=int, default=512)
    parser.add_argument(
        "--target-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of post-seed CacheBlend target requests to measure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = AdapterConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    with build_adapter("lmcache_cacheblend", config) as adapter:
        tokenizer = adapter.get_tokenizer()

        def encode_segment(text: str) -> list[int]:
            return tokenizer.encode(text)[1:]

        sep = encode_segment(getattr(adapter, "blend_special_str", " # # "))
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        sys_prompt = [1, 733, 16289, 28793] + tokenizer.encode(
            "You are a very helpful assistant. "
            "Please answer the question with instructions."
        )
        chunk1 = tokenizer.encode("Hello, how are you?" * 500)[1:]
        chunk2 = tokenizer.encode("Hello, what's up?" * 500)[1:]
        chunk3 = tokenizer.encode("Hi, what are you up to?" * 500)[1:]
        first_prompt = (
            sys_prompt
            + sep
            + chunk1
            + sep
            + chunk2
            + sep
            + chunk3
            + sep
            + tokenizer.encode("Hello, my name is")[1:]
            + [733, 28748, 16289, 28793]
        )
        second_prompt = (
            sys_prompt
            + sep
            + chunk2
            + sep
            + chunk1
            + sep
            + chunk3
            + sep
            + tokenizer.encode("Hello, how are you?")[1:]
            + [733, 28748, 16289, 28793]
        )
        third_prompt = (
            sys_prompt
            + sep
            + chunk2
            + sep
            + chunk1
            + sep
            + chunk3
            + sep
            + tokenizer.encode("Hello, what's up?")[1:]
            + [733, 28748, 16289, 28793]
        )

        warmup = adapter.measure_request({"prompt_token_ids": warmup_prompt})
        seed = adapter.measure_request({"prompt_token_ids": first_prompt})
        targets = [
            adapter.measure_request({"prompt_token_ids": second_prompt}),
        ]
        if args.target_count == 2:
            targets.append(adapter.measure_request({"prompt_token_ids": third_prompt}))

    rows = {
        "model": args.model,
        "blend_special_str": getattr(adapter, "blend_special_str", " # # "),
        "warmup": warmup,
        "seed": seed,
        "targets": targets,
        "pass": (
            any(
                int(target.get("lmcache_returned_cached_tokens") or 0)
                >= args.min_cached_tokens
                or int(target.get("lmcache_retrieved_tokens") or 0)
                >= args.min_cached_tokens
                or int(target.get("num_cached_tokens") or 0)
                >= args.min_cached_tokens
                or float(target.get("oracle_hbm_materialization_ms") or 0.0) > 0.0
                for target in targets
            )
        ),
    }
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    if not rows["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
