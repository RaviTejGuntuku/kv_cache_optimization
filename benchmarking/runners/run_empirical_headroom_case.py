#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.case_runner import (
    run_replay_case,
    run_single_case,
    write_case_measurement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one empirical headroom measurement case.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preload-object-ids", nargs="*", default=[])
    parser.add_argument("--history-request-ids", nargs="*", default=[])
    parser.add_argument("--warmup-request-ids", nargs="*", default=[])
    parser.add_argument("--warmup-full-path", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root)
    if args.history_request_ids:
        measurement = run_replay_case(
            system=args.system,
            model=args.model,
            bundle_root=bundle_root,
            request_id=args.request_id,
            history_request_ids=args.history_request_ids,
            preload_object_ids=args.preload_object_ids,
            mode=args.mode,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            warmup_request_ids=args.warmup_request_ids,
            warmup_full_path=args.warmup_full_path,
        )
    else:
        measurement = run_single_case(
            system=args.system,
            model=args.model,
            bundle_root=bundle_root,
            request_id=args.request_id,
            preload_object_ids=args.preload_object_ids,
            mode=args.mode,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            warmup_request_ids=args.warmup_request_ids,
            warmup_full_path=args.warmup_full_path,
        )
    write_case_measurement(Path(args.output), measurement)


if __name__ == "__main__":
    main()
