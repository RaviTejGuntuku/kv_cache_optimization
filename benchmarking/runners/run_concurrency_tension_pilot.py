#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.adapters import AdapterConfig, build_adapter
from benchmarking.headroom_empirical.interface import candidate_object_ids
from benchmarking.headroom_empirical.nvtx import nvtx_range
from benchmarking.headroom_empirical.schema import load_workload_bundle
from benchmarking.runners.headroom_common import resolve_timestamped_output_root, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concurrency-tension headroom pilot.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--target-count-per-level", type=int, default=4)
    parser.add_argument("--max-counterfactuals-per-request", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    return parser.parse_args()


def _measure_batch_case(
    *,
    adapter,
    system: str,
    bundle,
    cohort_requests: list,
    target_request,
    mode: str,
    preload_object_ids: list[str] | None = None,
) -> dict:
    object_index = {obj.object_id: obj for obj in bundle.objects}
    target_idx = next(idx for idx, request in enumerate(cohort_requests) if request.request_id == target_request.request_id)
    preload_ids = list(preload_object_ids or [])
    if mode == "oracle0" and not preload_ids:
        preload_ids = list(target_request.reusable_object_ids)
    target_preload_prompts = [object_index[obj_id].seed_prompt for obj_id in preload_ids]

    with nvtx_range(f"{system}:{mode}:clear_state"):
        adapter.clear_state()
    if target_preload_prompts:
        with nvtx_range(f"{system}:{mode}:target_preload"):
            adapter.prewarm(target_preload_prompts)
    prompts = [request.prompt for request in cohort_requests]
    with nvtx_range(f"{system}:{mode}:cohort_batch"):
        batch_metrics = adapter.measure_batch(prompts)
    target_metrics = dict(batch_metrics[target_idx])

    measurement_metadata = {
        "cohort_request_ids": [request.request_id for request in cohort_requests],
        "cohort_prompt_tokens": [int(request.prompt_tokens) for request in cohort_requests],
        "target_index_in_batch": target_idx,
    }
    prefill_time_ms = (
        float(target_metrics["prefill_time_ms"])
        if target_metrics["prefill_time_ms"] is not None
        else None
    )
    if mode == "oracle0" and system.startswith("lmcache_") and prefill_time_ms is not None:
        raw_prefill_time_ms = prefill_time_ms
        retrieve_time_ms = float(target_metrics.get("lmcache_retrieve_time_ms") or 0.0)
        prefill_time_ms = max(0.0, raw_prefill_time_ms - retrieve_time_ms)
        measurement_metadata.update(
            {
                "oracle0_fetch_exclusion_applied": True,
                "raw_prefill_time_ms": raw_prefill_time_ms,
                "fetch_excluded_prefill_time_ms": prefill_time_ms,
                "excluded_retrieve_time_ms": retrieve_time_ms,
            }
        )

    return {
        "system": system,
        "mode": mode,
        "track": target_request.track,
        "concurrency": len(cohort_requests),
        "target_request_id": target_request.request_id,
        "target_prompt_tokens": target_request.prompt_tokens,
        "target_output_len": target_request.output_len,
        "target_reusable_object_ids": list(target_request.reusable_object_ids),
        "preload_object_ids": preload_ids,
        "wall_time_ms": float(target_metrics["wall_time_ms"]),
        "prefill_time_ms": prefill_time_ms,
        "decode_time_ms": (
            float(target_metrics["decode_time_ms"])
            if target_metrics["decode_time_ms"] is not None
            else None
        ),
        "ttft_ms": float(target_metrics["ttft_ms"]) if target_metrics["ttft_ms"] is not None else None,
        "num_cached_tokens": (
            int(target_metrics["num_cached_tokens"])
            if target_metrics["num_cached_tokens"] is not None
            else None
        ),
        "metadata": {
            **measurement_metadata,
            "bundle_root": str(Path(args.bundle_root)),
            "history_request_ids": [],
            **{
                key: target_metrics[key]
                for key in (
                    "lmcache_retrieve_requests",
                    "lmcache_requested_tokens",
                    "lmcache_retrieved_tokens",
                    "lmcache_retrieve_time_ms",
                    "lmcache_retrieve_process_tokens_ms",
                    "lmcache_retrieve_broadcast_ms",
                    "lmcache_retrieve_to_gpu_ms",
                    "lmcache_lookup_requests",
                    "lmcache_lookup_tokens",
                    "lmcache_lookup_hit_tokens",
                    "lmcache_lookup_time_ms",
                    "lmcache_remote_read_requests_delta",
                    "lmcache_remote_read_bytes_delta",
                )
                if key in target_metrics
            },
        },
    }


def _choose_cohorts(bundle, concurrency: int, target_count_per_level: int) -> list[tuple[object, list[object]]]:
    requests = list(bundle.requests)
    if concurrency > len(requests):
        raise ValueError(f"Concurrency {concurrency} exceeds request count {len(requests)} for bundle {bundle.bundle_name}")
    cohorts: list[tuple[object, list[object]]] = []
    max_start = len(requests) - concurrency + 1
    for start in range(min(target_count_per_level, max_start)):
        cohort_requests = requests[start : start + concurrency]
        target_request = cohort_requests[0]
        cohorts.append((target_request, cohort_requests))
    return cohorts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _summary(rows: list[dict]) -> dict:
    by_concurrency: dict[int, list[dict]] = {}
    for row in rows:
        by_concurrency.setdefault(int(row["concurrency"]), []).append(row)
    levels = []
    for concurrency in sorted(by_concurrency):
        level_rows = by_concurrency[concurrency]
        levels.append(
            {
                "concurrency": concurrency,
                "requests_measured": len(level_rows),
                "prefill_time_ms_mean": statistics.fmean(float(row["prefill_time_ms"]) for row in level_rows if row["prefill_time_ms"] is not None),
                "prefill_time_ms_p50": statistics.median(float(row["prefill_time_ms"]) for row in level_rows if row["prefill_time_ms"] is not None),
                "ttft_ms_mean": statistics.fmean(float(row["ttft_ms"]) for row in level_rows if row["ttft_ms"] is not None),
                "num_cached_tokens_mean": statistics.fmean(float(row["num_cached_tokens"]) for row in level_rows if row["num_cached_tokens"] is not None),
            }
        )
    return {
        "levels": levels,
        "requests_measured": len(rows),
    }


def main() -> None:
    global args
    args = parse_args()
    output_root = resolve_timestamped_output_root(args.output_root, no_timestamp=args.no_timestamp)
    bundle = load_workload_bundle(Path(args.bundle_root))
    manifest = {
        "experiment": "concurrency_tension_pilot",
        "system": args.system,
        "model": args.model,
        "bundle_root": args.bundle_root,
        "concurrency_levels": args.concurrency_levels,
        "target_count_per_level": args.target_count_per_level,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "track": bundle.track,
    }
    write_manifest(output_root / "run_manifest.json", manifest)

    adapter_config = AdapterConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    baseline_rows: list[dict] = []
    oracle_rows: list[dict] = []
    with build_adapter(args.system, adapter_config) as adapter:
        if args.warmup and bundle.requests:
            adapter.clear_state()
            adapter.prewarm([bundle.requests[0].prompt])
            adapter.clear_state()
        for concurrency in args.concurrency_levels:
            for target_request, cohort_requests in _choose_cohorts(
                bundle,
                concurrency,
                args.target_count_per_level,
            ):
                baseline_rows.append(
                    _measure_batch_case(
                        adapter=adapter,
                        system=args.system,
                        bundle=bundle,
                        cohort_requests=cohort_requests,
                        target_request=target_request,
                        mode="baseline",
                    )
                )
                oracle_rows.append(
                    _measure_batch_case(
                        adapter=adapter,
                        system=args.system,
                        bundle=bundle,
                        cohort_requests=cohort_requests,
                        target_request=target_request,
                        mode="oracle0",
                    )
                )

    baseline_idx = {
        (int(row["concurrency"]), row["target_request_id"]): row
        for row in baseline_rows
    }
    object_index = {obj.object_id: obj for obj in bundle.objects}
    marginal_rows: list[dict] = []
    with build_adapter(args.system, adapter_config) as adapter:
        if args.warmup and bundle.requests:
            adapter.clear_state()
            adapter.prewarm([bundle.requests[0].prompt])
            adapter.clear_state()
        for concurrency in args.concurrency_levels:
            for target_request, cohort_requests in _choose_cohorts(
                bundle,
                concurrency,
                args.target_count_per_level,
            ):
                baseline_row = baseline_idx[(concurrency, target_request.request_id)]
                baseline_cached_tokens = baseline_row.get("num_cached_tokens")
                for object_id in candidate_object_ids(
                    bundle,
                    target_request,
                    limit=args.max_counterfactuals_per_request,
                ):
                    cf_row = _measure_batch_case(
                        adapter=adapter,
                        system=args.system,
                        bundle=bundle,
                        cohort_requests=cohort_requests,
                        target_request=target_request,
                        mode=f"counterfactual__{object_id}",
                        preload_object_ids=[object_id],
                    )
                    object_meta = object_index[object_id]
                    counterfactual_cached_tokens = cf_row.get("num_cached_tokens")
                    cached_token_gain = None
                    if baseline_cached_tokens is not None and counterfactual_cached_tokens is not None:
                        cached_token_gain = int(counterfactual_cached_tokens) - int(baseline_cached_tokens)
                    marginal_rows.append(
                        {
                            "system": args.system,
                            "track": target_request.track,
                            "concurrency": concurrency,
                            "target_request_id": target_request.request_id,
                            "object_id": object_id,
                            "object_type": object_meta.object_type,
                            "source_tier": object_meta.source_tier,
                            "object_size_tokens": int(object_meta.metadata.get("object_size_tokens", 0)),
                            "baseline_prefill_time_ms": baseline_row.get("prefill_time_ms"),
                            "counterfactual_prefill_time_ms": cf_row.get("prefill_time_ms"),
                            "marginal_gain_ms": (
                                float(baseline_row["prefill_time_ms"]) - float(cf_row["prefill_time_ms"])
                                if baseline_row.get("prefill_time_ms") is not None and cf_row.get("prefill_time_ms") is not None
                                else None
                            ),
                            "baseline_num_cached_tokens": baseline_cached_tokens,
                            "counterfactual_num_cached_tokens": counterfactual_cached_tokens,
                            "cached_token_gain": cached_token_gain,
                            "was_missed_in_baseline": bool(cached_token_gain is not None and cached_token_gain > 0),
                            "repair_expected": object_meta.object_type == "approximate",
                            "metadata": {
                                "cohort_request_ids": [request.request_id for request in cohort_requests],
                                "preload_object_ids": [object_id],
                            },
                        }
                    )

    _write_jsonl(output_root / "baseline_batch_measurements.jsonl", baseline_rows)
    _write_jsonl(output_root / "oracle0_batch_measurements.jsonl", oracle_rows)
    _write_jsonl(output_root / "marginal_counterfactuals_batch.jsonl", marginal_rows)
    (output_root / "baseline_summary.json").write_text(json.dumps(_summary(baseline_rows), indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "oracle0_summary.json").write_text(json.dumps(_summary(oracle_rows), indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "marginal_summary.json").write_text(
        json.dumps(
            {
                "rows_measured": len(marginal_rows),
                "missed_rows": sum(1 for row in marginal_rows if row.get("was_missed_in_baseline")),
                "mean_marginal_gain_ms": (
                    statistics.fmean(float(row["marginal_gain_ms"]) for row in marginal_rows if row.get("marginal_gain_ms") is not None)
                    if marginal_rows
                    else 0.0
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
