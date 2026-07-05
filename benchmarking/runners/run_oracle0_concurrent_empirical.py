#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.adapters import AdapterConfig, build_adapter
from benchmarking.headroom_empirical.case_runner import (
    _oracle_kv_transfer_params,
    _preload_prompts,
    _request_prompt_payload,
)
from benchmarking.headroom_empirical.interface import build_request_execution_plan
from benchmarking.headroom_empirical.nvtx import nvtx_range
from benchmarking.headroom_empirical.runtime_alignment import align_bundle_to_runtime
from benchmarking.headroom_empirical.schema import (
    HeadroomRequest,
    RequestMeasurement,
    WorkloadBundle,
    load_workload_bundle,
    write_measurements,
)
from benchmarking.runners.headroom_common import profile_payload, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Oracle 0/baseline measurements with concurrent target batches.")
    parser.add_argument("--system", required=True, choices=["vllm_apc", "lmcache_exact", "lmcache_cacheblend"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--allow-unsafe-lmcache-per-request-concurrency",
        action="store_true",
        help=(
            "Allow LMCache broad runs with concurrency > 1 even though vLLM/LMCache "
            "timings may be batch-level and unsafe for per-request Oracle gaps."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _index_requests(bundle: WorkloadBundle) -> dict[str, HeadroomRequest]:
    return {request.request_id: request for request in bundle.requests}


def _cohorts(requests: list[HeadroomRequest], concurrency: int) -> list[tuple[list[HeadroomRequest], list[HeadroomRequest]]]:
    cohorts = []
    for start in range(0, len(requests), concurrency):
        cohort = requests[start : start + concurrency]
        if len(cohort) != concurrency:
            continue
        history = requests[:start]
        cohorts.append((history, cohort))
    return cohorts


def _payload_for_request(adapter, request: HeadroomRequest, *, system: str, mode: str) -> Any:
    payload = _request_prompt_payload(request)
    if mode == "oracle0" and system.startswith("lmcache_"):
        kv_transfer_params = _oracle_kv_transfer_params(adapter, request)
        if kv_transfer_params and isinstance(payload, dict):
            payload = {
                **payload,
                "__kv_transfer_params": kv_transfer_params,
            }
    return payload


def _prewarm_fcfs(adapter, prompts: list[Any]) -> None:
    for prompt in prompts:
        adapter.prewarm([prompt])


def _execution_metadata(
    *,
    bundle: WorkloadBundle,
    request: HeadroomRequest,
    history: list[HeadroomRequest],
    cohort: list[HeadroomRequest],
    preload_object_ids: list[str],
    mode: str,
    system: str,
    model: str,
    bundle_root: Path,
    metrics: dict,
) -> dict:
    plan = build_request_execution_plan(
        bundle,
        request_id=request.request_id,
        mode=mode,
        history_request_ids=[item.request_id for item in history],
        preload_object_ids=preload_object_ids,
    )
    measurement_metadata = dict(metrics.pop("measurement_metadata", {}) or {})
    return {
        "bundle_root": str(bundle_root),
        "system": system,
        "tokenization_model": model,
        "concurrency": len(cohort),
        "cohort_request_ids": [item.request_id for item in cohort],
        "history_request_ids": [item.request_id for item in history],
        "workload_prompt_tokens": request.metadata.get("workload_prompt_tokens"),
        "runtime_prompt_tokens": request.prompt_tokens,
        "runtime_alignment_success": request.metadata.get("runtime_alignment_success"),
        "runtime_alignment_errors": request.metadata.get("runtime_alignment_errors", []),
        "execution_plan": {
            "request_id": plan.request_id,
            "track": plan.track,
            "mode": plan.mode,
            "history_request_ids": plan.history_request_ids,
            "preload_object_ids": plan.preload_object_ids,
            "reusable_object_ids": plan.reusable_object_ids,
            "repair_object_ids": plan.repair_object_ids,
            "missed_candidate_object_ids": plan.missed_candidate_object_ids,
            "ordered_occurrences": [
                {
                    "object_id": occ.object_id,
                    "object_type": occ.object_type,
                    "occurrence_index": occ.occurrence_index,
                    "token_start": occ.token_start,
                    "token_end": occ.token_end,
                    "approximate": occ.approximate,
                }
                for occ in plan.ordered_occurrences
            ],
        },
        **measurement_metadata,
        **{
            key: metrics[key]
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
                "oracle_hbm_materialization_ms",
                "oracle_repair_compute_ms",
                "oracle_load_kv_total_ms",
                "lmcache_returned_cached_tokens",
            )
            if key in metrics
        },
    }


def _measurement_from_metrics(
    *,
    bundle: WorkloadBundle,
    request: HeadroomRequest,
    history: list[HeadroomRequest],
    cohort: list[HeadroomRequest],
    preload_object_ids: list[str],
    mode: str,
    system: str,
    model: str,
    bundle_root: Path,
    metrics: dict,
) -> RequestMeasurement:
    metrics = dict(metrics)
    prefill_time_ms = (
        float(metrics["prefill_time_ms"])
        if metrics.get("prefill_time_ms") is not None
        else None
    )
    if mode == "oracle0" and system.startswith("lmcache_") and prefill_time_ms is not None:
        raw_prefill_time_ms = prefill_time_ms
        materialization_ms = float(metrics.get("oracle_hbm_materialization_ms") or 0.0)
        prefill_time_ms = max(0.0, raw_prefill_time_ms - materialization_ms)
        raw_ttft_ms = (
            float(metrics["ttft_ms"]) if metrics.get("ttft_ms") is not None else None
        )
        adjusted_ttft_ms = (
            max(0.0, raw_ttft_ms - materialization_ms)
            if raw_ttft_ms is not None
            else None
        )
        metrics.setdefault("measurement_metadata", {})
        metrics["measurement_metadata"].update(
            {
                "oracle0_fetch_exclusion_applied": True,
                "oracle0_timed_region_definition": "raw_prefill_minus_hbm_materialization",
                "raw_prefill_time_ms": raw_prefill_time_ms,
                "fetch_excluded_prefill_time_ms": prefill_time_ms,
                "excluded_hbm_materialization_ms": materialization_ms,
                "raw_ttft_ms": raw_ttft_ms,
                "fetch_excluded_ttft_ms": adjusted_ttft_ms,
            }
        )
        metrics["ttft_ms"] = adjusted_ttft_ms

    return RequestMeasurement(
        system=system,
        mode=mode,
        track=request.track,
        request_id=request.request_id,
        wall_time_ms=float(metrics["wall_time_ms"]),
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=(
            float(metrics["decode_time_ms"])
            if metrics.get("decode_time_ms") is not None
            else None
        ),
        ttft_ms=float(metrics["ttft_ms"]) if metrics.get("ttft_ms") is not None else None,
        num_cached_tokens=(
            int(metrics["num_cached_tokens"])
            if metrics.get("num_cached_tokens") is not None
            else None
        ),
        prompt_tokens=request.prompt_tokens,
        output_len=request.output_len,
        preload_object_ids=preload_object_ids,
        repair_expected=any(
            object_id.startswith("chunk_approx_")
            or object_id.startswith("rag_doc_approx")
            for object_id in preload_object_ids
        ),
        metadata=_execution_metadata(
            bundle=bundle,
            request=request,
            history=history,
            cohort=cohort,
            preload_object_ids=preload_object_ids,
            mode=mode,
            system=system,
            model=model,
            bundle_root=bundle_root,
            metrics=metrics,
        ),
    )


def _summary(rows: list[dict]) -> dict:
    wall = [float(row["wall_time_ms"]) for row in rows]
    prefill = [float(row["prefill_time_ms"]) for row in rows if row.get("prefill_time_ms") is not None]
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    cached = [int(row["num_cached_tokens"]) for row in rows if row.get("num_cached_tokens") is not None]
    return {
        "requests_measured": len(rows),
        "wall_time_ms_mean": statistics.fmean(wall) if wall else 0.0,
        "prefill_time_ms_mean": statistics.fmean(prefill) if prefill else None,
        "prefill_time_ms_p50": statistics.median(prefill) if prefill else None,
        "ttft_ms_mean": statistics.fmean(ttft) if ttft else None,
        "ttft_ms_p50": statistics.median(ttft) if ttft else None,
        "num_cached_tokens_mean": statistics.fmean(cached) if cached else None,
    }


def _run_mode(
    *,
    adapter,
    bundle: WorkloadBundle,
    cohorts: list[tuple[list[HeadroomRequest], list[HeadroomRequest]]],
    system: str,
    model: str,
    bundle_root: Path,
    mode: str,
) -> list[RequestMeasurement]:
    rows: list[RequestMeasurement] = []
    request_index = _index_requests(bundle)
    for history, cohort in cohorts:
        with nvtx_range(f"{system}:{mode}:clear_state"):
            adapter.clear_state()
        if history:
            history_prompts = [_request_prompt_payload(request_index[item.request_id]) for item in history]
            with nvtx_range(f"{system}:{mode}:history_replay"):
                _prewarm_fcfs(adapter, history_prompts)
        preload_by_request: dict[str, list[str]] = {}
        if mode == "oracle0":
            preload_prompts: list[Any] = []
            for request in cohort:
                preload_ids = list(request.reusable_object_ids)
                preload_by_request[request.request_id] = preload_ids
                preload_prompts.extend(
                    _preload_prompts(bundle, request, preload_ids, system=system)
                )
            if preload_prompts:
                with nvtx_range(f"{system}:{mode}:cohort_preload"):
                    adapter.prewarm(preload_prompts)
                if system.startswith("lmcache_"):
                    time.sleep(1.0)

        payloads = [
            _payload_for_request(adapter, request, system=system, mode=mode)
            for request in cohort
        ]
        with nvtx_range(f"{system}:{mode}:target_cohort"):
            metric_rows = adapter.measure_batch(payloads)
        for request, metrics in zip(cohort, metric_rows, strict=True):
            if system.startswith("lmcache_"):
                metrics.setdefault("measurement_metadata", {})
                metrics["measurement_metadata"]["lmcache_async_loading_enabled"] = bool(
                    getattr(adapter, "async_loading_enabled", False)
                )
            preload_ids = preload_by_request.get(request.request_id, [])
            rows.append(
                _measurement_from_metrics(
                    bundle=bundle,
                    request=request,
                    history=history,
                    cohort=cohort,
                    preload_object_ids=preload_ids,
                    mode=mode,
                    system=system,
                    model=model,
                    bundle_root=bundle_root,
                    metrics=metrics,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    if (
        args.system.startswith("lmcache_")
        and args.concurrency > 1
        and not args.allow_unsafe_lmcache_per_request_concurrency
    ):
        raise RuntimeError(
            "LMCache broad Oracle 0 with concurrency > 1 is disabled in this "
            "per-request runner because vLLM/LMCache can expose cohort-level "
            "prefill/TTFT timings. Run concurrency=1 for valid request-level "
            "broad headroom, or add a cohort-level analyzer and pass "
            "--allow-unsafe-lmcache-per-request-concurrency only for diagnostic "
            "data that will not be interpreted as per-request headroom."
        )
    output_root = Path(args.output_root)
    baseline_root = output_root / "raw" / "baseline_replay_fcfs"
    oracle_root = output_root / "raw" / "oracle0_fcfs"
    bundle_root = Path(args.bundle_root)
    bundle = load_workload_bundle(bundle_root)
    requests = bundle.requests[args.request_offset :]
    if args.request_limit is not None:
        requests = requests[: args.request_limit]
    cohorts = _cohorts(requests, args.concurrency)

    manifest = profile_payload(
        "oracle0_concurrent_empirical",
        "oracle0_concurrent",
        {
            "system": args.system,
            "model": args.model,
            "bundle_root": args.bundle_root,
            "request_count": len(requests),
            "cohort_count": len(cohorts),
            "concurrency": args.concurrency,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "track": bundle.track,
        },
    )
    write_manifest(output_root / "run_manifest.json", manifest)
    write_manifest(baseline_root / "run_manifest.json", manifest)
    write_manifest(oracle_root / "run_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    adapter_config = AdapterConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    with build_adapter(args.system, adapter_config) as adapter:
        baseline_bundle = align_bundle_to_runtime(bundle, adapter)
        baseline_requests = baseline_bundle.requests[args.request_offset :]
        if args.request_limit is not None:
            baseline_requests = baseline_requests[: args.request_limit]
        baseline_cohorts = _cohorts(baseline_requests, args.concurrency)
        requests = bundle.requests[args.request_offset :]
        if args.request_limit is not None:
            requests = requests[: args.request_limit]
        if args.warmup and requests:
            adapter.clear_state()
            adapter.prewarm([_request_prompt_payload(baseline_requests[0])])
            adapter.clear_state()
        baseline_rows = _run_mode(
            adapter=adapter,
            bundle=baseline_bundle,
            cohorts=baseline_cohorts,
            system=args.system,
            model=args.model,
            bundle_root=bundle_root,
            mode="baseline_replay",
        )

    # Use a fresh engine for Oracle 0. This is required for LMCache systems:
    # reset_prefix_cache(reset_connector=True) does not reliably make the
    # external CPU/LMCache state equivalent to a fresh baseline-independent run.
    with build_adapter(args.system, adapter_config) as adapter:
        oracle_bundle = align_bundle_to_runtime(bundle, adapter)
        oracle_requests = oracle_bundle.requests[args.request_offset :]
        if args.request_limit is not None:
            oracle_requests = oracle_requests[: args.request_limit]
        oracle_cohorts = _cohorts(oracle_requests, args.concurrency)
        if args.warmup and oracle_requests:
            adapter.clear_state()
            adapter.prewarm([_request_prompt_payload(oracle_requests[0])])
            adapter.clear_state()
        oracle_rows = _run_mode(
            adapter=adapter,
            bundle=oracle_bundle,
            cohorts=oracle_cohorts,
            system=args.system,
            model=args.model,
            bundle_root=bundle_root,
            mode="oracle0",
        )

    write_measurements(baseline_root / "baseline_replay_measurements.jsonl", baseline_rows)
    write_measurements(oracle_root / "oracle0_measurements.jsonl", oracle_rows)
    (baseline_root / "summary.json").write_text(
        json.dumps(_summary([asdict(row) for row in baseline_rows]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (oracle_root / "summary.json").write_text(
        json.dumps(_summary([asdict(row) for row in oracle_rows]), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
