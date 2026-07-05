from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from benchmarking.headroom_empirical.adapters import AdapterConfig, build_adapter
from benchmarking.headroom_empirical.interface import build_request_execution_plan
from benchmarking.headroom_empirical.nvtx import nvtx_range
from benchmarking.headroom_empirical.runtime_alignment import align_bundle_to_runtime
from benchmarking.headroom_empirical.schema import (
    HeadroomRequest,
    RequestMeasurement,
    WorkloadBundle,
    load_workload_bundle,
)


def _split_token_segments(token_ids: list[int], sep_token_ids: list[int]) -> list[list[int]]:
    if not sep_token_ids:
        return [token_ids]
    segments: list[list[int]] = []
    start = 0
    idx = 0
    sep_len = len(sep_token_ids)
    while idx <= len(token_ids) - sep_len:
        if token_ids[idx : idx + sep_len] == sep_token_ids:
            segments.append(token_ids[start:idx])
            start = idx + sep_len
            idx = start
            continue
        idx += 1
    segments.append(token_ids[start:])
    return segments


def _oracle_kv_transfer_params(adapter, request: HeadroomRequest) -> dict[str, object]:
    metadata = request.metadata or {}
    target_token_ids = metadata.get("prompt_token_ids")
    source_token_ids = metadata.get("oracle_preload_prompt_token_ids")
    if not target_token_ids or not source_token_ids:
        return {}

    sep_token_ids = adapter.tokenize_text(
        getattr(adapter, "blend_special_str", " # # ")
    )
    target_segments = _split_token_segments(list(target_token_ids), sep_token_ids)
    source_segments = _split_token_segments(list(source_token_ids), sep_token_ids)
    if not target_segments or not source_segments:
        return {}

    # The preload prompt replaces the target tail with a sentinel. Do not map that
    # sentinel onto the measured request tail; only reusable source segments are
    # oracle-overridden.
    reusable_source_count = max(0, min(len(source_segments) - 1, len(target_segments) - 1))
    source_segments_for_target: list[list[int] | None] = []
    for idx in range(len(target_segments)):
        if idx < reusable_source_count:
            source_segments_for_target.append(source_segments[idx])
        else:
            source_segments_for_target.append(None)

    return {
        "lmcache.oracle.source_segments_token_ids": json.dumps(source_segments_for_target),
        "lmcache.skip_save": "true",
        **(
            {"lmcache.oracle.timing_path": getattr(adapter, "_timing_path")}
            if getattr(adapter, "_timing_path", None)
            else {}
        ),
    }


def _index_requests(bundle: WorkloadBundle) -> dict[str, HeadroomRequest]:
    return {request.request_id: request for request in bundle.requests}


def _index_objects(bundle: WorkloadBundle) -> dict[str, dict]:
    return {obj.object_id: asdict(obj) for obj in bundle.objects}


def _preload_prompts(
    bundle: WorkloadBundle,
    request: HeadroomRequest,
    object_ids: Iterable[str],
    *,
    system: str,
) -> list[object]:
    oracle_preload_prompt_token_ids = (request.metadata or {}).get(
        "oracle_preload_prompt_token_ids"
    )
    if system.startswith("lmcache_") and oracle_preload_prompt_token_ids:
        requested_ids = list(object_ids)
        if requested_ids and set(requested_ids) == set(request.reusable_object_ids):
            return [{"prompt_token_ids": list(oracle_preload_prompt_token_ids)}]
    object_map = {obj.object_id: obj for obj in bundle.objects}
    prompts: list[object] = []
    for object_id in object_ids:
        obj = object_map[object_id]
        seed_prompt_token_ids = (obj.metadata or {}).get("seed_prompt_token_ids")
        if seed_prompt_token_ids:
            prompts.append({"prompt_token_ids": list(seed_prompt_token_ids)})
        else:
            prompts.append(obj.seed_prompt)
    return prompts


def _request_prompt_payload(request: HeadroomRequest) -> object:
    prompt_token_ids = (request.metadata or {}).get("prompt_token_ids")
    if prompt_token_ids:
        return {"prompt_token_ids": list(prompt_token_ids)}
    return request.prompt


def _prewarm_fcfs(adapter, prompts: list[object]) -> None:
    for prompt in prompts:
        adapter.prewarm([prompt])


def _measure_with_adapter(
    *,
    adapter,
    bundle: WorkloadBundle,
    request: HeadroomRequest,
    preload_object_ids: list[str],
    preload_prompts: list[object],
    history_prompts: list[object] | None,
    warmup_prompts: list[object] | None,
    system: str,
    mode: str,
    bundle_root: Path,
    history_request_ids: list[str] | None = None,
    warmup_full_path: bool = False,
) -> RequestMeasurement:
    request_payload = _request_prompt_payload(request)
    if mode == "oracle0" and system.startswith("lmcache_"):
        kv_transfer_params = _oracle_kv_transfer_params(adapter, request)
        if kv_transfer_params and isinstance(request_payload, dict):
            request_payload = {
                **request_payload,
                "__kv_transfer_params": kv_transfer_params,
            }
    plan = build_request_execution_plan(
        bundle,
        request_id=request.request_id,
        mode=mode,
        history_request_ids=history_request_ids,
        preload_object_ids=preload_object_ids,
    )
    with nvtx_range(f"{system}:{mode}:clear_state"):
        adapter.clear_state()
    if warmup_full_path:
        if history_prompts:
            with nvtx_range(f"{system}:{mode}:warmup_history_replay"):
                _prewarm_fcfs(adapter, history_prompts)
        if preload_prompts:
            with nvtx_range(f"{system}:{mode}:warmup_preload"):
                adapter.prewarm(preload_prompts)
        with nvtx_range(f"{system}:{mode}:warmup_target_request"):
            adapter.measure_request(request_payload)
        with nvtx_range(f"{system}:{mode}:clear_after_full_path_warmup"):
            adapter.clear_state()
    elif warmup_prompts:
        with nvtx_range(f"{system}:{mode}:warmup"):
            adapter.prewarm(warmup_prompts)
        with nvtx_range(f"{system}:{mode}:clear_after_warmup"):
            adapter.clear_state()
    if history_prompts:
        with nvtx_range(f"{system}:{mode}:history_replay"):
            _prewarm_fcfs(adapter, history_prompts)
    if preload_prompts:
        preload_label = "counterfactual_preload" if history_prompts is not None else "preload"
        with nvtx_range(f"{system}:{mode}:{preload_label}"):
            adapter.prewarm(preload_prompts)
        if system.startswith("lmcache_"):
            # LMCache's broad-reuse path can index/store asynchronously.
            # Oracle 0 should not start until preload materialization is actually ready.
            time.sleep(1.0)
    with nvtx_range(f"{system}:{mode}:target_request"):
        metrics = adapter.measure_request(request_payload)
    measurement_metadata = dict(metrics.pop("measurement_metadata", {}) or {})
    prefill_time_ms = (
        float(metrics["prefill_time_ms"])
        if metrics["prefill_time_ms"] is not None
        else None
    )
    if mode == "oracle0" and system.startswith("lmcache_") and prefill_time_ms is not None:
        raw_prefill_time_ms = prefill_time_ms
        materialization_ms = metrics.get("oracle_hbm_materialization_ms")
        if materialization_ms is None:
            materialization_ms = float(metrics.get("lmcache_retrieve_time_ms") or 0.0)
        materialization_ms = float(materialization_ms)
        prefill_time_ms = max(0.0, raw_prefill_time_ms - materialization_ms)
        raw_ttft_ms = (
            float(metrics["ttft_ms"]) if metrics.get("ttft_ms") is not None else None
        )
        adjusted_ttft_ms = (
            max(0.0, raw_ttft_ms - materialization_ms)
            if raw_ttft_ms is not None
            else None
        )
        measurement_metadata.update(
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
            if metrics["decode_time_ms"] is not None
            else None
        ),
        ttft_ms=float(metrics["ttft_ms"]) if metrics["ttft_ms"] is not None else None,
        num_cached_tokens=(
            int(metrics["num_cached_tokens"])
            if metrics["num_cached_tokens"] is not None
            else None
        ),
        prompt_tokens=request.prompt_tokens,
        output_len=request.output_len,
        preload_object_ids=preload_object_ids,
        repair_expected=any(
            object_id.startswith("chunk_approx_") for object_id in preload_object_ids
        ),
        metadata={
            "bundle_root": str(bundle_root),
            "system": system,
            "tokenization_model": adapter.config.model,
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
            **(
                {
                    "lmcache_async_loading_enabled": bool(
                        getattr(adapter, "async_loading_enabled", False)
                    )
                }
                if system.startswith("lmcache_")
                else {}
            ),
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
            **(
                {"history_request_ids": history_request_ids}
                if history_request_ids is not None
                else {}
            ),
        },
    )


def run_single_case(
    *,
    system: str,
    model: str,
    bundle_root: Path,
    request_id: str,
    preload_object_ids: list[str],
    mode: str,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
    warmup_request_ids: list[str] | None = None,
    warmup_full_path: bool = False,
) -> RequestMeasurement:
    bundle = load_workload_bundle(bundle_root)
    adapter_config = AdapterConfig(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    with build_adapter(system, adapter_config) as adapter:
        bundle = align_bundle_to_runtime(bundle, adapter)
        request_index = _index_requests(bundle)
        request = request_index[request_id]
        preload_prompts = _preload_prompts(bundle, request, preload_object_ids, system=system)
        warmup_prompts = [_request_prompt_payload(request_index[item]) for item in (warmup_request_ids or [])]
        return _measure_with_adapter(
            adapter=adapter,
            bundle=bundle,
            request=request,
            preload_object_ids=preload_object_ids,
            preload_prompts=preload_prompts,
            history_prompts=None,
            warmup_prompts=warmup_prompts,
            system=system,
            mode=mode,
            bundle_root=bundle_root,
            warmup_full_path=warmup_full_path,
        )


def run_single_case_with_adapter(
    *,
    adapter,
    bundle: WorkloadBundle,
    system: str,
    bundle_root: Path,
    request_id: str,
    preload_object_ids: list[str],
    mode: str,
) -> RequestMeasurement:
    bundle = align_bundle_to_runtime(bundle, adapter)
    request = _index_requests(bundle)[request_id]
    preload_prompts = _preload_prompts(bundle, request, preload_object_ids, system=system)
    return _measure_with_adapter(
        adapter=adapter,
        bundle=bundle,
        request=request,
        preload_object_ids=preload_object_ids,
        preload_prompts=preload_prompts,
        history_prompts=None,
        warmup_prompts=None,
        system=system,
        mode=mode,
        bundle_root=bundle_root,
    )


def run_replay_case(
    *,
    system: str,
    model: str,
    bundle_root: Path,
    request_id: str,
    history_request_ids: list[str],
    preload_object_ids: list[str],
    mode: str,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
    warmup_request_ids: list[str] | None = None,
    warmup_full_path: bool = False,
) -> RequestMeasurement:
    bundle = load_workload_bundle(bundle_root)
    adapter_config = AdapterConfig(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    with build_adapter(system, adapter_config) as adapter:
        bundle = align_bundle_to_runtime(bundle, adapter)
        request_index = _index_requests(bundle)
        request = request_index[request_id]
        history_prompts = [_request_prompt_payload(request_index[item]) for item in history_request_ids]
        preload_prompts = _preload_prompts(bundle, request, preload_object_ids, system=system)
        warmup_prompts = [_request_prompt_payload(request_index[item]) for item in (warmup_request_ids or [])]
        return _measure_with_adapter(
            adapter=adapter,
            bundle=bundle,
            request=request,
            preload_object_ids=preload_object_ids,
            preload_prompts=preload_prompts,
            history_prompts=history_prompts,
            warmup_prompts=warmup_prompts,
            system=system,
            mode=mode,
            bundle_root=bundle_root,
            history_request_ids=history_request_ids,
            warmup_full_path=warmup_full_path,
        )


def run_replay_case_with_adapter(
    *,
    adapter,
    bundle: WorkloadBundle,
    system: str,
    bundle_root: Path,
    request_id: str,
    history_request_ids: list[str],
    preload_object_ids: list[str],
    mode: str,
) -> RequestMeasurement:
    bundle = align_bundle_to_runtime(bundle, adapter)
    request_index = _index_requests(bundle)
    request = request_index[request_id]
    history_prompts = [_request_prompt_payload(request_index[item]) for item in history_request_ids]
    preload_prompts = _preload_prompts(bundle, request, preload_object_ids, system=system)
    return _measure_with_adapter(
        adapter=adapter,
        bundle=bundle,
        request=request,
        preload_object_ids=preload_object_ids,
        preload_prompts=preload_prompts,
        history_prompts=history_prompts,
        warmup_prompts=None,
        system=system,
        mode=mode,
        bundle_root=bundle_root,
        history_request_ids=history_request_ids,
    )


def write_case_measurement(output_path: Path, measurement: RequestMeasurement) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(measurement), indent=2, sort_keys=True), encoding="utf-8"
    )
