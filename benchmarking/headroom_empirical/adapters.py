from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchmarking.headroom_empirical.nvtx import nvtx_range


@dataclass(frozen=True)
class AdapterConfig:
    model: str
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 32768
    enforce_eager: bool = True
    max_tokens: int = 1


def _extract_metric(obj: Any, name: str) -> float | int | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return value
    return None


def _extract_request_metrics(output: Any) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    kv_transfer_params = getattr(output, "kv_transfer_params", None) or {}
    extracted = {
        "num_cached_tokens": getattr(output, "num_cached_tokens", None),
        "prefill_time_ms": (
            float(_extract_metric(metrics, "prefill_time")) * 1000.0
            if _extract_metric(metrics, "prefill_time") is not None
            else None
        ),
        "decode_time_ms": (
            float(_extract_metric(metrics, "decode_time")) * 1000.0
            if _extract_metric(metrics, "decode_time") is not None
            else None
        ),
        "ttft_ms": (
            float(_extract_metric(metrics, "first_token_latency")) * 1000.0
            if _extract_metric(metrics, "first_token_latency") is not None
            else None
        ),
        "oracle_hbm_materialization_ms": (
            float(kv_transfer_params["oracle_hbm_materialization_ms"])
            if kv_transfer_params.get("oracle_hbm_materialization_ms") is not None
            else None
        ),
        "oracle_repair_compute_ms": (
            float(kv_transfer_params["oracle_repair_compute_ms"])
            if kv_transfer_params.get("oracle_repair_compute_ms") is not None
            else None
        ),
        "oracle_load_kv_total_ms": (
            float(kv_transfer_params["oracle_load_kv_total_ms"])
            if kv_transfer_params.get("oracle_load_kv_total_ms") is not None
            else None
        ),
        "lmcache_returned_cached_tokens": (
            int(kv_transfer_params["num_lmcache_cached_tokens"])
            if kv_transfer_params.get("num_lmcache_cached_tokens") is not None
            else None
        ),
    }
    lookup_metadata = {
        key: kv_transfer_params.get(key)
        for key in (
            "lmcache_lookup_token_spans",
            "lmcache_lookup_hit_span_count",
            "lmcache_lookup_hit_token_end",
            "lmcache_lookup_hit_tokens",
        )
        if kv_transfer_params.get(key) is not None
    }
    if lookup_metadata:
        extracted["measurement_metadata"] = lookup_metadata
    return extracted


def _metric_key(metric: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    labels = getattr(metric, "labels", {}) or {}
    if isinstance(labels, Mapping):
        label_items = tuple(sorted((str(k), str(v)) for k, v in labels.items()))
    else:
        label_items = tuple()
    return (str(getattr(metric, "name", "")), label_items)


def _snapshot_vllm_metrics(llm: Any) -> dict[tuple[str, tuple[tuple[str, str], ...]], Any]:
    get_metrics = getattr(llm, "get_metrics", None)
    if not callable(get_metrics):
        return {}
    try:
        metrics = get_metrics()
    except Exception:
        return {}
    return {_metric_key(metric): metric for metric in metrics}


def _extract_histogram_delta_ms(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], Any],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], Any],
    metric_name: str,
) -> float | None:
    total_sum_delta = 0.0
    total_count_delta = 0
    matched = False
    for key, post_metric in after.items():
        if key[0] != metric_name:
            continue
        matched = True
        pre_metric = before.get(key)
        post_sum = float(getattr(post_metric, "sum", 0.0) or 0.0)
        post_count = int(getattr(post_metric, "count", 0) or 0)
        pre_sum = float(getattr(pre_metric, "sum", 0.0) or 0.0) if pre_metric is not None else 0.0
        pre_count = int(getattr(pre_metric, "count", 0) or 0) if pre_metric is not None else 0
        total_sum_delta += post_sum - pre_sum
        total_count_delta += post_count - pre_count
    if not matched or total_count_delta <= 0:
        return None
    return (total_sum_delta / total_count_delta) * 1000.0


def _register_lmcache_vllm_model(vllm_model: Any) -> str:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.compute.models.utils import VLLMModelTracker

    VLLMModelTracker.register_model(ENGINE_NAME, vllm_model)
    return type(vllm_model).__name__


class BaseAdapter:
    system_name: str

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self.llm = None
        self._tokenizer = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def clear_state(self) -> None:
        raise NotImplementedError

    def get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.llm is None:
            raise RuntimeError("Tokenizer requested before adapter start().")
        get_tokenizer = getattr(self.llm, "get_tokenizer", None)
        if not callable(get_tokenizer):
            raise RuntimeError("Serving backend does not expose a tokenizer.")
        self._tokenizer = get_tokenizer()
        return self._tokenizer

    def tokenize_text(self, text: str) -> list[int]:
        tokenizer = self.get_tokenizer()
        attempts = [
            lambda: tokenizer.encode(text, add_special_tokens=False),
            lambda: tokenizer.encode(text),
            lambda: tokenizer(text, add_special_tokens=False)["input_ids"],
            lambda: tokenizer(text)["input_ids"],
        ]
        for attempt in attempts:
            try:
                token_ids = attempt()
            except Exception:
                continue
            if token_ids is None:
                continue
            if isinstance(token_ids, list):
                return [int(token_id) for token_id in token_ids]
            if hasattr(token_ids, "tolist"):
                values = token_ids.tolist()
                if isinstance(values, list):
                    return [int(token_id) for token_id in values]
        raise RuntimeError(f"Unable to tokenize text for model {self.config.model}.")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize_text(text))

    def prewarm(self, prompts: list[Any]) -> None:
        if not prompts:
            return
        sampling_params = self._sampling_params()
        with nvtx_range(f"{self.system_name}:prewarm"):
            self.llm.generate(prompts, sampling_params)

    def _snapshot_backend_state(self) -> Any:
        return None

    def _extract_backend_metrics(
        self,
        before: Any,
        after: Any,
    ) -> dict[str, Any]:
        return {}

    def measure_request(self, prompt: Any) -> dict[str, Any]:
        prompt_payload = prompt
        kv_transfer_params = None
        if isinstance(prompt, dict) and "__kv_transfer_params" in prompt:
            prompt_payload = {
                key: value
                for key, value in prompt.items()
                if key != "__kv_transfer_params"
            }
            kv_transfer_params = prompt["__kv_transfer_params"]
        sampling_params = self._sampling_params(kv_transfer_params=kv_transfer_params)
        metrics_before = _snapshot_vllm_metrics(self.llm)
        backend_before = self._snapshot_backend_state()
        with nvtx_range(f"{self.system_name}:measure_request"):
            start = time.perf_counter()
            outputs = self.llm.generate([prompt_payload], sampling_params)
            wall_ms = (time.perf_counter() - start) * 1000.0
        metrics_after = _snapshot_vllm_metrics(self.llm)
        backend_after = self._snapshot_backend_state()
        output = outputs[0]
        metrics = _extract_request_metrics(output)
        metrics["wall_time_ms"] = wall_ms
        if metrics.get("prefill_time_ms") is None:
            metrics["prefill_time_ms"] = _extract_histogram_delta_ms(
                metrics_before, metrics_after, "vllm:request_prefill_time_seconds"
            )
        if metrics.get("decode_time_ms") is None:
            metrics["decode_time_ms"] = _extract_histogram_delta_ms(
                metrics_before, metrics_after, "vllm:request_decode_time_seconds"
            )
        if metrics.get("ttft_ms") is None:
            metrics["ttft_ms"] = _extract_histogram_delta_ms(
                metrics_before, metrics_after, "vllm:time_to_first_token_seconds"
            )
        metrics.update(self._extract_backend_metrics(backend_before, backend_after))
        return metrics

    def measure_batch(self, prompts: list[Any]) -> list[dict[str, Any]]:
        prompt_payloads: list[Any] = []
        kv_transfer_params_by_prompt: list[dict[str, Any] | None] = []
        for prompt in prompts:
            if isinstance(prompt, dict) and "__kv_transfer_params" in prompt:
                prompt_payloads.append(
                    {
                        key: value
                        for key, value in prompt.items()
                        if key != "__kv_transfer_params"
                    }
                )
                kv_transfer_params_by_prompt.append(prompt["__kv_transfer_params"])
            else:
                prompt_payloads.append(prompt)
                kv_transfer_params_by_prompt.append(None)
        if any(params is not None for params in kv_transfer_params_by_prompt):
            sampling_params = [
                self._sampling_params(kv_transfer_params=params)
                for params in kv_transfer_params_by_prompt
            ]
        else:
            sampling_params = self._sampling_params()
        metrics_before = _snapshot_vllm_metrics(self.llm)
        backend_before = self._snapshot_backend_state()
        with nvtx_range(f"{self.system_name}:measure_batch"):
            start = time.perf_counter()
            outputs = self.llm.generate(prompt_payloads, sampling_params)
            wall_ms = (time.perf_counter() - start) * 1000.0
        metrics_after = _snapshot_vllm_metrics(self.llm)
        backend_after = self._snapshot_backend_state()
        shared_backend_metrics = self._extract_backend_metrics(backend_before, backend_after)
        timing_records = (
            (
                shared_backend_metrics.get("measurement_metadata", {}) or {}
            ).get("lmcache_headroom_timing_records")
            or []
        )
        per_request_metrics: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            metrics = _extract_request_metrics(output)
            metrics["wall_time_ms"] = wall_ms
            if idx < len(timing_records):
                record = timing_records[idx]
                metrics["oracle_hbm_materialization_ms"] = float(
                    record.get("oracle_hbm_materialization_ms", 0.0) or 0.0
                )
                metrics["oracle_repair_compute_ms"] = float(
                    record.get("oracle_repair_compute_ms", 0.0) or 0.0
                )
                metrics["oracle_load_kv_total_ms"] = float(
                    record.get("oracle_load_kv_total_ms", 0.0) or 0.0
                )
                metrics["measurement_metadata"] = {
                    "lmcache_headroom_timing_records": [record]
                }
            per_request_metrics.append(metrics)

        if per_request_metrics:
            prefill_delta_ms = _extract_histogram_delta_ms(
                metrics_before,
                metrics_after,
                "vllm:request_prefill_time_seconds",
            )
            decode_delta_ms = _extract_histogram_delta_ms(
                metrics_before,
                metrics_after,
                "vllm:request_decode_time_seconds",
            )
            ttft_delta_ms = _extract_histogram_delta_ms(
                metrics_before,
                metrics_after,
                "vllm:time_to_first_token_seconds",
            )
            for metrics in per_request_metrics:
                if metrics.get("prefill_time_ms") is None:
                    metrics["prefill_time_ms"] = prefill_delta_ms
                if metrics.get("decode_time_ms") is None:
                    metrics["decode_time_ms"] = decode_delta_ms
                if metrics.get("ttft_ms") is None:
                    metrics["ttft_ms"] = ttft_delta_ms
                existing_metadata = dict(metrics.pop("measurement_metadata", {}) or {})
                per_request_oracle_metrics = {
                    key: metrics.pop(key)
                    for key in (
                        "oracle_hbm_materialization_ms",
                        "oracle_repair_compute_ms",
                        "oracle_load_kv_total_ms",
                    )
                    if key in metrics
                }
                metrics.update(shared_backend_metrics)
                shared_metadata = dict(metrics.pop("measurement_metadata", {}) or {})
                metrics.update(per_request_oracle_metrics)
                metrics["measurement_metadata"] = {
                    **shared_metadata,
                    **existing_metadata,
                }
        return per_request_metrics

    def _post_start(self) -> None:
        return

    @staticmethod
    def _sampling_params(kv_transfer_params: dict[str, Any] | None = None):
        from vllm import SamplingParams

        extra_args = (
            {"kv_transfer_params": kv_transfer_params}
            if kv_transfer_params is not None
            else None
        )
        try:
            return SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                extra_args=extra_args,
            )
        except TypeError:
            params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
            if extra_args is not None:
                setattr(params, "extra_args", extra_args)
            return params


class VllmApcAdapter(BaseAdapter):
    system_name = "vllm_apc"

    def start(self) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=self.config.model,
            enable_prefix_caching=True,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enforce_eager=self.config.enforce_eager,
            disable_log_stats=False,
        )
        self._post_start()

    def stop(self) -> None:
        self.llm = None

    def clear_state(self) -> None:
        if self.llm is not None:
            self.llm.reset_prefix_cache()


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class LMCacheExactAdapter(BaseAdapter):
    system_name = "lmcache_exact"

    def __init__(self, config: AdapterConfig, *, local_cpu_gb: float = 5.0) -> None:
        super().__init__(config)
        self.local_cpu_gb = local_cpu_gb
        self._env_ctx = None
        self.async_loading_enabled = True
        self._timing_path = "/tmp/lmcache_headroom_timings.jsonl"

    def start(self) -> None:
        from vllm import LLM
        from vllm.config import KVTransferConfig

        self._env_ctx = _temporary_env(
            {
                "LMCACHE_CHUNK_SIZE": "256",
                "LMCACHE_ENABLE_ASYNC_LOADING": "True",
                "LMCACHE_LOCAL_CPU": "True",
                "LMCACHE_MAX_LOCAL_CPU_SIZE": str(self.local_cpu_gb),
                "LMCACHE_HEADROOM_TIMING_PATH": self._timing_path,
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._timing_path)
        self._env_ctx.__enter__()
        self.llm = LLM(
            model=self.config.model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1Dynamic",
                kv_role="kv_both",
                kv_connector_module_path="lmcache.integration.vllm.lmcache_connector_v1",
            ),
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=self.config.enforce_eager,
            disable_log_stats=False,
        )
        self._post_start()

    def stop(self) -> None:
        try:
            if self.llm is not None:
                from lmcache.integration.vllm.utils import ENGINE_NAME
                from lmcache.v1.cache_engine import LMCacheEngineBuilder

                LMCacheEngineBuilder.destroy(ENGINE_NAME)
        finally:
            self.llm = None
            if self._env_ctx is not None:
                self._env_ctx.__exit__(None, None, None)
                self._env_ctx = None

    def clear_state(self) -> None:
        if self.llm is not None:
            try:
                self.llm.reset_prefix_cache(reset_connector=True)
            except TypeError:
                try:
                    self.llm.reset_prefix_cache()
                except Exception:
                    return

    def _snapshot_backend_state(self) -> Any:
        return _snapshot_lmcache_state()

    def _extract_backend_metrics(
        self,
        before: Any,
        after: Any,
    ) -> dict[str, Any]:
        return _extract_lmcache_delta(before, after)

    def _post_start(self) -> None:
        if self.llm is None:
            return
        self.llm.llm_engine.apply_model(_register_lmcache_vllm_model)


class LMCacheCacheBlendAdapter(BaseAdapter):
    system_name = "lmcache_cacheblend"

    def __init__(
        self,
        config: AdapterConfig,
        *,
        local_cpu_gb: float = 5.0,
        blend_special_str: str = " # # ",
    ) -> None:
        super().__init__(config)
        self.local_cpu_gb = local_cpu_gb
        self.blend_special_str = blend_special_str
        self._env_ctx = None
        self.async_loading_enabled = True
        self._timing_path = "/tmp/lmcache_headroom_timings.jsonl"

    def start(self) -> None:
        from vllm import LLM
        from vllm.config import KVTransferConfig

        self._env_ctx = _temporary_env(
            {
                "LMCACHE_CHUNK_SIZE": "256",
                "LMCACHE_ENABLE_ASYNC_LOADING": "True",
                "LMCACHE_ENABLE_BLENDING": "True",
                "LMCACHE_BLEND_SPECIAL_STR": self.blend_special_str,
                "LMCACHE_USE_LAYERWISE": "True",
                "LMCACHE_BLEND_CHECK_LAYERS": "1",
                "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
                "LMCACHE_LOCAL_CPU": "True",
                "LMCACHE_MAX_LOCAL_CPU_SIZE": str(self.local_cpu_gb),
                "LMCACHE_HEADROOM_TIMING_PATH": self._timing_path,
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._timing_path)
        self._env_ctx.__enter__()
        self.llm = LLM(
            model=self.config.model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",
            ),
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=True,
            disable_log_stats=False,
        )
        self._post_start()

    def stop(self) -> None:
        try:
            if self.llm is not None:
                from lmcache.integration.vllm.utils import ENGINE_NAME
                from lmcache.v1.cache_engine import LMCacheEngineBuilder

                LMCacheEngineBuilder.destroy(ENGINE_NAME)
        finally:
            self.llm = None
            if self._env_ctx is not None:
                self._env_ctx.__exit__(None, None, None)
                self._env_ctx = None

    def clear_state(self) -> None:
        if self.llm is not None:
            try:
                self.llm.reset_prefix_cache(reset_connector=True)
            except TypeError:
                try:
                    self.llm.reset_prefix_cache()
                except Exception:
                    return

    def _snapshot_backend_state(self) -> Any:
        return _snapshot_lmcache_state()

    def _extract_backend_metrics(
        self,
        before: Any,
        after: Any,
    ) -> dict[str, Any]:
        return _extract_lmcache_delta(before, after)

    def _post_start(self) -> None:
        if self.llm is None:
            return
        self.llm.llm_engine.apply_model(_register_lmcache_vllm_model)


def _snapshot_lmcache_state() -> dict[str, Any]:
    try:
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.observability import LMCStatsMonitor
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
    except Exception:
        return {}

    timing_records: list[dict[str, Any]] = []
    timing_path = os.getenv("LMCACHE_HEADROOM_TIMING_PATH")
    if timing_path and os.path.exists(timing_path):
        with contextlib.suppress(Exception):
            with open(timing_path, "r", encoding="utf-8") as timing_file:
                for line in timing_file:
                    line = line.strip()
                    if line:
                        timing_records.append(json.loads(line))
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    monitor = LMCStatsMonitor.GetOrCreate()
    if engine is None or monitor is None:
        return {"headroom_timing_records": timing_records}

    retrieve_requests = getattr(monitor, "retrieve_requests", {}) or {}
    lookup_requests = getattr(monitor, "lookup_requests", {}) or {}
    return {
        "engine_frozen": bool(getattr(engine, "is_frozen", lambda: False)()),
        "hot_cache_enabled": bool(
            getattr(engine, "is_hot_cache_enabled", lambda: False)()
        ),
        "interval_retrieve_requests": int(
            getattr(monitor, "interval_retrieve_requests", 0) or 0
        ),
        "interval_requested_tokens": int(
            getattr(monitor, "interval_requested_tokens", 0) or 0
        ),
        "interval_hit_tokens": int(getattr(monitor, "interval_hit_tokens", 0) or 0),
        "interval_lookup_requests": int(
            getattr(monitor, "interval_lookup_requests", 0) or 0
        ),
        "interval_lookup_tokens": int(
            getattr(monitor, "interval_lookup_tokens", 0) or 0
        ),
        "interval_lookup_hits": int(
            getattr(monitor, "interval_lookup_hits", 0) or 0
        ),
        "interval_remote_read_requests": int(
            getattr(monitor, "interval_remote_read_requests", 0) or 0
        ),
        "interval_remote_read_bytes": int(
            getattr(monitor, "interval_remote_read_bytes", 0) or 0
        ),
        "retrieve_request_ids": tuple(sorted(retrieve_requests.keys())),
        "lookup_request_ids": tuple(sorted(lookup_requests.keys())),
        "headroom_timing_records": timing_records,
        "retrieve_requests": {
            int(request_id): {
                "num_tokens": int(getattr(stats, "num_tokens", 0) or 0),
                "local_hit_tokens": int(getattr(stats, "local_hit_tokens", 0) or 0),
                "remote_hit_tokens": int(getattr(stats, "remote_hit_tokens", 0) or 0),
                "time_to_retrieve_ms": float(stats.time_to_retrieve() * 1000.0),
                "process_tokens_time_ms": float(
                    getattr(stats, "process_tokens_time", 0.0) * 1000.0
                ),
                "broadcast_time_ms": float(
                    getattr(stats, "broadcast_time", 0.0) * 1000.0
                ),
                "to_gpu_time_ms": float(getattr(stats, "to_gpu_time", 0.0) * 1000.0),
            }
            for request_id, stats in retrieve_requests.items()
        },
        "lookup_requests": {
            int(request_id): {
                "num_tokens": int(getattr(stats, "num_tokens", 0) or 0),
                "hit_tokens": int(getattr(stats, "hit_tokens", 0) or 0),
                "time_to_lookup_ms": float(stats.time_to_lookup() * 1000.0),
            }
            for request_id, stats in lookup_requests.items()
        },
    }


def _extract_lmcache_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    before = before or {}
    after = after or {}
    before_retrieve = before.get("retrieve_requests", {}) or {}
    after_retrieve = after.get("retrieve_requests", {}) or {}
    new_retrieve_ids = sorted(set(after_retrieve) - set(before_retrieve))
    before_lookup = before.get("lookup_requests", {}) or {}
    after_lookup = after.get("lookup_requests", {}) or {}
    new_lookup_ids = sorted(set(after_lookup) - set(before_lookup))
    before_timing_records = before.get("headroom_timing_records", []) or []
    after_timing_records = after.get("headroom_timing_records", []) or []
    new_timing_records = after_timing_records[len(before_timing_records) :]

    retrieve_time_ms = sum(
        float(after_retrieve[request_id]["time_to_retrieve_ms"])
        for request_id in new_retrieve_ids
    )
    retrieve_requested_tokens = sum(
        int(after_retrieve[request_id]["num_tokens"]) for request_id in new_retrieve_ids
    )
    retrieved_tokens = sum(
        int(after_retrieve[request_id]["local_hit_tokens"])
        + int(after_retrieve[request_id]["remote_hit_tokens"])
        for request_id in new_retrieve_ids
    )
    retrieve_process_tokens_ms = sum(
        float(after_retrieve[request_id]["process_tokens_time_ms"])
        for request_id in new_retrieve_ids
    )
    retrieve_broadcast_ms = sum(
        float(after_retrieve[request_id]["broadcast_time_ms"])
        for request_id in new_retrieve_ids
    )
    retrieve_to_gpu_ms = sum(
        float(after_retrieve[request_id]["to_gpu_time_ms"])
        for request_id in new_retrieve_ids
    )
    lookup_time_ms = sum(
        float(after_lookup[request_id]["time_to_lookup_ms"])
        for request_id in new_lookup_ids
    )
    lookup_tokens = sum(
        int(after_lookup[request_id]["num_tokens"]) for request_id in new_lookup_ids
    )
    lookup_hit_tokens = sum(
        int(after_lookup[request_id]["hit_tokens"]) for request_id in new_lookup_ids
    )

    metrics = {
        "lmcache_retrieve_requests": len(new_retrieve_ids),
        "lmcache_requested_tokens": retrieve_requested_tokens,
        "lmcache_retrieved_tokens": retrieved_tokens,
        "lmcache_retrieve_time_ms": retrieve_time_ms,
        "lmcache_retrieve_process_tokens_ms": retrieve_process_tokens_ms,
        "lmcache_retrieve_broadcast_ms": retrieve_broadcast_ms,
        "lmcache_retrieve_to_gpu_ms": retrieve_to_gpu_ms,
        "lmcache_lookup_requests": len(new_lookup_ids),
        "lmcache_lookup_tokens": lookup_tokens,
        "lmcache_lookup_hit_tokens": lookup_hit_tokens,
        "lmcache_lookup_time_ms": lookup_time_ms,
        "lmcache_remote_read_requests_delta": int(
            after.get("interval_remote_read_requests", 0)
        )
        - int(before.get("interval_remote_read_requests", 0)),
        "lmcache_remote_read_bytes_delta": int(
            after.get("interval_remote_read_bytes", 0)
        )
        - int(before.get("interval_remote_read_bytes", 0)),
        "measurement_metadata": {
            "lmcache_engine_frozen": after.get("engine_frozen"),
            "lmcache_hot_cache_enabled": after.get("hot_cache_enabled"),
            "lmcache_new_retrieve_request_ids": new_retrieve_ids,
            "lmcache_new_lookup_request_ids": new_lookup_ids,
        },
    }
    if new_timing_records:
        metrics.update(
            {
                "oracle_hbm_materialization_ms": sum(
                    float(record.get("oracle_hbm_materialization_ms", 0.0) or 0.0)
                    for record in new_timing_records
                ),
                "oracle_repair_compute_ms": sum(
                    float(record.get("oracle_repair_compute_ms", 0.0) or 0.0)
                    for record in new_timing_records
                ),
                "oracle_load_kv_total_ms": sum(
                    float(record.get("oracle_load_kv_total_ms", 0.0) or 0.0)
                    for record in new_timing_records
                ),
            }
        )
        metrics["measurement_metadata"]["lmcache_headroom_timing_records"] = (
            new_timing_records
        )
    return metrics


def build_adapter(system: str, config: AdapterConfig) -> BaseAdapter:
    if system == "vllm_apc":
        return VllmApcAdapter(config)
    if system == "lmcache_exact":
        return LMCacheExactAdapter(config)
    if system == "lmcache_cacheblend":
        return LMCacheCacheBlendAdapter(config)
    raise ValueError(f"Unknown empirical headroom system: {system}")
