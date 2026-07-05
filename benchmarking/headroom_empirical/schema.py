from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReusableObject:
    object_id: str
    track: str
    object_type: str
    seed_prompt: str
    seed_prompt_tokens: int
    source_tier: str
    canonical_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HeadroomRequest:
    request_id: str
    track: str
    prompt: str
    prompt_tokens: int
    output_len: int
    reusable_object_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadBundle:
    bundle_name: str
    track: str
    description: str
    model_hint: str | None
    objects: list[ReusableObject]
    requests: list[HeadroomRequest]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestMeasurement:
    system: str
    mode: str
    track: str
    request_id: str
    wall_time_ms: float
    prefill_time_ms: float | None
    decode_time_ms: float | None
    ttft_ms: float | None
    num_cached_tokens: int | None
    prompt_tokens: int
    output_len: int
    preload_object_ids: list[str]
    repair_expected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_workload_bundle(root: Path, bundle: WorkloadBundle) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_name": bundle.bundle_name,
        "track": bundle.track,
        "description": bundle.description,
        "model_hint": bundle.model_hint,
        "metadata": bundle.metadata,
        "num_objects": len(bundle.objects),
        "num_requests": len(bundle.requests),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_jsonl(root / "objects.jsonl", (asdict(obj) for obj in bundle.objects))
    _write_jsonl(root / "requests.jsonl", (asdict(req) for req in bundle.requests))


def load_workload_bundle(root: Path) -> WorkloadBundle:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    objects = [
        ReusableObject(**json.loads(line))
        for line in (root / "objects.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [
        HeadroomRequest(**json.loads(line))
        for line in (root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return WorkloadBundle(
        bundle_name=manifest["bundle_name"],
        track=manifest["track"],
        description=manifest["description"],
        model_hint=manifest.get("model_hint"),
        objects=objects,
        requests=requests,
        metadata=manifest.get("metadata", {}),
    )


def write_measurements(path: Path, rows: Iterable[RequestMeasurement]) -> None:
    _write_jsonl(path, (asdict(row) for row in rows))
