from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarking.headroom_empirical.schema import HeadroomRequest, WorkloadBundle


@dataclass(frozen=True)
class ObjectOccurrence:
    object_id: str
    object_type: str
    occurrence_index: int
    token_start: int
    token_end: int
    text: str
    approximate: bool

    @property
    def token_length(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class RequestExecutionPlan:
    request_id: str
    track: str
    mode: str
    history_request_ids: list[str]
    preload_object_ids: list[str]
    reusable_object_ids: list[str]
    ordered_occurrences: list[ObjectOccurrence]
    repair_object_ids: list[str]
    missed_candidate_object_ids: list[str]
    metadata: dict[str, Any]


def _object_map(bundle: WorkloadBundle) -> dict[str, dict[str, Any]]:
    return {
        obj.object_id: {
            "object_type": obj.object_type,
            "canonical_text": obj.canonical_text,
            "metadata": obj.metadata,
        }
        for obj in bundle.objects
    }


def request_occurrences(bundle: WorkloadBundle, request: HeadroomRequest) -> list[ObjectOccurrence]:
    object_map = _object_map(bundle)
    raw_occurrences = list(request.metadata.get("object_occurrences", []))
    occurrences: list[ObjectOccurrence] = []
    for idx, raw in enumerate(raw_occurrences):
        object_id = raw["object_id"]
        object_info = object_map[object_id]
        object_type = str(raw.get("object_type", object_info["object_type"]))
        text = str(raw.get("text", object_info["canonical_text"]))
        occurrences.append(
            ObjectOccurrence(
                object_id=object_id,
                object_type=object_type,
                occurrence_index=int(raw.get("occurrence_index", idx)),
                token_start=int(raw["token_start"]),
                token_end=int(raw["token_end"]),
                text=text,
                approximate=object_type == "approximate",
            )
        )
    return sorted(occurrences, key=lambda occ: (occ.token_start, occ.occurrence_index))


def candidate_object_ids(
    bundle: WorkloadBundle,
    request: HeadroomRequest,
    *,
    limit: int | None = None,
) -> list[str]:
    ordered = request_occurrences(bundle, request)
    deduped_occurrences: list[ObjectOccurrence] = []
    seen: set[str] = set()
    for occurrence in ordered:
        if occurrence.object_id in seen:
            continue
        seen.add(occurrence.object_id)
        deduped_occurrences.append(occurrence)
    if not deduped_occurrences:
        deduped_ids = list(request.reusable_object_ids)
        return deduped_ids[:limit] if limit is not None else deduped_ids
    if limit is None or limit >= len(deduped_occurrences):
        return [occurrence.object_id for occurrence in deduped_occurrences]

    # Cover each object type first so small pilot budgets still exercise
    # prefix, exact non-prefix, and approximate counterfactuals when present.
    prioritized_by_type: dict[str, list[ObjectOccurrence]] = {}
    for occurrence in deduped_occurrences:
        prioritized_by_type.setdefault(occurrence.object_type, []).append(occurrence)

    type_priority = ("approximate", "nonprefix_exact", "prefix_exact")
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    for object_type in type_priority:
        for occurrence in prioritized_by_type.get(object_type, []):
            if occurrence.object_id in selected_set:
                continue
            selected_ids.append(occurrence.object_id)
            selected_set.add(occurrence.object_id)
            break
        if len(selected_ids) >= limit:
            return selected_ids[:limit]

    for occurrence in deduped_occurrences:
        if occurrence.object_id in selected_set:
            continue
        selected_ids.append(occurrence.object_id)
        selected_set.add(occurrence.object_id)
        if len(selected_ids) >= limit:
            break
    return selected_ids[:limit]


def build_request_execution_plan(
    bundle: WorkloadBundle,
    *,
    request_id: str,
    mode: str,
    history_request_ids: list[str] | None = None,
    preload_object_ids: list[str] | None = None,
) -> RequestExecutionPlan:
    request_index = {request.request_id: request for request in bundle.requests}
    request = request_index[request_id]
    history_request_ids = list(history_request_ids or [])
    preload_object_ids = list(preload_object_ids or [])
    ordered = request_occurrences(bundle, request)
    repair_object_ids = [
        occurrence.object_id
        for occurrence in ordered
        if occurrence.object_id in preload_object_ids and occurrence.approximate
    ]
    missed_candidate_ids = [
        object_id
        for object_id in candidate_object_ids(bundle, request)
        if object_id not in preload_object_ids
    ]
    return RequestExecutionPlan(
        request_id=request.request_id,
        track=request.track,
        mode=mode,
        history_request_ids=history_request_ids,
        preload_object_ids=preload_object_ids,
        reusable_object_ids=list(request.reusable_object_ids),
        ordered_occurrences=ordered,
        repair_object_ids=repair_object_ids,
        missed_candidate_object_ids=missed_candidate_ids,
        metadata={
            "prompt_tokens": request.prompt_tokens,
            "output_len": request.output_len,
            "object_occurrence_count": len(ordered),
        },
    )
