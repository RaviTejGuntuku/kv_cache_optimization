from __future__ import annotations

from dataclasses import replace

from benchmarking.headroom_empirical.schema import (
    HeadroomRequest,
    ReusableObject,
    WorkloadBundle,
)


def _sorted_occurrences(request: HeadroomRequest) -> list[dict]:
    raw_occurrences = list((request.metadata or {}).get("object_occurrences", []))
    return sorted(
        raw_occurrences,
        key=lambda raw: (
            int(raw.get("token_start", 0)),
            int(raw.get("occurrence_index", 0)),
        ),
    )


def _locate_occurrence(prompt: str, text: str, search_start: int) -> tuple[int, int] | None:
    if not text:
        return None
    char_start = prompt.find(text, search_start)
    if char_start < 0:
        char_start = prompt.find(text)
    if char_start < 0:
        return None
    return char_start, char_start + len(text)


def align_bundle_to_runtime(bundle: WorkloadBundle, adapter) -> WorkloadBundle:
    tokenization_model = adapter.config.model
    existing_model = (bundle.metadata or {}).get("runtime_alignment_model")
    if existing_model == tokenization_model:
        return bundle

    aligned_objects: list[ReusableObject] = []
    for obj in bundle.objects:
        seed_prompt_token_ids = (obj.metadata or {}).get("seed_prompt_token_ids")
        runtime_seed_prompt_tokens = (
            len(seed_prompt_token_ids)
            if seed_prompt_token_ids
            else adapter.count_tokens(obj.seed_prompt)
        )
        metadata = dict(obj.metadata or {})
        metadata.update(
            {
                "workload_seed_prompt_tokens": obj.seed_prompt_tokens,
                "runtime_seed_prompt_tokens": runtime_seed_prompt_tokens,
                "runtime_alignment_model": tokenization_model,
            }
        )
        aligned_objects.append(
            replace(
                obj,
                seed_prompt_tokens=runtime_seed_prompt_tokens,
                metadata=metadata,
            )
        )

    aligned_requests: list[HeadroomRequest] = []
    for request in bundle.requests:
        metadata = dict(request.metadata or {})
        raw_occurrences = _sorted_occurrences(request)
        prompt_token_ids = metadata.get("prompt_token_ids")
        if prompt_token_ids:
            runtime_occurrences = []
            for raw in raw_occurrences:
                runtime_occurrences.append(
                    {
                        **raw,
                        "runtime_token_length": max(
                            0,
                            int(raw.get("token_end", 0)) - int(raw.get("token_start", 0)),
                        ),
                    }
                )
            runtime_prompt_tokens = len(prompt_token_ids)
            metadata.update(
                {
                    "workload_prompt_tokens": request.prompt_tokens,
                    "runtime_prompt_tokens": runtime_prompt_tokens,
                    "workload_object_occurrences": raw_occurrences,
                    "object_occurrences": runtime_occurrences,
                    "runtime_alignment_model": tokenization_model,
                    "runtime_alignment_success": True,
                    "runtime_alignment_errors": [],
                }
            )
            aligned_requests.append(
                replace(
                    request,
                    prompt_tokens=runtime_prompt_tokens,
                    metadata=metadata,
                )
            )
            continue
        runtime_occurrences: list[dict] = []
        alignment_errors: list[dict] = []
        search_start = 0
        for raw in raw_occurrences:
            text = str(raw.get("text", ""))
            located = _locate_occurrence(request.prompt, text, search_start)
            if located is None:
                alignment_errors.append(
                    {
                        "object_id": raw.get("object_id"),
                        "occurrence_index": raw.get("occurrence_index"),
                        "reason": "text_not_found_in_prompt",
                    }
                )
                continue
            char_start, char_end = located
            token_start = adapter.count_tokens(request.prompt[:char_start])
            token_end = adapter.count_tokens(request.prompt[:char_end])
            search_start = char_end
            runtime_occurrences.append(
                {
                    **raw,
                    "token_start": token_start,
                    "token_end": token_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "runtime_token_length": max(0, token_end - token_start),
                }
            )

        runtime_prompt_tokens = adapter.count_tokens(request.prompt)
        metadata.update(
            {
                "workload_prompt_tokens": request.prompt_tokens,
                "runtime_prompt_tokens": runtime_prompt_tokens,
                "workload_object_occurrences": raw_occurrences,
                "object_occurrences": runtime_occurrences,
                "runtime_alignment_model": tokenization_model,
                "runtime_alignment_success": not alignment_errors,
                "runtime_alignment_errors": alignment_errors,
            }
        )
        aligned_requests.append(
            replace(
                request,
                prompt_tokens=runtime_prompt_tokens,
                metadata=metadata,
            )
        )

    bundle_metadata = dict(bundle.metadata or {})
    bundle_metadata.update(
        {
            "runtime_alignment_model": tokenization_model,
            "runtime_alignment_success": all(
                bool((request.metadata or {}).get("runtime_alignment_success", True))
                for request in aligned_requests
            ),
        }
    )
    return replace(
        bundle,
        objects=aligned_objects,
        requests=aligned_requests,
        metadata=bundle_metadata,
    )
