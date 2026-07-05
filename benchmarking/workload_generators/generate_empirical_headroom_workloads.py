#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.headroom_empirical.schema import (
    HeadroomRequest,
    ReusableObject,
    WorkloadBundle,
    write_workload_bundle,
)


COMMON_VOCAB = (
    "the",
    "analysis",
    "context",
    "memory",
    "request",
    "shared",
    "prefix",
    "cache",
    "branch",
    "token",
    "sequence",
    "inference",
    "trace",
    "reuse",
    "scheduler",
    "radix",
    "chunk",
    "document",
    "retrieval",
    "evidence",
)


def _build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _encode_prompt(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def _encode_segment(tokenizer, text: str) -> list[int]:
    token_ids = _encode_prompt(tokenizer, text)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and token_ids and token_ids[0] == bos_token_id:
        token_ids = token_ids[1:]
    return token_ids


def render_tokens(count: int, *, offset: int = 0) -> str:
    return " ".join(COMMON_VOCAB[(offset + i) % len(COMMON_VOCAB)] for i in range(count))


def perturb_text(text: str, *, step: int) -> str:
    words = text.split()
    if not words:
        return text
    mutated = list(words)
    for idx in range(0, len(mutated), max(7, step)):
        mutated[idx] = mutated[idx] + " revised"
    return " ".join(mutated)


def perturb_token_ids_preserve_length(token_ids: list[int], *, step: int) -> list[int]:
    """Create a same-length approximate token sequence for CacheBlend repair."""
    if not token_ids:
        return token_ids
    mutated = list(token_ids)
    stride = max(7, step)
    for idx in range(0, len(mutated), stride):
        mutated[idx] = token_ids[(idx + 5) % len(token_ids)]
    return mutated


def slugify(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return lowered or "untitled"


def build_prefix_bundle(
    *,
    bundle_name: str,
    num_groups: int,
    prompts_per_group: int,
    prefix_tokens: int,
    suffix_tokens: int,
    output_len: int,
) -> WorkloadBundle:
    objects: list[ReusableObject] = []
    grouped_requests: list[list[HeadroomRequest]] = []
    for group_idx in range(num_groups):
        prefix_text = (
            f"System directive for group {group_idx}. "
            f"Retain the shared prefix exactly.\n\n"
            + render_tokens(prefix_tokens, offset=group_idx)
        )
        object_id = f"prefix_group_{group_idx:03d}"
        objects.append(
            ReusableObject(
                object_id=object_id,
                track="prefix",
                object_type="prefix_exact",
                seed_prompt=prefix_text,
                seed_prompt_tokens=prefix_tokens,
                source_tier="HBM_seed",
                canonical_text=prefix_text,
                metadata={
                    "group_id": group_idx,
                    "object_size_tokens": prefix_tokens,
                },
            )
        )
        group_requests: list[HeadroomRequest] = []
        for branch_idx in range(prompts_per_group):
            suffix = (
                f"\n\nBranch question {branch_idx} for group {group_idx}.\n\n"
                + render_tokens(suffix_tokens, offset=1000 + group_idx * 13 + branch_idx)
            )
            prompt = prefix_text + suffix
            group_requests.append(
                HeadroomRequest(
                    request_id=f"prefix_req_{group_idx:03d}_{branch_idx:03d}",
                    track="prefix",
                    prompt=prompt,
                    prompt_tokens=prefix_tokens + suffix_tokens,
                    output_len=output_len,
                    reusable_object_ids=[object_id],
                    metadata={
                        "group_id": group_idx,
                        "branch_id": branch_idx,
                        "object_occurrences": [
                            {
                                "object_id": object_id,
                                "object_type": "prefix_exact",
                                "occurrence_index": 0,
                                "token_start": 0,
                                "token_end": prefix_tokens,
                                "text": prefix_text,
                            }
                        ],
                    },
                )
            )
        grouped_requests.append(group_requests)
    requests = [
        request
        for branch_idx in range(prompts_per_group)
        for group_idx in range(num_groups)
        for request in grouped_requests[group_idx][branch_idx : branch_idx + 1]
    ]
    return WorkloadBundle(
        bundle_name=bundle_name,
        track="prefix",
        description="Exact-prefix empirical headroom bundle.",
        model_hint=None,
        objects=objects,
        requests=requests,
        metadata={
            "num_groups": num_groups,
            "prompts_per_group": prompts_per_group,
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "output_len": output_len,
        },
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_sharegpt_reordered_prefix_bundle(
    *,
    bundle_name: str,
    source_path: Path,
    manifest_path: Path,
    max_groups: int,
    max_requests_per_group: int,
) -> WorkloadBundle:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(source_path)

    group_stats = {
        item["group_id"]: item
        for item in manifest.get("groups", [])
        if int(item.get("selected_count", 0)) >= max_requests_per_group
    }
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {}) or {}
        group_id = metadata.get("group_id")
        if group_id in group_stats:
            grouped_rows[group_id].append(row)

    selected_groups = sorted(
        grouped_rows,
        key=lambda gid: (
            -int(group_stats[gid].get("shared_prefix_chars", 0)),
            -int(group_stats[gid].get("selected_count", 0)),
            gid,
        ),
    )[:max_groups]

    objects: list[ReusableObject] = []
    grouped_requests: list[list[HeadroomRequest]] = []
    for group_id in selected_groups:
        stat = group_stats[group_id]
        group_rows = grouped_rows[group_id][:max_requests_per_group]
        if len(group_rows) < max_requests_per_group:
            continue
        first_prompt = str(group_rows[0]["conversations"][0]["content"])
        shared_prefix_chars = int(stat["shared_prefix_chars"])
        prefix_text = first_prompt[:shared_prefix_chars]
        prefix_tokens = min(_word_count(prefix_text), int(group_rows[0]["prompt_len"]) - 1)
        object_id = f"sharegpt_prefix_{group_id}"
        objects.append(
            ReusableObject(
                object_id=object_id,
                track="prefix",
                object_type="prefix_exact",
                seed_prompt=prefix_text,
                seed_prompt_tokens=prefix_tokens,
                source_tier="HBM_seed",
                canonical_text=prefix_text,
                metadata={
                    "group_id": group_id,
                    "group_key": stat.get("group_key"),
                    "shared_prefix_chars": shared_prefix_chars,
                    "object_size_tokens": prefix_tokens,
                    "source_dataset": "sharegpt_prefix_competition",
                },
            )
        )
        reqs: list[HeadroomRequest] = []
        for branch_idx, row in enumerate(group_rows):
            prompt = str(row["conversations"][0]["content"])
            prompt_tokens = int(row["prompt_len"])
            output_len = max(1, int(row.get("output_len", 1)))
            reqs.append(
                HeadroomRequest(
                    request_id=f"{group_id}__req_{branch_idx:02d}",
                    track="prefix",
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    output_len=output_len,
                    reusable_object_ids=[object_id],
                    metadata={
                        "group_id": group_id,
                        "branch_id": branch_idx,
                        "group_key": stat.get("group_key"),
                        "source_dataset": "sharegpt_prefix_competition",
                        "object_occurrences": [
                            {
                                "object_id": object_id,
                                "object_type": "prefix_exact",
                                "occurrence_index": 0,
                                "token_start": 0,
                                "token_end": prefix_tokens,
                                "text": prefix_text,
                            }
                        ],
                    },
                )
            )
        grouped_requests.append(reqs)

    requests = [
        request
        for round_idx in range(max_requests_per_group)
        for reqs in grouped_requests
        for request in reqs[round_idx : round_idx + 1]
    ]

    return WorkloadBundle(
        bundle_name=bundle_name,
        track="prefix",
        description=(
            "Natural ShareGPT-derived shared-prefix workload reordered round-robin across "
            "prefix groups to create realistic but cache-lousy reuse."
        ),
        model_hint=None,
        objects=objects,
        requests=requests,
        metadata={
            "source_dataset": "sharegpt_prefix_competition",
            "source_path": str(source_path),
            "manifest_path": str(manifest_path),
            "max_groups": max_groups,
            "max_requests_per_group": max_requests_per_group,
            "ordering": "round_robin_by_group",
        },
    )


def _load_hotpot_examples(examples_count: int) -> list[dict]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split=f"validation[:{examples_count}]",
        )
        return list(dataset)
    except Exception:
        params = urlencode(
            {
                "dataset": "hotpotqa/hotpot_qa",
                "config": "distractor",
                "split": "validation",
                "offset": 0,
                "length": examples_count,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        try:
            with urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                "Real-RAG workload generation requires HuggingFace datasets access."
            ) from exc
        return [item.get("row", item) for item in payload["rows"]]


def subset_prefix_group_heads(
    *,
    source_bundle: WorkloadBundle,
    bundle_name: str,
    groups: int,
) -> WorkloadBundle:
    selected_requests = [
        request
        for request in source_bundle.requests
        if request.metadata.get("branch_id") == 0
    ][:groups]
    selected_group_ids = {request.metadata["group_id"] for request in selected_requests}
    selected_objects = [
        obj
        for obj in source_bundle.objects
        if obj.metadata.get("group_id") in selected_group_ids
    ]
    return WorkloadBundle(
        bundle_name=bundle_name,
        track=source_bundle.track,
        description=(
            "Miss-heavy prefix pilot bundle: one cold request per prefix group to validate "
            "Oracle 0 and marginal-counterfactual measurement."
        ),
        model_hint=source_bundle.model_hint,
        objects=selected_objects,
        requests=selected_requests,
        metadata={
            **source_bundle.metadata,
            "pilot_kind": "prefix_group_heads",
            "selected_group_count": groups,
        },
    )


def build_mixed_bundle(
    *,
    bundle_name: str,
    tokenizer_model: str,
    requests_count: int,
    chunk_library_size: int,
    chunks_per_request: int,
    prefix_tokens: int,
    chunk_tokens: int,
    tail_tokens: int,
    output_len: int,
) -> WorkloadBundle:
    tokenizer = _build_tokenizer(tokenizer_model)
    blend_special_str = " # # "
    blend_special_token_ids = _encode_segment(tokenizer, blend_special_str)
    prefix_text = (
        "System directive for mixed reusable-object study. "
        "Chunk boundaries use a separator marker between reusable objects.\n\n"
        + render_tokens(prefix_tokens, offset=77)
    )
    prefix_token_ids = _encode_prompt(tokenizer, prefix_text)
    objects: list[ReusableObject] = [
        ReusableObject(
            object_id="broad_prefix_shared",
            track="broad",
            object_type="prefix_exact",
            seed_prompt=prefix_text,
            seed_prompt_tokens=len(prefix_token_ids),
            source_tier="HBM_seed",
            canonical_text=prefix_text,
            metadata={
                "shared": True,
                "object_size_tokens": len(prefix_token_ids),
                "seed_prompt_token_ids": prefix_token_ids,
            },
        )
    ]

    exact_chunks: list[dict[str, object]] = []
    approx_chunks: list[dict[str, object]] = []
    for chunk_idx in range(chunk_library_size):
        canonical = (
            f"Chunk {chunk_idx}. "
            + render_tokens(chunk_tokens, offset=200 + chunk_idx * 17)
        )
        exact_id = f"chunk_exact_{chunk_idx:03d}"
        approx_id = f"chunk_approx_{chunk_idx:03d}"
        canonical_token_ids = _encode_segment(tokenizer, canonical)
        exact_chunks.append(
            {
                "text": canonical,
                "token_ids": canonical_token_ids,
            }
        )
        approx_token_ids = perturb_token_ids_preserve_length(
            canonical_token_ids, step=11 + (chunk_idx % 3)
        )
        approx = tokenizer.decode(approx_token_ids)
        approx_chunks.append(
            {
                "text": approx,
                "token_ids": approx_token_ids,
            }
        )
        seed_suffix = _encode_segment(tokenizer, f"Summarize chunk {chunk_idx}.")
        seed_prompt_token_ids = (
            prefix_token_ids
            + blend_special_token_ids
            + canonical_token_ids
            + blend_special_token_ids
            + seed_suffix
        )
        objects.append(
            ReusableObject(
                object_id=exact_id,
                track="broad",
                object_type="nonprefix_exact",
                seed_prompt=f"{prefix_text}\n\n{blend_special_str}\n{canonical}\n\n{blend_special_str}\nSummarize chunk {chunk_idx}.",
                seed_prompt_tokens=len(seed_prompt_token_ids),
                source_tier="HBM_seed",
                canonical_text=canonical,
                metadata={
                    "chunk_index": chunk_idx,
                    "object_size_tokens": len(canonical_token_ids),
                    "seed_prompt_token_ids": seed_prompt_token_ids,
                    "canonical_token_ids": canonical_token_ids,
                },
            )
        )
        objects.append(
            ReusableObject(
                object_id=approx_id,
                track="broad",
                object_type="approximate",
                seed_prompt=f"{prefix_text}\n\n{blend_special_str}\n{canonical}\n\n{blend_special_str}\nSummarize chunk {chunk_idx}.",
                seed_prompt_tokens=len(seed_prompt_token_ids),
                source_tier="HBM_seed",
                canonical_text=approx,
                metadata={
                    "chunk_index": chunk_idx,
                    "seed_object_id": exact_id,
                    "object_size_tokens": len(approx_token_ids),
                    "seed_prompt_token_ids": seed_prompt_token_ids,
                    "canonical_token_ids": approx_token_ids,
                },
            )
        )

    requests: list[HeadroomRequest] = []
    for req_idx in range(requests_count):
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        oracle_preload_token_ids = list(prefix_token_ids)
        object_occurrences = [
            {
                "object_id": "broad_prefix_shared",
                "object_type": "prefix_exact",
                "occurrence_index": 0,
                "token_start": 0,
                "token_end": len(prefix_token_ids),
                "text": prefix_text,
            }
        ]
        cursor = len(prefix_token_ids)
        prompt_token_ids = list(prefix_token_ids)
        for slot in range(chunks_per_request):
            lib_idx = (req_idx * 5 + slot * 3) % chunk_library_size
            if slot == 0:
                object_id = f"chunk_exact_{lib_idx:03d}"
                chunk_text = str(exact_chunks[lib_idx]["text"])
                chunk_token_ids = list(exact_chunks[lib_idx]["token_ids"])
            elif slot % 3 == 0:
                object_id = f"chunk_approx_{lib_idx:03d}"
                chunk_text = str(approx_chunks[lib_idx]["text"])
                chunk_token_ids = list(approx_chunks[lib_idx]["token_ids"])
            else:
                object_id = f"chunk_exact_{lib_idx:03d}"
                chunk_text = str(exact_chunks[lib_idx]["text"])
                chunk_token_ids = list(exact_chunks[lib_idx]["token_ids"])
            chunk_ids.append(object_id)
            chunk_texts.append(chunk_text)
            prompt_token_ids += blend_special_token_ids + chunk_token_ids
            if object_id.startswith("chunk_approx_"):
                oracle_chunk_token_ids = list(exact_chunks[lib_idx]["token_ids"])
            else:
                oracle_chunk_token_ids = list(chunk_token_ids)
            oracle_preload_token_ids += blend_special_token_ids + oracle_chunk_token_ids
            object_token_start = cursor + len(blend_special_token_ids)
            object_occurrences.append(
                {
                    "object_id": object_id,
                    "object_type": "approximate" if object_id.startswith("chunk_approx_") else "nonprefix_exact",
                    "occurrence_index": slot + 1,
                    "token_start": object_token_start,
                    "token_end": object_token_start + len(chunk_token_ids),
                    "text": chunk_text,
                }
            )
            cursor += len(blend_special_token_ids) + len(chunk_token_ids)

        prompt = prefix_text + "\n\n" + f"\n\n{blend_special_str}\n".join(chunk_texts)
        prompt += (
            f"\n\n{blend_special_str}\nQuestion: explain the interaction among the retrieved chunks.\n"
            + render_tokens(tail_tokens, offset=5000 + req_idx)
        )
        tail_suffix_text = (
            "Question: explain the interaction among the retrieved chunks.\n"
            + render_tokens(tail_tokens, offset=5000 + req_idx)
        )
        tail_token_ids = _encode_segment(
            tokenizer,
            tail_suffix_text,
        )
        prompt_token_ids += blend_special_token_ids + tail_token_ids
        oracle_preload_token_ids += blend_special_token_ids + _encode_segment(
            tokenizer, "Oracle preload sentinel."
        )
        requests.append(
            HeadroomRequest(
                request_id=f"broad_req_{req_idx:04d}",
                track="broad",
                prompt=prompt,
                prompt_tokens=len(prompt_token_ids),
                output_len=output_len,
                reusable_object_ids=["broad_prefix_shared", *chunk_ids],
                metadata={
                    "request_index": req_idx,
                    "chunk_object_ids": chunk_ids,
                    "object_occurrences": object_occurrences,
                    "oracle_preload_prompt_token_ids": oracle_preload_token_ids,
                    "tail_token_start": cursor,
                    "tail_token_end": cursor + len(blend_special_token_ids) + len(tail_token_ids),
                    "prompt_token_ids": prompt_token_ids,
                },
            )
        )

    return WorkloadBundle(
        bundle_name=bundle_name,
        track="broad",
        description="Mixed prefix / exact non-prefix / approximate reusable-object bundle.",
        model_hint=None,
        objects=objects,
        requests=requests,
        metadata={
            "tokenizer_model": tokenizer_model,
            "requests_count": requests_count,
            "chunk_library_size": chunk_library_size,
            "chunks_per_request": chunks_per_request,
            "prefix_tokens": prefix_tokens,
            "chunk_tokens": chunk_tokens,
            "tail_tokens": tail_tokens,
            "output_len": output_len,
        },
    )


def build_rag_bundle(
    *,
    bundle_name: str,
    tokenizer_model: str,
    examples_count: int,
    top_k: int,
    chunk_token_budget: int,
    align_requests_to: int | None,
    output_len: int,
) -> WorkloadBundle:
    tokenizer = _build_tokenizer(tokenizer_model)
    blend_special_str = " # # "
    blend_special_token_ids = _encode_segment(tokenizer, blend_special_str)
    dataset = _load_hotpot_examples(examples_count)
    prefix_text = (
        "You are a retrieval-grounded assistant. Use only the retrieved evidence blocks "
        "to answer the question. If evidence conflicts, say so explicitly.\n\n"
        + render_tokens(256, offset=313)
    )
    prefix_token_ids = _encode_prompt(tokenizer, prefix_text)
    objects: list[ReusableObject] = [
        ReusableObject(
            object_id="rag_prefix_shared",
            track="broad",
            object_type="prefix_exact",
            seed_prompt=prefix_text,
            seed_prompt_tokens=len(prefix_token_ids),
            source_tier="HBM_seed",
            canonical_text=prefix_text,
            metadata={
                "shared": True,
                "object_size_tokens": len(prefix_token_ids),
                "seed_prompt_token_ids": prefix_token_ids,
            },
        )
    ]
    object_ids_by_text: dict[tuple[str, str], str] = {}
    requests: list[HeadroomRequest] = []

    def trim_words(text: str, budget: int) -> str:
        words = text.split()
        return " ".join(words[:budget]) if len(words) > budget else text

    def pad_to_budget(text: str, budget: int, *, offset: int) -> str:
        words = text.split()
        if len(words) >= budget:
            return " ".join(words[:budget])
        return text + " " + render_tokens(budget - len(words), offset=offset)

    def ensure_object(*, title: str, text: str, object_type: str) -> str:
        key = (title, object_type)
        existing = object_ids_by_text.get(key)
        if existing is not None:
            return existing
        title_slug = slugify(title)[:48]
        suffix = "approx" if object_type == "approximate" else "exact"
        object_id = f"rag_{suffix}_{title_slug}_{len(object_ids_by_text):04d}"
        seed_prompt = f"{prefix_text}\n\n{blend_special_str}\n{text}\n\n{blend_special_str}\nQuestion: summarize the evidence."
        text_token_ids = _encode_segment(tokenizer, text)
        seed_prompt_token_ids = (
            prefix_token_ids
            + blend_special_token_ids
            + text_token_ids
            + blend_special_token_ids
            + _encode_segment(tokenizer, "Question: summarize the evidence.")
        )
        objects.append(
            ReusableObject(
                object_id=object_id,
                track="broad",
                object_type=object_type,
                seed_prompt=seed_prompt,
                seed_prompt_tokens=len(seed_prompt_token_ids),
                source_tier="HBM_seed",
                canonical_text=text,
                metadata={
                    "title": title,
                    "object_size_tokens": len(text_token_ids),
                    "seed_prompt_token_ids": seed_prompt_token_ids,
                    "canonical_token_ids": text_token_ids,
                },
            )
        )
        object_ids_by_text[key] = object_id
        return object_id

    for req_idx, example in enumerate(dataset):
        question = example["question"].strip()
        answer = example["answer"].strip()
        supporting_titles: list[str] = []
        for title in example["supporting_facts"]["title"]:
            if title not in supporting_titles:
                supporting_titles.append(title)

        context_titles = example["context"]["title"]
        context_sentences = example["context"]["sentences"]
        contexts = list(zip(context_titles, context_sentences))
        context_map = {title: sentences for title, sentences in contexts}

        selected_titles: list[str] = [title for title in supporting_titles if title in context_map]
        for title, _sentences in contexts:
            if len(selected_titles) >= top_k:
                break
            if title not in selected_titles:
                selected_titles.append(title)
        selected_titles = selected_titles[:top_k]

        request_object_ids = ["rag_prefix_shared"]
        rendered_chunks: list[str] = []
        object_occurrences = [
            {
                "object_id": "rag_prefix_shared",
                "object_type": "prefix_exact",
                "occurrence_index": 0,
                "token_start": 0,
                "token_end": 256,
                "text": prefix_text,
            }
        ]
        cursor = len(prefix_token_ids)
        prompt_token_ids = list(prefix_token_ids)
        for doc_rank, title in enumerate(selected_titles):
            sentences = context_map[title]
            exact_text = pad_to_budget(
                f"Title: {title}\n" + " ".join(sentences),
                chunk_token_budget,
                offset=7000 + req_idx * 17 + doc_rank * 5,
            )
            approx_text = perturb_text(exact_text, step=9 + (doc_rank % 3))
            use_approx = doc_rank == len(selected_titles) - 1 and (req_idx % 2 == 1)
            object_type = "approximate" if use_approx else "nonprefix_exact"
            chunk_text = approx_text if use_approx else exact_text
            object_id = ensure_object(title=title, text=chunk_text, object_type=object_type)
            request_object_ids.append(object_id)
            rendered_chunks.append(chunk_text)
            chunk_token_ids = _encode_segment(tokenizer, chunk_text)
            prompt_token_ids += blend_special_token_ids + chunk_token_ids
            chunk_len = len(chunk_token_ids)
            object_occurrences.append(
                {
                    "object_id": object_id,
                    "object_type": object_type,
                    "occurrence_index": doc_rank + 1,
                    "token_start": cursor,
                    "token_end": cursor + chunk_len,
                    "text": chunk_text,
                }
            )
            cursor += chunk_len

        prompt = (
            prefix_text
            + "\n\n"
            + f"\n\n{blend_special_str}\n".join(rendered_chunks)
            + f"\n\n{blend_special_str}\nQuestion: "
            + question
            + "\nShort answer:"
        )
        question_suffix = _encode_segment(tokenizer, "Question: " + question + "\nShort answer:")
        prompt_token_ids += blend_special_token_ids + question_suffix
        prompt_tokens = len(prompt_token_ids)
        if align_requests_to is not None:
            remainder = prompt_tokens % align_requests_to
            if remainder:
                pad_tokens = align_requests_to - remainder
                prompt += "\n\nPadding:\n" + render_tokens(pad_tokens, offset=9000 + req_idx)
                padding_token_ids = _encode_segment(
                    tokenizer, "Padding:\n" + render_tokens(pad_tokens, offset=9000 + req_idx)
                )
                prompt_token_ids += padding_token_ids
                prompt_tokens = len(prompt_token_ids)
        requests.append(
            HeadroomRequest(
                request_id=f"rag_req_{req_idx:04d}",
                track="broad",
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                output_len=output_len,
                reusable_object_ids=request_object_ids,
                metadata={
                    "question": question,
                    "answer": answer,
                    "selected_titles": selected_titles,
                    "supporting_titles": supporting_titles,
                    "object_occurrences": object_occurrences,
                    "prompt_token_ids": prompt_token_ids,
                },
            )
        )

    return WorkloadBundle(
        bundle_name=bundle_name,
        track="broad",
        description="Deterministic real-RAG empirical headroom bundle derived from HotpotQA distractor validation.",
        model_hint=None,
        objects=objects,
        requests=requests,
        metadata={
            "tokenizer_model": tokenizer_model,
            "examples_count": examples_count,
            "top_k": top_k,
            "chunk_token_budget": chunk_token_budget,
            "align_requests_to": align_requests_to,
            "output_len": output_len,
            "dataset": "hotpot_qa/distractor validation",
            "deterministic_selection": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empirical headroom workload bundles.")
    parser.add_argument(
        "--output-dir",
        default="datasets/processed/empirical_headroom",
        help="Output directory for workload bundles.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Tokenizer model used for broad-track token-ID workload construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_bundle = build_prefix_bundle(
        bundle_name="shared_prefix_64x16",
        num_groups=64,
        prompts_per_group=16,
        prefix_tokens=2048,
        suffix_tokens=256,
        output_len=1,
    )
    write_workload_bundle(output_dir / prefix_bundle.bundle_name, prefix_bundle)

    prefix_pilot_bundle = build_prefix_bundle(
        bundle_name="shared_prefix_pilot_long_8x1",
        num_groups=8,
        prompts_per_group=1,
        prefix_tokens=8192,
        suffix_tokens=256,
        output_len=1,
    )
    write_workload_bundle(output_dir / prefix_pilot_bundle.bundle_name, prefix_pilot_bundle)

    prefix_group_head_subset = subset_prefix_group_heads(
        source_bundle=prefix_bundle,
        bundle_name="shared_prefix_group_heads_8",
        groups=8,
    )
    write_workload_bundle(output_dir / prefix_group_head_subset.bundle_name, prefix_group_head_subset)

    sharegpt_reordered_pilot_bundle = build_sharegpt_reordered_prefix_bundle(
        bundle_name="sharegpt_prefix_reordered_pilot_16x4",
        source_path=ROOT / "datasets" / "processed" / "sharegpt_prefix_competition.jsonl",
        manifest_path=ROOT / "datasets" / "processed" / "sharegpt_prefix_competition.manifest.json",
        max_groups=16,
        max_requests_per_group=4,
    )
    write_workload_bundle(output_dir / sharegpt_reordered_pilot_bundle.bundle_name, sharegpt_reordered_pilot_bundle)

    sharegpt_reordered_main_bundle = build_sharegpt_reordered_prefix_bundle(
        bundle_name="sharegpt_prefix_reordered_main_24x4",
        source_path=ROOT / "datasets" / "processed" / "sharegpt_prefix_competition.jsonl",
        manifest_path=ROOT / "datasets" / "processed" / "sharegpt_prefix_competition.manifest.json",
        max_groups=24,
        max_requests_per_group=4,
    )
    write_workload_bundle(output_dir / sharegpt_reordered_main_bundle.bundle_name, sharegpt_reordered_main_bundle)

    broad_bundle = build_mixed_bundle(
        bundle_name="mixed_reuse_1024req",
        tokenizer_model=args.tokenizer_model,
        requests_count=1024,
        chunk_library_size=128,
        chunks_per_request=8,
        prefix_tokens=1024,
        chunk_tokens=256,
        tail_tokens=96,
        output_len=1,
    )
    write_workload_bundle(output_dir / broad_bundle.bundle_name, broad_bundle)

    broad_pilot_bundle = build_mixed_bundle(
        bundle_name="mixed_reuse_pilot_24req",
        tokenizer_model=args.tokenizer_model,
        requests_count=24,
        chunk_library_size=12,
        chunks_per_request=4,
        prefix_tokens=768,
        chunk_tokens=192,
        tail_tokens=96,
        output_len=1,
    )
    write_workload_bundle(output_dir / broad_pilot_bundle.bundle_name, broad_pilot_bundle)

    broad_aligned_pilot_bundle = build_mixed_bundle(
        bundle_name="mixed_reuse_aligned_pilot_24req",
        tokenizer_model=args.tokenizer_model,
        requests_count=24,
        chunk_library_size=12,
        chunks_per_request=4,
        prefix_tokens=1024,
        chunk_tokens=256,
        tail_tokens=256,
        output_len=1,
    )
    write_workload_bundle(output_dir / broad_aligned_pilot_bundle.bundle_name, broad_aligned_pilot_bundle)

    rag_pilot_bundle = build_rag_bundle(
        bundle_name="hotpotqa_rag_pilot_16req",
        tokenizer_model=args.tokenizer_model,
        examples_count=16,
        top_k=3,
        chunk_token_budget=160,
        align_requests_to=None,
        output_len=1,
    )
    write_workload_bundle(output_dir / rag_pilot_bundle.bundle_name, rag_pilot_bundle)

    rag_aligned_pilot_bundle = build_rag_bundle(
        bundle_name="hotpotqa_rag_aligned_pilot_16req",
        tokenizer_model=args.tokenizer_model,
        examples_count=16,
        top_k=3,
        chunk_token_budget=256,
        align_requests_to=256,
        output_len=1,
    )
    write_workload_bundle(output_dir / rag_aligned_pilot_bundle.bundle_name, rag_aligned_pilot_bundle)

    rag_main_bundle = build_rag_bundle(
        bundle_name="hotpotqa_rag_main_32req",
        tokenizer_model=args.tokenizer_model,
        examples_count=32,
        top_k=3,
        chunk_token_budget=160,
        align_requests_to=None,
        output_len=1,
    )
    write_workload_bundle(output_dir / rag_main_bundle.bundle_name, rag_main_bundle)


if __name__ == "__main__":
    main()
