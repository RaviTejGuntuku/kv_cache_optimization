# Pilot Plan

## Purpose

The pilot suite is a plumbing-validation stage.

It is **not** the main headroom experiment.

Its only job is to verify that:

- workload generation is correct
- baseline, Oracle 0, and marginal-counterfactual measurements are computed correctly
- prefix, non-prefix, and approximate reusable objects are all exercised where intended
- the real RAG path is deterministic and reproducible
- plots and summaries are readable enough to catch mistakes before the full run

The pilot suite should be cheap enough to rerun quickly.

Target budget:

- `1-2 GPU-hours total`

## Pilot Outputs

Each pilot should write:

- `summary.md`
- `coverage_table.csv`
- `oracle_vs_baseline_by_request.png`
- `marginal_gain_by_object_type.png`
- `marginal_gain_cdf.png`
- `topk_cumulative_gain.png`

The outputs should be interpretable without inspecting raw JSONL unless debugging.

## Pilot P: Exact-Prefix Plumbing

### Goal

Validate exact-prefix Oracle 0 and true-miss marginal-counterfactual semantics.

### Workload

- shared-prefix synthetic
- `8-16` requests
- long enough prefixes that reuse clearly matters

### System

- `vllm_apc`

### What Must Be True

- Oracle 0 latency is lower than baseline latency
- marginal rows correspond to blocks that were actually unavailable on time in the baseline
- baseline cached-token count is near zero for missed rows
- counterfactual cached-token count increases sharply for those same rows

### Pass Conditions

- all or nearly all counterfactual rows satisfy `was_missed_in_baseline = true`
- mean marginal gain is clearly positive
- no setup artifact causes baseline to look better than Oracle 0

## Pilot M: Mixed-Object Plumbing

### Goal

Validate broad-reuse workload coverage and broad-runner output structure.

### Workload

- mixed reusable-object synthetic
- `24-48` requests
- must include:
  - `prefix_exact`
  - `nonprefix_exact`
  - `approximate`

### Systems

- `vllm_apc` for exact-prefix sanity
- `lmcache_cacheblend` for broad-reuse sanity

### What Must Be True

- all three object types appear in bundle manifests
- all three object types appear in marginal outputs for the broad-reuse stack
- approximate rows are marked `repair_expected = true`
- Oracle 0 broad runs preload all reusable objects listed for each request

### Pass Conditions

- prefix / non-prefix / approximate all appear in outputs
- at least some non-prefix or approximate rows have positive marginal gain
- no object-type-specific measurement path is silently missing

## Pilot R: Real RAG Plumbing

### Goal

Validate deterministic real-prompt assembly and reusable-object manifest persistence.

### Workload

- small `HotpotQA` slice
- `16-32` requests

### Fixed Setup

- deterministic retriever
- fixed chunk size
- fixed chunk overlap
- fixed `top-k`
- persisted retrieved chunk ids and ordering

### Systems

- exact-prefix sanity path as applicable
- `lmcache_cacheblend` for broad-reuse sanity

### What Must Be True

- retrieved chunk ids are stable across reruns
- Oracle 0 and marginal code run correctly on real prompt assembly
- reusable-object manifests are saved and readable
- plots reveal which object types are actually present in real RAG prompts

### Pass Conditions

- no retrieval nondeterminism
- at least one positive marginal row on real data
- no disagreement between saved manifest and executed prompt assembly

## Pilot Coverage Table

Each pilot should populate a `coverage_table.csv` with:

- `pilot_name`
- `workload_name`
- `system`
- `request_count`
- `object_types_observed`
- `true_missed_rows`
- `positive_marginal_rows`
- `approximate_rows`
- `repair_expected_rows`
- `mean_oracle0_gap_ms`
- `mean_marginal_gain_ms_over_true_misses`

## Pilot Gate

Do **not** run the main headroom experiment unless all three pilots pass.

Required gate:

- `Pilot P` validates exact-prefix miss semantics
- `Pilot M` validates prefix / non-prefix / approximate coverage
- `Pilot R` validates real-RAG determinism and manifest correctness
- plots and summaries are readable enough to catch obvious metric contradictions
