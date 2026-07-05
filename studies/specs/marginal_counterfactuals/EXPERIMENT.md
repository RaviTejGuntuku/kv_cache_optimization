# Marginal Counterfactuals Study

## Objective

This study answers the granular question:

- for a block that was late in the baseline run, how much latency is saved if that **one** block is on time?

This is not a separate scheduler study.

It is a diagnostic follow-on to the main Oracle 0 headroom study.

## Fixed Assumptions

- scheduling policy remains `FCFS`
- the baseline system remains unchanged
- the counterfactual changes only one block
- fetch / transfer time for that one block is excluded
- repair time for approximate objects is included

## Counterfactual Definition

For a baseline request `r` and a late object `o`:

1. reproduce the same `FCFS` workload state seen by the baseline
2. make only object `o` ready in HBM before timed prefill begins
3. do not pre-repair approximate objects
4. run the normal target-request prefill path
5. stop timing at:
   - prefill completion
   - `TTFT`

Define:

- `marginal_prefill_gain(r,o) = baseline_prefill_ms(r) - counterfactual_prefill_ms(r,o)`
- `marginal_ttft_gain(r,o) = baseline_ttft_ms(r) - counterfactual_ttft_ms(r,o)`
- `marginal_prefill_gain_ms_per_1k_request_tokens = 1000 * marginal_prefill_gain(r,o) / request_prompt_tokens`
- `marginal_ttft_gain_ms_per_1k_request_tokens = 1000 * marginal_ttft_gain(r,o) / request_prompt_tokens`
- `marginal_prefill_gain_ms_per_1k_object_tokens = 1000 * marginal_prefill_gain(r,o) / object_size_tokens`
- `marginal_ttft_gain_ms_per_1k_object_tokens = 1000 * marginal_ttft_gain(r,o) / object_size_tokens`

Keep both raw and normalized gains. Raw milliseconds show absolute request-level
headroom; normalized gains prevent larger requests or larger objects from
dominating interpretation only because they contain more tokens.

## Scope

### Prefix track

System:

- `vllm_apc`

Objects in scope:

- exact prefix objects

### Broad track

System:

- `lmcache_cacheblend`

Objects in scope:

- exact prefix objects
- exact non-prefix objects
- approximate objects

Baseline requirement:

- use the same LMCache configuration as the main Oracle 0 study
- this includes LMCache async loading / preload machinery being enabled
- marginal counterfactuals should measure residual missed-opportunity value beyond that stronger LMCache baseline

## Input Dependency

This study depends on the main Oracle 0 / baseline runs.

Use the same:

- workload ordering
- concurrency setting
- target requests
- system configuration

## Baseline-Late Filter

Primary analysis set:

- objects that were actually late in the baseline

This is the important distinction:

- the study is about **late blocks**
- not all reusable-object candidates

Keep both:

- `all_candidates`
- `late_only`

But the primary headroom interpretation must use `late_only`.

## Required Raw Data

Each row should contain:

- `workload`
- `system`
- `concurrency`
- `request_id`
- `object_id`
- `object_type`
- `object_size_tokens`
- `source_tier`
- `request_prompt_tokens`
- `request_reusable_object_count`
- `token_position_coverage`
- `baseline_prefill_ms`
- `counterfactual_prefill_ms`
- `baseline_ttft_ms`
- `counterfactual_ttft_ms`
- `marginal_prefill_gain_ms`
- `marginal_ttft_gain_ms`
- `marginal_prefill_gain_ms_per_1k_request_tokens`
- `marginal_ttft_gain_ms_per_1k_request_tokens`
- `marginal_prefill_gain_ms_per_1k_object_tokens`
- `marginal_ttft_gain_ms_per_1k_object_tokens`
- `was_late_in_baseline`
- `repair_expected`

`token_position_coverage` is important for later interpretation even if it is not yet fully used in the study design.

## Procedure

### Step 1: Run baseline

Use the normal system under fixed `FCFS`.

Record:

- baseline prefill
- baseline `TTFT`
- late-object evidence

### Step 2: Enumerate late objects

For each target request:

- enumerate reusable objects in scope
- identify which of them were actually late in the baseline

### Step 3: One-object-on-time rerun

For each late object `(r, o)`:

- reproduce the same baseline-visible state
- make only `o` ready in HBM before the timed region
- run the normal target request
- measure prefill and `TTFT`

## Recommended First Analysis Pass

Before generating polished plots, inspect the raw rows directly.

Specifically look at:

- largest positive gains
- negative gains
- gains by object type
- gains by token-position coverage
- gains by object size
- gains by request length

This study is especially useful for raw table inspection before plotting.

## Useful Plots

Generate only plots that are easy to interpret.

### Primary

- CDF of marginal prefill gain over late objects
- CDF of marginal `TTFT` gain over late objects
- mean marginal gain by object type
- total marginal gain by object type
- marginal saved milliseconds by request/object
- marginal saved milliseconds per 1k request tokens
- marginal saved milliseconds per 1k object tokens
- marginal gain vs object size
- marginal gain vs request prompt length
- top-`k` cumulative gain curve

### Optional

- marginal gain vs token-position coverage
- marginal gain vs concurrency

Do **not** generate a request heatmap for this study.

## Nsight Systems

`Nsight Systems` is optional for this study.

It is useful for a small sample of rows to verify:

- the one-block change is the only meaningful change
- repair actually appears inside the timed region
- copy / compute overlap is understood
- negative-gain cases are real and not instrumentation artifacts

It is not necessary for every marginal row.

## Result Directory Naming

Use explicit names such as:

- `marginal_counterfactuals_fcfs__fabricated_prefix__vllm_apc`
- `marginal_counterfactuals_fcfs__natural_reordered__lmcache_cacheblend`
- `marginal_counterfactuals_fcfs__real_rag__lmcache_cacheblend`

Within each folder:

- `raw/`
- `late_only/`
- `plots/`
- `summary/`
- `nsys_samples/`

## Interpretation

If Oracle 0 says headroom exists, this study explains where it is concentrated.

Questions it should answer:

- are a few late blocks dominating the missed value?
- are non-prefix or approximate objects the important misses?
- does value scale with size, position coverage, or request length?
