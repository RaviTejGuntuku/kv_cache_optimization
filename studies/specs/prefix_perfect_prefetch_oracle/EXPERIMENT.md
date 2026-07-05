# Oracle 0 FCFS Headroom Study

## Objective

This is the main headroom study.

It asks:

- under fixed `FCFS` scheduling, how far are current systems from the upper bound where the reusable KV needed by a target request is already on time?
- how does that gap vary with:
  - workload
  - request length
  - reusable-token volume
  - concurrency
  - reuse type

This study consolidates the old standalone concurrency study into Oracle 0 itself.

## Fixed Assumptions

These are not knobs in this study:

- scheduling policy is `FCFS`
- the serving stack remains the system under test
- approximate objects are **not** pre-repaired
- repair time is part of Oracle 0 timing
- all reusable objects for the target request are loaded for the Oracle 0 rerun

## Oracle 0 Definition

For a target request `r`:

1. reproduce the same `FCFS` request order and same competing requests as the baseline run
2. reproduce the same baseline-visible cache state right before `r` would execute
3. force **all reusable objects for `r` that are in scope for the comparison track** into the ready-to-use HBM state
4. start timing only after those objects are ready in HBM
5. run the normal prefill path for `r`
6. include any approximate-object repair done during prefill
7. stop timing at:
   - prefill completion
   - `TTFT`

Excluded from the timed region:

- object discovery
- fetch / transfer / realization time
- preload scheduling delay

Included in the timed region:

- normal prefill compute
- any repair needed for approximate reuse
- any remaining system overhead after the objects are already ready

## Comparison Tracks

There are two apples-to-apples comparisons.

### Track P: Prefix Reuse Headroom

Compare:

- `vLLM APC`
- `Oracle 0P`

Reusable-object universe:

- exact prefix objects only

Interpretation:

- how much prefix-reuse headroom remains beyond vLLM APC?

### Track B: Broad Reuse Headroom

Compare:

- `LMCache + CacheBlend`
- `Oracle 0B`

Reusable-object universe:

- exact prefix objects
- exact non-prefix objects
- approximate objects

Interpretation:

- how much broad reusable-KV headroom remains even when LMCache already performs scheduler-aware preload / realization?

## Systems Under Test

### `vllm_apc`

Purpose:

- real prefix-only baseline

Behavior:

- vLLM automatic prefix caching
- no non-prefix reuse
- no approximate reuse

### `lmcache_cacheblend`

Purpose:

- real broad-reuse baseline

Behavior:

- LMCache retrieval path
- LMCache async loading enabled
- CacheBlend enabled
- scheduler-aware preload / realization left intact
- same `FCFS` request order as the baseline trace

The entire point is to measure residual headroom **given** that LMCache already has a prefetching mechanism.

Practical requirement:

- the LMCache baseline used in this study must enable the async-loading / preload machinery that can move reusable KV from lower tiers into GPU-resident serving buffers ahead of or during request execution
- do not benchmark a reactive-only LMCache path and call it the final broad-reuse baseline

## Workloads

Use three workload classes.

### Workload A: Fabricated Prefix Demonstration

Purpose:

- cleanly demonstrate prefix-reuse headroom

Construction:

- long shared-prefix families
- interleaved request order
- enough reuse distance that passive retention does not trivially keep the prefix in HBM

Required property:

- without a prefetcher, the useful prefix generally does not remain in HBM by reuse time

### Workload B: Natural But Reordered Reuse Trace

Purpose:

- evaluate broad reuse without overfabricating request contents

Construction:

- start from natural reusable-object-bearing requests
- apply one defensible reordering

Required property of the reordering:

- reuse exists
- the reuse distance is large enough that, absent timely preload, the useful reusable objects generally would not still reside in HBM

This is the “cache-lousy but reuse-bearing” trace.

### Workload C: Real RAG

Purpose:

- test the same question on a realistic retrieval-heavy workload

Construction:

- real RAG benchmark requests
- keep the natural retrieval semantics
- use a defensible ordering that preserves realism while exposing reuse distance

## Primary Independent Variables

### 1. Workload

- fabricated prefix
- natural reordered reuse
- real RAG

### 2. System

- `vllm_apc`
- `lmcache_cacheblend`

### 3. Concurrency

Concurrency is part of Oracle 0, not a separate experiment.

This is request-level prefill concurrency under fixed `FCFS`.

Final concurrency panel will be chosen after pilot timing, but the study should be structured to support a small set such as:

- `1`
- `4`
- `8`
- `16`

### 4. Request length

At minimum:

- prompt tokens
- reusable tokens
- reusable object count

## Measurement Protocol

For each workload `W`, system `S`, concurrency level `n`, and target request `r`:

### Step 1: Baseline replay

Run the real system under:

- fixed `FCFS`
- the chosen workload order
- the chosen concurrency level

Measure per target request:

- prefill latency
- `TTFT`
- prompt tokens
- reusable-token volume
- reusable object count
- cached-token count if exposed

### Step 2: Oracle 0 rerun

Run the same workload order and same concurrency level, but for the target request `r`:

- load all reusable objects in scope for `r`
- ensure they are ready in HBM before timed prefill begins
- do not pre-repair approximate objects
- allow the normal repair path to execute during timed prefill

Measure:

- oracle prefill latency
- oracle `TTFT`

### Step 3: Join baseline and oracle rows

Compute:

- `prefill_gap_ms = baseline_prefill_ms - oracle_prefill_ms`
- `ttft_gap_ms = baseline_ttft_ms - oracle_ttft_ms`
- `prefill_gap_ms_per_1k_prompt_tokens = 1000 * prefill_gap_ms / prompt_tokens`
- `ttft_gap_ms_per_1k_prompt_tokens = 1000 * ttft_gap_ms / prompt_tokens`
- `prefill_gap_ms_per_1k_reusable_tokens = 1000 * prefill_gap_ms / reusable_token_volume`
- `ttft_gap_ms_per_1k_reusable_tokens = 1000 * ttft_gap_ms / reusable_token_volume`

Keep both raw and normalized gains. Raw milliseconds show absolute user-visible
headroom; normalized gains prevent long requests from looking more impressive
only because they contain more tokens.

## Broad LMCache Rerun Design After Audit

The LMCache broad path has an extra measurement constraint: under batched
generation, vLLM/LMCache may expose cohort-level prefill and `TTFT` timings
rather than independent per-request timings. If those cohort-level timings are
then adjusted by different per-request HBM materialization times, the resulting
per-request Oracle gaps are not interpretable.

Therefore the broad LMCache result must be produced in two stages:

- **Stage 1: valid request-level headroom.** Run `lmcache_cacheblend` at
  `concurrency = 1`, preferably with isolated baseline and Oracle processes.
  This is the headline broad Oracle 0 result until cohort-level timing is added.
- **Stage 2: concurrency stress diagnostic.** If `concurrency > 1` is used for
  LMCache, treat the measured unit as the cohort, not the individual request,
  unless instrumentation proves that per-request prefill and `TTFT` are truly
  independent. Do not plot these rows as per-request Oracle gaps.

Any LMCache broad run must pass the Oracle 0 audit before interpretation:

- baseline and Oracle rows pair exactly by request id
- LMCache async loading is enabled
- Oracle rows applied fetch/materialization exclusion
- HBM materialization is non-negative and does not exceed raw prefill
- no batch-shared timing is being presented as per-request timing for
  `concurrency > 1`

The practical rerun entrypoint for this corrected broad path is:

```bash
benchmarking/setup/run_oracle0_broad_request_level_rerun.sh
```

## Required Metrics

### Per-request metrics

- baseline prefill latency
- oracle prefill latency
- baseline `TTFT`
- oracle `TTFT`
- saved prefill milliseconds per request
- saved `TTFT` milliseconds per request
- saved prefill milliseconds per 1k prompt tokens
- saved `TTFT` milliseconds per 1k prompt tokens
- saved prefill milliseconds per 1k reusable tokens
- saved `TTFT` milliseconds per 1k reusable tokens
- prompt tokens
- reusable-token volume
- reusable object count
- object-type coverage

### Aggregates

For each workload × system × concurrency bucket:

- prefill `p50`
- prefill `p90`
- prefill `p99`
- `TTFT p50`
- `TTFT p90`
- `TTFT p99`
- oracle-gap `p50`
- oracle-gap `p90`
- oracle-gap `p99`

## Required Plots

For each workload:

- concurrency vs prefill latency bars:
  - `p50`
  - `p90`
  - `p99`
- concurrency vs `TTFT` bars:
  - `p50`
  - `p90`
  - `p99`

These plots must show:

- `Oracle 0P` vs `vLLM APC` for prefix headroom
- `Oracle 0B` vs `LMCache + CacheBlend` for broad-reuse headroom

Also generate:

- saved prefill milliseconds by request
- saved `TTFT` milliseconds by request
- saved prefill milliseconds per 1k prompt tokens by request
- saved `TTFT` milliseconds per 1k prompt tokens by request
- saved prefill milliseconds per 1k reusable tokens by request
- saved `TTFT` milliseconds per 1k reusable tokens by request
- prefill gap vs prompt tokens
- prefill gap vs reusable tokens
- `TTFT` gap vs prompt tokens
- `TTFT` gap vs reusable tokens

## Nsight Systems

`Nsight Systems` is useful but not required for the first quantitative headroom study.

What Nsight adds:

- CUDA kernel timeline for prefill
- H2D / D2H memcpy timing
- overlap or lack of overlap between retrieval-related copies and compute
- CPU launch gaps / synchronization stalls
- stream-level idle regions
- confirmation that the timed region boundaries are correct
- evidence about whether repair or retrieval is actually dominating

What Nsight does **not** replace:

- per-request latency measurement
- oracle / baseline headroom numbers

Recommended use:

- do not profile every row in the main run
- profile a small, representative sample:
  - one short request
  - one medium request
  - one long request
  - one high-gap case
  - one near-zero-gap case

So:

- `Nsight Systems` is **not required** to run the first headroom study
- it is highly useful for auditing whether the experiment is behaving correctly

## Result Directory Naming

Result folders should be self-describing.

Use names like:

- `oracle0_fcfs_prefix_headroom__fabricated_prefix__vllm_apc`
- `oracle0_fcfs_broad_headroom__natural_reordered__lmcache_cacheblend`
- `oracle0_fcfs_broad_headroom__real_rag__lmcache_cacheblend`

Within each folder:

- `raw/`
- `joined/`
- `plots/`
- `summary/`
- `nsys_samples/`

## Non-Goals

This study is not:

- a scheduler-policy comparison
- a queue-completion optimality proof
- a storage-hierarchy sweep

It is a fixed-`FCFS`, empirical, request-level headroom study.
