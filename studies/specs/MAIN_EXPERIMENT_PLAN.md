# Main Experiment Plan

## Purpose

This is the actual headroom study.

It begins only after the pilot suite has passed.

The goal is to measure:

- `Oracle 0` headroom
- `marginal counterfactual` headroom

across:

- exact prefix reuse
- exact non-prefix reuse
- approximate reuse
- synthetic and real RAG workloads

## Runtime Budget

Hard budget target:

- about `8 GPU-hours total`

This budget rules out exhaustive broad-reuse marginal enumeration.

The main experiment must therefore be designed as a **sampled headroom estimation study**, not a full census.

## Workloads

Run only three workloads:

1. exact-prefix synthetic
2. mixed reusable-object synthetic
3. small real RAG workload

### Suggested Sizes

- exact-prefix synthetic: `96` requests
- mixed reusable-object synthetic: `64` requests
- real RAG workload: `32` requests

These can be adjusted slightly, but should stay in the same order of magnitude.

## Systems

### Exact-prefix track

- `vllm_apc`
- `lmcache_exact`
- `Oracle 0P`

### Broad-reuse track

- `lmcache_cacheblend`
- `Oracle 0B`

## Oracle 0 Budget

Target:

- `2-3 GPU-hours total`

### Run Set

- exact-prefix synthetic
- mixed reusable-object synthetic
- real RAG workload

### Execution Rules

- `repeat_count = 1`
- decode minimized
- no exhaustive Nsight profiling
- only sampled representative trace capture

## Marginal Counterfactual Budget

Target:

- `4-5 GPU-hours total`

### Exact-prefix workload

- near-full marginal enumeration is acceptable if request count is modest

### Mixed and RAG workloads

- sample, do not exhaustively enumerate
- measure at most `1-2` missed objects per request
- stratify sampling by:
  - `object_type`
  - object size bucket
  - source tier if relevant

### Required Reporting Set

Primary summaries should be computed over:

- rows with `was_missed_in_baseline = true`

or the stronger broad-reuse equivalent once object-level realization instrumentation is available.

## Nsight Budget

Target:

- `< 1 GPU-hour total`

Only collect representative traces for:

- one exact-prefix miss
- one non-prefix miss
- one approximate miss with repair
- one real-RAG case

Nsight is for verification and interpretation, not bulk measurement.

## Required Final Outputs

### Oracle 0

- latency gap by workload
- latency gap by system
- latency gap by reusable-object track
- saved milliseconds by request
- saved milliseconds per 1k prompt tokens
- saved milliseconds per 1k reusable tokens

### Marginal Counterfactuals

- marginal gain CDF
- marginal gain by object type
- marginal saved milliseconds by request/object
- marginal saved milliseconds per 1k request tokens
- marginal saved milliseconds per 1k object tokens
- top-`k` cumulative missed value
- share of missed value attributable to:
  - prefix exact
  - non-prefix exact
  - approximate

### Real RAG

- retrieved chunk reuse profile
- which object classes actually contributed headroom

## Summary Runtime Envelope

Recommended allocation:

- pilots: `1-2 GPU-hours`
- Oracle 0 main run: `2-3 GPU-hours`
- marginal main run: `4-5 GPU-hours`
- sampled Nsight traces: folded into the above, but ideally `<1 GPU-hour`

If runtime projections exceed this envelope, reduce:

- request counts
- marginal sample count per request
- number of profiled trace samples

before reducing methodological clarity.

## Non-Goals

The main experiment is not intended to:

- exhaustively enumerate every broad-reuse object
- benchmark every possible serving system
- produce a final system paper benchmark suite

Its purpose is strictly:

- to gauge reusable-KV headroom quickly and defensibly
