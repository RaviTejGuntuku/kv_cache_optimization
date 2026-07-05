# Missed-Opportunity Accounting Study

## Objective

This study asks:

- when a reusable object is not realized by a baseline system, **why** did that happen?

The purpose is not to estimate a perfect upper bound.

The purpose is to attribute unrealized reusable value to concrete causes such as:

- source unavailability
- HBM pressure
- transfer lateness
- bandwidth conflict
- policy miss
- approximate reuse not being worth it

## Core Idea

For every reusable object that a baseline system failed to realize for a request:

1. compute or look up its estimated value
2. determine the first binding reason it failed
3. aggregate unrealized value by cause

This is the main `why is there headroom?` study.

This study runs as an `offline analysis pass` over the Oracle 0 and marginal-counterfactual outputs collected in the same decode-minimized setup, so observed misses are attributable to reusable-object realization rather than decode-heavy serving interference.

## Baseline Systems and Comparison Scope

Use the same three baselines as Oracle 0:

- `vLLM APC`
- `LMCache exact hierarchical reuse`
- `LMCache + CacheBlend`

Comparison scope:

- for `vLLM APC` and `LMCache exact hierarchical reuse`, account only for exact-prefix objects
- for `LMCache + CacheBlend`, account for exact-prefix, exact non-prefix, and approximate objects

## Workloads

Use the exact same workloads as Oracle 0:

- shared-prefix synthetic
- mixed reusable-object synthetic
- `HotpotQA` real RAG benchmark

## Implemented Analysis Inputs

From the workload bundle:

- `manifest.json`
- `objects.jsonl`
- `requests.jsonl`

From the Oracle 0 run:

- `oracle0_measurements.jsonl`

From the marginal-counterfactual run:

- `baseline_replay_measurements.jsonl`
- `marginal_counterfactuals.jsonl`

Optional side evidence:

- sampled `Nsight Systems` traces from the first two studies

Implemented analysis script:

- [analyze_empirical_missed_opportunities.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/analysis_scripts/analyze_empirical_missed_opportunities.py)

## Cause Taxonomy

The long-term goal is to assign each missed object to the **first** cause that explains why it failed.

For the pilot, the implemented script uses a smaller heuristic cause set that can be derived from the first two studies without new instrumentation:

- `already_realized_or_low_value`
- `approx_not_worth_it`
- `no_request_level_gap`
- `prefix_realization_gap`
- `nonprefix_realization_gap`
- `approximate_realization_gap`

These categories are enough to quantify pure-number headroom and start inferring why broad reusable value is not being captured. The richer taxonomy below remains the target for a later refinement pass.

### 1. `source_absent`

The object was not actually present in any backing tier when the request needed it.

Examples:

- first-touch object
- object was never stored
- object was evicted from lower-tier storage before reuse

### 2. `identified_too_late`

The object became known only after there was no longer enough time to make it available.

This is expected to be rare in the current setup because reusable objects are assumed known by queue entry, but the category should still exist.

### 3. `late_arrival`

The baseline did issue a fetch or retrieval, but the object reached HBM after it could help the request.

### 4. `bandwidth_conflict`

The object would have been useful and the source existed, but the transfer path was occupied by other activity and the object could not be moved in time.

### 5. `HBM_conflict`

The object could have been fetched in time, but no HBM residency window was available under the observed run.

### 6. `policy_miss`

Under the observed source availability and approximate physical conditions, the baseline could have realized the object, but did not try or chose a weaker alternative.

### 7. `approx_not_worth_it`

The object is approximate, but its repair-aware marginal gain is non-positive or negligible.

## Procedure

### Step 1: Build the missed-object table

For each analyzed counterfactual row, create one row with:

- request id
- object id
- object type
- source tier
- baseline request-local time
- Oracle 0 request-local time
- marginal counterfactual gain

### Step 2: Assign causes

In the pilot implementation, assign each object a heuristic cause bucket using:

- object type
- Oracle 0 gap for the request
- marginal gain for the object
- cached-token deltas when available

Use Nsight traces as side evidence when a large marginal gain needs manual interpretation.

### Step 3: Aggregate by cause

For each baseline and workload, compute:

- total unrealized value by cause
- unrealized value by object type and cause
- request-level Oracle 0 gap
- positive marginal-gain coverage of the Oracle 0 gap

## Nsight Systems Usage

Nsight Systems is supporting evidence for this study.

Use it to determine:

- whether the object-transfer path was active or idle
- whether a transfer started but completed after the relevant prefill window
- whether repair overlapped with prefill or became a critical-path stall
- whether HBM-related memory activity indicates severe overlap/conflict at the missed opportunity
- whether any decode-side kernels or transfers were present on the measured prefill-GPU timeline

The pilot accounting script does not require Nsight to run, but Nsight traces are the grounding evidence you should use when deciding whether a large realization gap is plausibly due to transfer timing, repair criticality, or other runtime effects.

## Metrics

### Primary

- unrealized value by cause

### Secondary

- unrealized value by object type
- unrealized value by source tier
- count of missed objects by cause
- bytes of missed objects by cause

## Required Outputs

- `missed_opportunities.jsonl`
- `request_gap_summary.jsonl`
- `cause_breakdown.csv`
- `cause_by_object_type.csv`
- `summary.json`

Plots:

- stacked bar of unrealized value by cause and baseline
- stacked bar of unrealized value by object type and cause
- request-level Oracle 0 gap versus recovered marginal value

## Interpretation

If unrealized value is dominated by:

- `policy_miss`, then there is strong case for better cache-state / prefetch policy
- `bandwidth_conflict`, then the movement path is the main bottleneck
- `HBM_conflict`, then HBM capacity or residency management is the main bottleneck
- `source_absent`, then lower-tier retention / write-back policy is the main bottleneck
- `approx_not_worth_it`, then approximate reuse is less promising than it appears on paper
