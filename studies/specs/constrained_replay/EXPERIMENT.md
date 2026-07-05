# Constrained Replay Study

## Objective

This optional study asks:

- how much missed value is recoverable under small, empirical counterfactual changes without constructing a full constrained oracle?

This is intentionally weaker than a formal oracle and should be treated as a secondary study.

## Why This Exists

Marginal counterfactuals isolate the value of one missed object at a time, but they ignore conflicts between multiple fixes.

Constrained replay provides a middle ground:

- replay the observed run
- allow only a small number of object-level corrections
- estimate how much value those corrections recover jointly

This study should reuse traces collected from the same `PD-disaggregated` or otherwise `decode-isolated` prefill environment as the other headroom studies.

## Baseline Systems and Comparison Scope

Use the same three baselines as Oracle 0:

- `vLLM APC`
- `LMCache exact hierarchical reuse`
- `LMCache + CacheBlend`

Comparison scope:

- for `vLLM APC` and `LMCache exact hierarchical reuse`, constrain replay to exact-prefix missed objects
- for `LMCache + CacheBlend`, allow exact-prefix, exact non-prefix, and approximate missed objects

## Workloads

Use the exact same workloads as Oracle 0:

- shared-prefix synthetic
- mixed reusable-object synthetic
- `HotpotQA` real RAG benchmark

## Procedure

### Step 1: Start from the missed-object table

Use the output of the marginal-counterfactual study:

- object ids
- request ids
- marginal gains

### Step 2: Pick a small correction budget

Use budgets like:

- `k in {1, 5, 10, 20, 50}` corrected objects

### Step 3: Replay with corrections

For each budget `k`:

- take the top-`k` missed objects by marginal value
- inject them into HBM for their corresponding requests
- if two corrections are obviously incompatible in the same request-local measurement context, keep only the higher-value one
- rerun the affected request-local measurements and recompute aggregate latency summaries

This is not an exact global optimum.

It is a bounded empirical recovery curve.

## Metrics

- recovered latency value vs `k`
- recovered value fraction vs `k`
- recovered value by object type

## Required Outputs

- `constrained_replay_recovery.csv`
- `constrained_replay_selected_objects.jsonl`

Plots:

- recovered value vs `k`
- recovered value fraction vs `k`

## Interpretation

If a small `k` captures most missed value:

- headroom is concentrated and likely policy-recoverable

If recovered value grows only slowly with `k`:

- headroom is diffuse and may be harder to exploit with simple policy improvements
