# Empirical Headroom Harness

This directory contains the first-pass GPU harness for the empirical KV-cache headroom studies.

## Components

- `schema.py`
  - workload bundle and measurement dataclasses
- `interface.py`
  - experiment-owned request/object execution plans
  - ordered object occurrences inside each request
  - counterfactual candidate selection independent of backend quirks
- `adapters.py`
  - backend adapters for:
    - `vllm_apc`
    - `lmcache_exact`
    - `lmcache_cacheblend`
- `case_runner.py`
  - request-local execution helpers used by the experiment runners
- `nvtx.py`
  - lightweight NVTX range helper for Nsight Systems traces

## Active studies

The implemented first-pass studies are:

- `Oracle 0`
- `Marginal counterfactuals`
- `Missed-opportunity accounting`

All three are run in a decode-minimized setup:

- single node
- `max_tokens = 1`
- prefill latency is the primary metric

## Main runners

- `benchmarking/runners/run_oracle0_empirical.py`
- `benchmarking/runners/run_marginal_counterfactuals_empirical.py`
- `benchmarking/analysis_scripts/analyze_empirical_missed_opportunities.py`

## Typical flow

1. Generate workload bundles:
   - `benchmarking/workload_generators/generate_empirical_headroom_workloads.py`
2. Run Oracle 0 for a baseline system and bundle.
3. Run marginal counterfactuals for the same baseline system and bundle.
4. Run missed-opportunity accounting over the Oracle 0 and marginal outputs.

## Notes

- The current accounting script is intentionally heuristic. It consumes the outputs of Oracle 0 and the marginal study and produces a first-pass cause breakdown from pure numbers plus available metadata.
- Nsight Systems traces are sampled by the Oracle 0 and marginal runners and should be used to validate suspected realization gaps, transfer stalls, and repair-path criticality.
