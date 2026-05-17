# Effective Residency Sweep

## Question

If better KV-cache management increased effective reusable residency, how much could user-facing serving metrics improve?

This experiment is the main headroom study for:

- compression
- larger effective HBM
- better cache-aware scheduling
- better eviction / admission only insofar as they increase reusable residency

## Workloads

Optimistic workload:

- [residency_hotset_capacity_ladder.jsonl](/data/synthetic/headroom_studies/effective_residency_sweep/residency_hotset_capacity_ladder.jsonl)

Near-real workload:

- [effective_residency_sweep\_\_realworld_sequence.jsonl](/data/processed/headroom_studies/effective_residency_sweep/effective_residency_sweep__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

Optional future replacement:

- `LMSYS-Chat-1M` slice preserving original request order

Hypothesis:

- `residency_hotset_capacity_ladder` should show the largest benefit because the reusable working set is deliberately tuned to straddle capacity.
- `effective_residency_sweep__realworld_sequence` should still show gain, but less, because reuse is taken from a naturally occurring request stream rather than tuned synthetic interference.

## Independent Variables

- effective KV capacity
- page size `b in {16, 32, 64, 128}`
- workload

The x-axis should be reported both as:

- `kv_capacity_blocks`
- `reuse working-set coverage`

## Fixed Controls

- model
- scheduler
- request rate
- max concurrency
- tokenizer / prompt formatting
- GPU type

Start with `fcfs` so the residency curve is interpretable before adding scheduler interactions.

## Baselines

These baselines should be computed for every capacity point:

- compulsory misses
- `LRU`
- `OPT` with admission disabled
- `OPT` with incoming-line competition enabled

Interpretation:

- compulsory misses are the lower floor on logical misses
- `LRU` is the practical baseline
- `OPT` tells us eviction-only headroom
- incoming-line-aware `OPT` tells us admission plus eviction headroom

## Metrics

- total miss count and miss rate
- compulsory miss count and miss rate
- reuse miss count and miss rate
- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`
- transfer proxy bytes
- resident reusable blocks over time
- evictions over time

## Impact Statistics

The user-facing impact statistics for this experiment are:

- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`

These should be measured from the serving benchmark outputs, not inferred from offline traces.

## Procedure

1. Generate workloads:

```bash
python3 data/generators/generate_headroom_study_workloads.py
python3 benchmarking/datasets/build_headroom_realworld_slices.py \
  --input data/processed/sharegpt_subset.jsonl \
  --dataset-name sharegpt_subset
```

If `LMSYS-Chat-1M` later becomes available locally in the repo's custom JSONL format, the same
command can be rerun with that input path. The experiment design itself does not depend on LMSYS.

2. Choose one workload and one page size.
3. Sweep effective KV capacity from clearly starved to near full reuse.
4. At each capacity, run:
   - `LRU`
   - `OPT`
   - incoming-line-aware `OPT`
5. Postprocess traces to compute:
   - compulsory misses
   - reuse misses
   - reuse working-set coverage
6. Plot:
   - throughput vs miss rate
   - `TTFT` vs miss rate
   - `ITL` vs miss rate
   - throughput vs reuse working-set coverage

## Implemented Runner

- [run_effective_residency_sweep.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_effective_residency_sweep.py)

Pilot command:

```bash
python3 benchmarking/runners/run_effective_residency_sweep.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_effective_residency_pilot \
  --mode pilot
```

Full command:

```bash
python3 benchmarking/runners/run_effective_residency_sweep.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_effective_residency_full \
  --mode full
```

Current implemented sweep:

- workloads:
  - optimistic synthetic hotset ladder
  - near-real ShareGPT sequence slice
- page size:
  - `32`
- pilot memory fractions:
  - `0.24`
- full memory fractions:
  - `0.20`, `0.24`, `0.28`, `0.32`
- pilot prompt count:
  - `16`
- pilot max concurrency:
  - `16`
- pilot request rate:
  - `8`
- policies at each point:
  - `LRU` first pass
  - `OPT` second pass
  - `OPT+bypass` second pass with default reusable-cache fraction `0.6`

Expected runtime:

- pilot: about `0.08 h`
- full: about `1.75 h`

Pilot success criteria:

- every run root contains `run_metadata.json`
- `benchmarks/lru.jsonl` and `benchmarks/belady.jsonl` exist
- `traces/lru.jsonl` and `traces/belady.jsonl` exist
- `reports/comparison.json` exists
- the run completes end to end and emits valid traces, plans, and reports

Pilot note:

- this pilot is intentionally tiny and is only a pipeline smoke test
- use `full` mode for any real residency conclusions

## Why This Answers the Headroom Question

This experiment estimates the value of making more reusable KV blocks resident, regardless of whether that is achieved by:

- compression
- better placement
- better scheduling
- better admission / eviction

If the curve saturates early, then further KV management work has limited upside.

If the curve remains steep down to the compulsory floor, then stronger residency mechanisms are worth pursuing.
