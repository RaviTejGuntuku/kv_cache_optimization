# Critical-Path Miss Attribution

## Question

How many logical KV misses actually delay user-visible progress, and how much of the recomputation cost is exposed rather than overlapped?

This experiment is the main headroom study for:

- prefetching
- precomputation
- asynchronous KV movement
- cache-aware scheduling that hides miss cost

## Workloads

Optimistic workload:

- [critical_path_serial_resume.jsonl](/data/synthetic/headroom_studies/critical_path_miss_attribution/critical_path_serial_resume.jsonl)

Near-real workload:

- [critical_path_miss_attribution\_\_realworld_sequence.jsonl](/data/processed/headroom_studies/critical_path_miss_attribution/critical_path_miss_attribution__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

Optional future replacement:

- `LMSYS-Chat-1M` slice preserving original request order

Hypothesis:

- `critical_path_serial_resume` should maximize exposed miss cost because resumptions are deliberately serialized and poorly overlapped.
- `critical_path_miss_attribution__realworld_sequence` should still show exposed misses, but a larger fraction of recomputation should overlap other useful work than in the cooked-up serial case.

## Independent Variables

- workload
- page size `b in {16, 32, 64, 128}`
- concurrency regime
- capacity regime

## Fixed Controls

- model
- scheduler for the primary run
- request rate within a chosen regime
- GPU

Use a fixed scheduler per panel. Compare schedulers only after attribution logging is stable.

## Baselines

Compute and retain:

- compulsory misses
- `LRU`
- `OPT`

The core output of this experiment is not just miss counts; it is exposed miss cost under each baseline.

## Required New Logging

For every miss event, log:

- request id
- phase: prefill / decode / resume
- page size
- recomputed block count
- recomputed token count
- whether the request was on the critical path
- whether the recomputation overlapped useful work
- estimated exposed delay from the miss

## Metrics

- total logical misses
- compulsory misses
- critical-path misses
- critical-path miss rate
- exposed recompute blocks
- exposed recompute tokens
- exposed recompute time
- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`

## Impact Statistics

The user-facing impact statistics for this experiment are:

- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`

These should be read from the online serving benchmark output for each run. The attribution logic is
for explaining those metrics, not replacing them.

## Procedure

1. Generate workloads:

```bash
python3 data/generators/generate_headroom_study_workloads.py
python3 benchmarking/datasets/build_headroom_realworld_slices.py \
  --input data/processed/sharegpt_subset.jsonl \
  --dataset-name sharegpt_subset
```

If a local `LMSYS-Chat-1M` sequence becomes available later, rebuild the near-real slice from that
source without changing the rest of the procedure.

2. Run the workload under one fixed scheduler and one fixed capacity regime.
3. Collect traces for:
   - `LRU`
   - `OPT`
4. Attribute every miss as:
   - compulsory or reuse
   - critical-path or overlapped
5. Aggregate:
   - miss counts
   - exposed recompute blocks
   - exposed recompute tokens
6. Plot:
   - user-facing metrics vs critical-path miss rate
   - user-facing metrics vs exposed recompute tokens

## Implemented Runner

- [run_critical_path_miss_attribution.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_critical_path_miss_attribution.py)

Pilot command:

```bash
python3 benchmarking/runners/run_critical_path_miss_attribution.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_critical_path_pilot \
  --mode pilot
```

Full command:

```bash
python3 benchmarking/runners/run_critical_path_miss_attribution.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_critical_path_full \
  --mode full
```

Current implemented sweep:

- workloads:
  - optimistic synthetic serial-resume pattern
  - near-real ShareGPT sequence slice
- page size:
  - `32`
- pilot memory fraction:
  - `0.24`
- pilot prompt count:
  - `16`
- pilot max concurrency:
  - `8`
- pilot request rate:
  - `4`
- full memory fractions:
  - `0.20`, `0.28`, `0.36`
- policies at each point:
  - `LRU` first pass
  - `OPT` second pass

Expected runtime:

- pilot: about `0.08 h`
- full: about `1.25 h`

Pilot success criteria:

- `reports/lru_critical_path.json` exists
- `reports/belady_critical_path.json` exists
- the report includes:
  - `critical_path_miss_rate`
  - `mean_missed_blocks_per_request`
  - `mean_missed_tokens_per_request`
  - TTFT correlation summaries

Pilot note:

- this pilot is only checking that attribution logs and post-processing are wired correctly
- use `full` mode before interpreting the critical-path statistics

## Why This Answers the Headroom Question

Prefetching and precomputation do not need to reduce raw miss count to help. They only need to remove misses from the critical path.

If exposed miss cost is already small, then those techniques have limited upside.

If exposed miss cost is large, then hiding misses may be more promising than reducing them.
