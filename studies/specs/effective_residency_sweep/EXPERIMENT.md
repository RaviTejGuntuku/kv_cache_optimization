# Effective Residency Headroom Study

## Actual Objective

This study is trying to answer one concrete question:

- if KV management were good enough to drive miss rate down toward the **compulsory-miss floor**, how much would user-facing serving metrics improve?

That is the real headroom question for:

- better eviction
- better prefetching
- better cache-aware scheduling
- better compression, insofar as it increases effective reusable residency

The point is **not** to compare policies for their own sake.

The point is to estimate:

- how much value is left if we asymptotically approach compulsory misses

Important boundary:

- this study is only about **effective reusable residency**
- it is **not** the queue-information study
- queue visibility should be treated as a separate experiment entirely

## Workloads

Optimistic workload:

- [residency_hotset_capacity_ladder.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/headroom_studies/effective_residency_sweep/residency_hotset_capacity_ladder.jsonl)

Near-real workload:

- [effective_residency_sweep__realworld_sequence.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/processed/headroom_studies/effective_residency_sweep/effective_residency_sweep__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

## Residency Headroom

### Independent Variables

- effective KV capacity
- workload

Keep fixed:

- page size
- model
- scheduler
- request rate
- max concurrency

Recommended first pass:

- scheduler: `fcfs`
- page size: `32`

### Baselines

At each capacity point compute:

- compulsory misses
- `LRU`
- `OPT` / offline Belady

Interpretation:

- compulsory misses = lower bound
- `LRU` = practical baseline
- `OPT` = best possible eviction-only point

### Metrics

- total miss count / miss rate
- compulsory miss count / miss rate
- reuse miss count / miss rate
- output throughput
- request throughput
- median / p99 `TTFT`
- median / p99 `ITL`

### Correct Capacity-Sweep Requirement

The sweep is **not complete** unless the highest-capacity points push the observed miss rate close to the compulsory floor.

If the top capacity point still leaves a large gap above compulsory misses, then the experiment has not actually measured full headroom.

So the stopping condition should be:

- continue increasing effective capacity until either:
  - miss rate is within a small tolerance of compulsory miss rate
  - or capacity is physically infeasible

Recommended tolerance:

- `miss_rate - compulsory_miss_rate <= 0.02`

### Procedure

1. Choose one workload.
2. Fix scheduler, page size, request rate, and concurrency.
3. Sweep effective capacity upward.
4. At each point run:
   - `LRU`
   - `OPT`
5. Stop only when `LRU` approaches the compulsory floor or hardware limits are reached.
6. Plot:
   - throughput vs miss rate
   - median / p99 `TTFT` vs miss rate
   - median / p99 `ITL` vs miss rate
   - throughput vs distance-above-compulsory

The key x-axis should be:

- `distance_above_compulsory = miss_rate - compulsory_miss_rate`

That is much more meaningful than raw memory fraction.

## Recommended Interpretation

### If the curve shows little gain near the compulsory floor

Then better KV management probably has limited upside overall.

### If the curve shows large gain as miss rate approaches the compulsory floor

Then there is real upside left in better KV management mechanisms that increase effective reusable residency, such as:

- better eviction
- prefetching
- compression
- cache-aware scheduling

## Implemented Runner

- [run_effective_residency_sweep.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_effective_residency_sweep.py)

Important note:

- this runner is only for the residency headroom study
- queue-information usefulness should be defined as a separate future study, not folded into this one

## Pilot Command

```bash
python3 benchmarking/runners/run_effective_residency_sweep.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root studies/results/headroom_effective_residency_pilot \
  --mode pilot
```

## Full Command

```bash
python3 benchmarking/runners/run_effective_residency_sweep.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root studies/results/headroom_effective_residency_full \
  --mode full
```
