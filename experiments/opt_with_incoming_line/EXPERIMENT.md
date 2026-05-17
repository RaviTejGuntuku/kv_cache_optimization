# OPT With Incoming-Line Competition

## Question

Does allowing the oracle to compare the incoming reusable KV block against the current resident set
meaningfully increase headroom over eviction-only `OPT`, once KV-HBM is explicitly partitioned into
live working state and a reusable cache?

This is the admission-control and KV-HBM partitioning experiment.

## Cache Model

Let the total KV-usable HBM budget be `M`.

Partition it into:

- `L = (1 - n) * M`: live working region for active prefills and decodes
- `C = n * M`: reusable cache region for KV blocks retained only for future reuse

Interpretation:

- `L` is not the cache. It is the mandatory live region needed for correctness and forward progress.
- `C` is the reusable prefix cache.
- Bypass applies only to admission into `C`, not to whether a live block may exist in `L`.

So the oracle decision point is:

- when a block leaves the strictly-live regime and becomes a reusable candidate, should it be admitted to `C`?

## Workloads

Optimistic workload:

- [incoming_suffix_pollution.jsonl](/data/synthetic/headroom_studies/opt_with_incoming_line/incoming_suffix_pollution.jsonl)

Near-real workload:

- [opt_with_incoming_line\_\_realworld_sequence.jsonl](/data/processed/headroom_studies/opt_with_incoming_line/opt_with_incoming_line__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

Optional future replacement:

- `LMSYS-Chat-1M` slice preserving original request order

Hypothesis:

- `incoming_suffix_pollution` should maximize the gap between eviction-only `OPT` and incoming-line-aware `OPT`, because many personalized suffix blocks should be poor admission candidates.
- `opt_with_incoming_line__realworld_sequence` should still show some gain, but less, because the incoming blocks come from a naturally occurring request stream rather than from deliberately suffix-polluting synthetic traffic.

## Independent Variables

- workload
- page size `b in {16, 32, 64, 128}`
- cache fraction `n`

## Fixed Controls

- model
- scheduler
- request rate
- concurrency

## Baselines

Always report:

- compulsory misses
- `LRU`
- eviction-only `OPT`
- incoming-line-aware `OPT`

Interpretation:

- the gap from `LRU` to eviction-only `OPT` is eviction headroom
- the extra gap from eviction-only `OPT` to incoming-line-aware `OPT` is admission headroom

Policy semantics:

- `LRU`: whole KV budget treated as reusable, i.e. effectively `n = 100%`
- eviction-only `OPT`: whole KV budget treated as reusable, i.e. effectively `n = 100%`
- incoming-line-aware `OPT`: uses the explicit partition `(L, C)` and sweeps `n`

This means the partition sweep is used only to find the best achievable `OPT+bypass` design point.
After that, compare the best `OPT+bypass` result against `LRU` and eviction-only `OPT` on the same workload.

## Partition Sweep

The first stage of the experiment is to choose the best cache fraction `n`.

Recommended sweep:

- `n in {0.4, 0.5, 0.6, 0.7, 0.8}`

Rationale:

- below `0.4`, the reusable cache may be too small to say much about admission quality
- above `0.8`, the live region may become unrealistically cramped for active decode/prefill state

For each `n`, evaluate feasibility and impact:

- if the live region `L` is too small for stable serving, that point is infeasible
- among feasible points, choose the `n` that gives the best user-facing metrics for `OPT+bypass`

## Metrics

- total miss rate
- compulsory miss rate
- reuse miss rate
- admitted reusable block count
- bypassed reusable block count
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

These must be read from the online serving benchmark outputs.

## Procedure

1. Generate workloads:

```bash
python3 data/generators/generate_headroom_study_workloads.py
python3 benchmarking/datasets/build_headroom_realworld_slices.py \
  --input data/processed/sharegpt_subset.jsonl \
  --dataset-name sharegpt_subset
```

If `LMSYS-Chat-1M` becomes available locally later, rebuild only the near-real slice; the admission
vs eviction experiment structure remains unchanged.

2. Sweep the partition for incoming-line-aware `OPT`:
   - for each `n in {0.4, 0.5, 0.6, 0.7, 0.8}`
   - reserve `L = (1 - n) * M` for live KV state
   - reserve `C = n * M` for the reusable cache
3. For incoming-line-aware `OPT`, the candidate set for an admission decision must include:
   - all current evictable residents
   - the incoming reusable block itself
4. If the incoming block is worst under the oracle objective, bypass it from `C`.
   - the block may still exist in `L` while needed for the active request
5. Choose the best feasible `n` for incoming-line-aware `OPT`.
6. Then run the final comparison panel on the same workload:
   - `LRU` with full reusable budget (`n = 100%`)
   - eviction-only `OPT` with full reusable budget (`n = 100%`)
   - incoming-line-aware `OPT` at the best feasible `n`
7. Plot:
   - miss rate by policy
   - throughput by policy
   - `TTFT` by policy
   - admitted vs bypassed block counts
   - output throughput / request throughput vs `n`
   - median / p99 `TTFT` vs `n`
   - median / p99 `ITL` vs `n`

## Implemented Runner

- [run_opt_with_incoming_line.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_opt_with_incoming_line.py)

Pilot command:

```bash
python3 benchmarking/runners/run_opt_with_incoming_line.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_opt_incoming_line_pilot \
  --mode pilot
```

Full command:

```bash
python3 benchmarking/runners/run_opt_with_incoming_line.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_opt_incoming_line_full \
  --mode full
```

Current implemented sweep:

- workloads:
  - optimistic synthetic incoming-suffix-pollution pattern
  - near-real ShareGPT sequence slice
- page size:
  - `32`
- pilot cache fractions:
  - `0.6`
- full cache fractions:
  - `0.4`, `0.5`, `0.6`, `0.7`, `0.8`
- request rate:
  - `16`
- pilot max concurrency:
  - `16`
- pilot request rate:
  - `8`
- pilot prompt count:
  - `24`
- full max concurrency:
  - `96`
- memory fraction static:
  - `0.24`

What the runner does:

1. for each workload and cache fraction `n`, run `LRU -> OPT+bypass`
2. choose the best `n` by `output_throughput`
3. run a separate `LRU -> OPT` baseline on the same workload
4. compare:
   - `LRU` from the baseline runs
   - eviction-only `OPT`
   - best feasible `OPT+bypass`

Expected runtime:

- pilot: about `0.08 h`
- full: about `2.5 h`

Pilot success criteria:

- per-fraction run roots exist
- a best-fraction file exists:
  - `optimistic__best_fraction.json`
  - `near_real__best_fraction.json`
- the selected best-fraction file records:
  - chosen run root
  - chosen metric
  - chosen value
- the baseline `opt_baseline` run also completes

Pilot note:

- this pilot is only meant to validate the `OPT+bypass` code path and best-`n` selection logic
- use `full` mode for any claim about the value of incoming-line competition

## Why This Answers the Headroom Question

If admission-aware `OPT` barely improves on eviction-only `OPT`, then bypass logic is not a major
missing lever.

If admission-aware `OPT` is much better, then compression and prefetching are not the only promising
directions; admission control and live/cache partitioning are also worth pursuing.
