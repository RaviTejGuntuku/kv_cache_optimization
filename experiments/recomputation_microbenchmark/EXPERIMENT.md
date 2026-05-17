# Recomputation Microbenchmark

## Question

What is the direct cost of recomputing `k` KV-cache blocks instead of reusing them, and how does that cost change with page size `b`?

This is the cleanest experiment for estimating whether misses are intrinsically expensive enough to matter.

## Workloads

Optimistic workload:

- [recompute_k_block_ladder.jsonl](/data/synthetic/headroom_studies/recomputation_microbenchmark/recompute_k_block_ladder.jsonl)

Near-real workload:

- [recomputation_microbenchmark\_\_realworld_sequence.jsonl](/data/processed/headroom_studies/recomputation_microbenchmark/recomputation_microbenchmark__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

Optional future replacement:

- `LMSYS-Chat-1M` slice preserving original request order

Hypothesis:

- `recompute_k_block_ladder` should give the cleanest linear estimate of per-block recompute cost.
- `recomputation_microbenchmark__realworld_sequence` should show lower and noisier penalties because naturally occurring request sequences create more overlap and batching effects than the ladder workload.

## Independent Variables

- page size `b in {16, 32, 64, 128}`
- recomputed block count `k in {1, 2, 4, 8, 16}`
- concurrency regime

## Fixed Controls

- model
- scheduler
- GPU
- prompt family formatting

## Baselines

Even though this is a microbenchmark, still retain:

- compulsory misses
- `LRU`
- `OPT`

For the ladder workload, the key baseline is the fully reused case for the same prompt family.

## Metrics

- extra `TTFT` vs reuse baseline
- extra `ITL` vs reuse baseline
- output throughput change vs reuse baseline
- request throughput change vs reuse baseline
- recomputed blocks
- recomputed tokens
- cost per recomputed block
- cost per recomputed token

## Impact Statistics

The user-facing impact statistics for this experiment are:

- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`

The microbenchmark should report both absolute values and deltas relative to the fully reused path
for the same prompt family.

## Procedure

1. Generate workloads:

```bash
python3 data/generators/generate_headroom_study_workloads.py
python3 benchmarking/datasets/build_headroom_realworld_slices.py \
  --input data/processed/sharegpt_subset.jsonl \
  --dataset-name sharegpt_subset
```

If a local `LMSYS-Chat-1M` request stream is later available, it can replace the ShareGPT slice
for the near-real panel without changing the benchmark structure.

2. For each `b`, run the ladder workload at low concurrency to estimate the clean per-block penalty.
3. Repeat at one higher concurrency point to measure how batching and overlap change the apparent penalty.
4. For each prompt family, compare:
   - reuse path
   - forced recompute path with target `k`
5. Fit:
   - `delta_ttft ~ k`
   - `delta_itl ~ k`
   - `delta_throughput ~ recomputed_tokens`

## Implemented Runner

- [run_recomputation_microbenchmark.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_recomputation_microbenchmark.py)

Pilot command:

```bash
python3 benchmarking/runners/run_recomputation_microbenchmark.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_recompute_pilot \
  --mode pilot
```

Full command:

```bash
python3 benchmarking/runners/run_recomputation_microbenchmark.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --output-root results/experiments/headroom_recompute_full \
  --mode full
```

Current implemented sweep:

- workloads:
  - optimistic synthetic `k`-block ladder
  - near-real ShareGPT sequence slice
- pilot page size:
  - `32`
- full page sizes:
  - `16`, `64`
- pilot sampled prompts per workload:
  - `2`
- full sampled prompts per workload:
  - `8`
- concurrency:
  - `1`
- memory fraction:
  - `0.24`

What the runner does:

1. runs a small `LRU -> OPT` two-pass baseline on the sampled dataset
2. launches an LRU server
3. for each sampled row, runs the same one-row dataset twice:
   - first cold
   - then warm
4. writes `microbench_summary.json` with cold-vs-warm deltas

Expected runtime:

- pilot: about `0.08 h`
- full: about `1.5 h`

Pilot success criteria:

- `microbench_summary.json` exists for each workload/page-size pair
- each summary row contains:
  - `target_recompute_blocks`
  - `cold_median_ttft_ms`
  - `warm_median_ttft_ms`
  - `delta_median_ttft_ms`
- the cold/warm runs complete and produce a nonempty summary

Pilot note:

- this pilot only checks the cold-vs-warm measurement path
- use `full` mode for any block-cost fit or page-size conclusion

## Why This Answers the Headroom Question

If recomputing a block is cheap, then even perfect cache management may not move user-facing metrics much.

If recomputing a block is expensive, then any method that reduces or hides reuse misses may have substantial upside.
