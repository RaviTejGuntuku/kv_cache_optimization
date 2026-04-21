# FCFS Belady-vs-LRU Experiment

## Objective

This experiment answers one narrow question:

`Under deterministic FCFS scheduling, on statically prefix-reordered synthetic workloads that genuinely pressure HBM, how far is SGLang's LRU from a block-level offline Belady oracle?`

The comparison is only between:

- `LRU`
- `Belady replay`

The main claim excludes dynamic prefix-aware scheduling. Under `prefix-coverage`, the scheduler itself depends on cache state, which makes the second pass no longer a clean fixed-trace replay.

## Exact Experimental Regime

### Scheduler

- `fcfs`

### Workloads

We run only the statically reordered datasets:

- `data/synthetic/natural_saturated_reordered/natural_bursty_return_hbm__static_prefix.jsonl`
- `data/synthetic/natural_saturated_reordered/natural_hotset_one_shot_hbm__static_prefix.jsonl`
- `data/synthetic/natural_saturated_reordered/natural_zipf_bursty_hbm__static_prefix.jsonl`

These come from the base synthetic natural-saturation panel, but the request order is rewritten offline so that requests with the same reusable family/prefix are clustered together. That gives us stronger temporal locality while keeping the runtime scheduler itself fixed at `fcfs`.

Pattern intent:

- `natural_bursty_return_hbm__static_prefix`
  - reusable families go hot, get displaced, then reappear later
- `natural_hotset_one_shot_hbm__static_prefix`
  - a reusable hot set competes against a large mostly one-shot background
- `natural_zipf_bursty_hbm__static_prefix`
  - skewed popularity plus bursty arrivals

### Static prefix sharing guarantee

Prefix sharing is ensured statically, not dynamically:

- the workload generator already embeds shared-family structure in metadata
- `benchmarking/datasets/reorder_prefix_aware_static.py` sorts rows by family / phase / branch metadata
- requests from the same reusable family are therefore placed adjacent in the JSONL file
- SGLang still runs plain `fcfs`; there is no runtime prefix-aware queue reordering

So the source of increased sharing is the dataset order, not the scheduler.

### Concurrency

Run one high-pressure point per workload:

- `mc=128`

If we need a stronger pressure point later, we can add one larger `mc`, but the default experiment is a single saturated operating point rather than a broad sweep of weak points.

### Fixed settings

- model: `Qwen/Qwen2.5-7B-Instruct`
- page size: `16`
- request rate: `inf`
- `mem_fraction_static=0.24`
- `gpu_kv_capacity_blocks=16000`
- one GPU
- one model replica

## Two-Pass Procedure

Each workload is run twice.

### Pass 1: LRU baseline

Run SGLang with:

- scheduler: `fcfs`
- eviction: `lru`

Outputs:

- `benchmarks/lru.jsonl`
- `traces/lru.jsonl`

The important trace event is `request_lookup`. For every lookup, it records the full list of requested block hashes for that request at that lookup step.

### Offline Belady plan construction

From `traces/lru.jsonl`, build a table:

- key: block hash
- value: sorted list of lookup steps at which that block is requested

This is implemented in:

- `benchmarking/analysis_scripts/compile_belady_plan.py`

Crucially:

- pass 1 tracks future accesses for **all** requested blocks
- it does not only track leaf blocks

### Pass 2: Belady replay

Run the exact same workload again with:

- same dataset file
- same scheduler: `fcfs`
- same server settings
- eviction policy: `belady`

At each eviction frontier:

- candidate set = current evictable radix leaves only
- each candidate is scored by the next use of that leaf's **last block hash only**
- the leaf whose last block is used farthest in the future is evicted

This matches the block-trie abstraction you specified:

- one logical cached object = one block
- only leaf blocks are valid eviction targets
- evict the leaf block used farthest in the future

The relevant runtime code is:

- [radix_cache.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/sglang/python/sglang/srt/mem_cache/radix_cache.py:1338)
- [belady_replay.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/sglang/python/sglang/srt/mem_cache/belady_replay.py:1)

## Oracle Definition And Correctness

### The abstraction

Use the following mental model:

- trie node = one fixed-size KV block
- a request corresponds to a path of blocks
- only the last block of a currently cached request-path is a legal eviction target

In radix-tree land, a runtime candidate is an evictable leaf node. For that leaf:

- `self._ensure_node_hashes(candidate)` returns the block hashes stored in that leaf node
- the runtime uses only the final hash in that list
- that final hash is the leaf block being scored

### Why pass 1 must track all blocks

Even though the eviction action is only over leaf blocks, the request stream in pass 1 must still log every requested block:

- this gives the full future-access table over the actual workload
- if a block is ever reused anywhere in any later request, that reuse is present in the plan
- the future table is therefore correct for every block that could later become a leaf victim

### Why this is the right FCFS oracle

Under `fcfs`, request order is fixed by the input stream, not by cache coverage. So:

- same JSONL workload
- same arrival order
- same scheduler
- same server configuration

means the second pass is a deterministic replay of the same workload under a different eviction policy.

For the block-level objective above, the replay oracle is exact:

- candidate set is the true runtime leaf frontier
- victim score is the true next use of that candidate leaf block in the pass-1 workload trace
- choosing the farthest-next-use leaf is exactly the block-level Belady rule for that frontier

### What this oracle does not guarantee

It is exact for the chosen cache objective, but that does **not** imply strict dominance on every serving metric.

The clean guarantee is about cache behavior:

- HBM hit count
- HBM miss count
- HBM hit rate
- HBM miss rate
- miss-derived transfer proxies

Serving metrics such as:

- output throughput
- TTFT
- ITL
- p99 latency

are downstream system metrics. They can move around due to batching and runtime effects even when the oracle is cache-optimal under this block-level objective.

## Cache Hit / Miss Accounting

We use a simple definition.

### Cache

- HBM-resident KV blocks

### Data object

- a block hash appearing in `request_lookup.block_hashes`

### Rule

For each requested block:

- `hit`: block already resident in HBM when needed
- `miss`: block not resident in HBM and must be computed or recomputed

Therefore:

- first touch is a miss
- reuse after eviction is also a miss

This is the metric we care about. We are not decomposing misses into compulsory / capacity / conflict buckets.

## Metrics Reported

For each run we report:

- request throughput
- output throughput
- median / p99 end-to-end latency
- median / p99 TTFT
- median / p99 ITL
- HBM hit count
- HBM miss count
- HBM hit rate
- HBM miss rate
- transfer proxy bytes

Interpretation:

- hit/miss and transfer quantify the cache objective directly
- throughput / TTFT / ITL quantify whether that cache improvement matters in practice

## Exact Files And Scripts

Dataset reorder script:

- [reorder_prefix_aware_static.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/datasets/reorder_prefix_aware_static.py:1)

Two-pass runner:

- [run_two_pass_benchmark.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_two_pass_benchmark.py:1)

Belady plan compiler:

- [compile_belady_plan.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/analysis_scripts/compile_belady_plan.py:1)

Trace analyzer:

- [analyze_kv_trace.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/analysis_scripts/analyze_kv_trace.py:1)

## Run Commands

Run one high-pressure FCFS point per statically reordered workload:

```bash
python benchmarking/runners/run_two_pass_benchmark.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/synthetic/natural_saturated_reordered/natural_bursty_return_hbm__static_prefix.jsonl \
  --output-root runs/experiments/fcfs_static_prefix/natural-bursty-return-hbm/mc-128 \
  --page-size 16 \
  --num-prompts 320 \
  --request-rate inf \
  --max-concurrency 128 \
  --schedule-policy fcfs \
  --bench-seed 1 \
  --mem-fraction-static 0.24 \
  --gpu-kv-capacity-blocks 16000
```

Repeat analogously for:

- `data/synthetic/natural_saturated_reordered/natural_hotset_one_shot_hbm__static_prefix.jsonl` with `--num-prompts 416`
- `data/synthetic/natural_saturated_reordered/natural_zipf_bursty_hbm__static_prefix.jsonl` with `--num-prompts 246`

### Static prefix-aware FCFS control

Generate reordered datasets:

```bash
python benchmarking/datasets/reorder_prefix_aware_static.py \
  --input data/synthetic/natural_saturated/natural_bursty_return_hbm.jsonl \
  --output data/synthetic/natural_saturated_reordered/natural_bursty_return_hbm__static_prefix.jsonl
```

Then run the same FCFS setup on the reordered dataset:

```bash
python benchmarking/runners/run_two_pass_benchmark.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/synthetic/natural_saturated_reordered/natural_bursty_return_hbm__static_prefix.jsonl \
  --output-root runs/experiments/fcfs_static_prefix/natural-bursty-return-hbm/mc-128 \
  --page-size 16 \
  --num-prompts 320 \
  --request-rate inf \
  --max-concurrency 128 \
  --schedule-policy fcfs \
  --bench-seed 1 \
  --mem-fraction-static 0.24 \
  --gpu-kv-capacity-blocks 16000
```

## Recommended Validation Checklist

Before trusting the results, verify all of the following:

- both passes completed
- `traces/lru.jsonl` exists
- `traces/belady.jsonl` exists
- `benchmarks/lru.jsonl` exists
- `benchmarks/belady.jsonl` exists
- `plans/belady_plan.json` exists
- the Belady victim selector is using the last block of each evictable leaf candidate
- the offline hit/miss replay over `request_lookup.block_hashes` shows:
  - `Belady misses <= LRU misses`

That final inequality is the central FCFS oracle sanity check.

## Expected Interpretation

If the FCFS experiment shows:

- Belady only slightly better than LRU on misses
- and little or no systems benefit

then KV eviction may not be the primary bottleneck, at least on these workloads.

If it shows:

- materially lower misses
- materially lower transfer volume
- and meaningful serving wins

then that is strong evidence that a better eviction policy is worth pursuing.

The static prefix-aware ordering control helps determine whether the gap is mainly due to:

- poor locality in arrival order
- or genuine weakness of LRU even when locality is made easier to exploit.
