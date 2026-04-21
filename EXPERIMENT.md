# Strong FCFS Static-Prefix Experiment

This file documents the exact methodology used for the strong Belady-vs-LRU experiment that produced the `fcfs_static_prefix_strong_20260420` results.

## Objective

The goal of this experiment was to test whether an offline Belady-style eviction oracle materially outperforms SGLang's default `LRU` eviction policy under a stronger shared-prefix workload than the earlier FCFS panel.

The main question was:

`If we keep scheduling fixed to FCFS, keep the runtime architecture fixed, and only change the leaf eviction policy from LRU to Belady replay, do we see a meaningful gain?`

## High-Level Setup

Fixed settings:

- Scheduler: `fcfs`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Page size: `16`
- Max concurrency: `128`
- Request rate: `inf`
- `mem_fraction_static=0.24`
- `gpu_kv_capacity_blocks=16000`
- One GPU
- One server process at a time

The experiment used a statically prefix-reordered synthetic workload so that requests from the same reusable family appear close together in the input JSONL, while the runtime scheduler itself remains plain `fcfs`.

## Dataset Used

Base strong dataset:

- [natural_hotset_one_shot_hbm_strong.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/data/synthetic/natural_saturated_strong/natural_hotset_one_shot_hbm_strong.jsonl)

Statically reordered dataset actually used in the run:

- [natural_hotset_one_shot_hbm_strong__static_prefix.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/data/synthetic/natural_saturated_strong/natural_hotset_one_shot_hbm_strong__static_prefix.jsonl)

Manifest:

- [natural_hotset_one_shot_hbm_strong.manifest.json](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/data/synthetic/natural_saturated_strong/natural_hotset_one_shot_hbm_strong.manifest.json)

Strong-workload knobs:

- `num_families = 40`
- `hot_set_size = 8`
- `branches_per_family = 32`
- `rounds = 32`
- `interference_per_round = 40`
- `prefix_tokens = 4096`
- `suffix_tokens = 512`
- `output_len = 512`

This produced:

- `1536` requests
- approximately `320` full-request blocks per request at `page_size=16`
- estimated full-request memory pressure at `mc=128`: `2.56x` relative to `gpu_kv_capacity_blocks=16000`

## Static Prefix Ordering

The reordered file was produced with:

- [reorder_prefix_aware_static.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/datasets/reorder_prefix_aware_static.py)

The point of this transformation is:

- requests from the same reusable family are clustered together
- locality is increased in the input stream
- the runtime scheduler is still `fcfs`
- there is no dynamic prefix-aware queue reordering in this experiment

## Two-Pass Methodology

This experiment uses a two-pass procedure.

### Pass 1: LRU baseline

Run the workload once with:

- scheduler: `fcfs`
- radix eviction policy: `lru`

Outputs written by the run:

- `benchmarks/lru.jsonl`
- `traces/lru.jsonl`

### Belady plan compilation

After the LRU run finishes, compile an offline future-use table from:

- `traces/lru.jsonl`

using:

- [compile_belady_plan.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/analysis_scripts/compile_belady_plan.py)

This plan records, for each block hash appearing in `request_lookup.block_hashes`, the later lookup steps at which that block is requested again.

### Pass 2: Belady replay

Run the exact same workload again with:

- same dataset
- same scheduler: `fcfs`
- same server settings
- radix eviction policy: `belady`

Outputs written by the run:

- `benchmarks/belady.jsonl`
- `traces/belady.jsonl`

At each runtime eviction frontier, the candidate set is the current set of evictable radix leaves. Each candidate is scored by the future use of its leaf's last block, and the farthest-future-use candidate is selected.

## Oracle Definition

The oracle used here is the same one used in the recent FCFS experiments:

- requests are viewed as paths through a block trie
- one logical node corresponds to one KV block
- only leaf blocks are legal eviction targets
- the victim score is the next use of that leaf block in the offline pass-1 lookup trace

Relevant runtime files:

- [radix_cache.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/sglang/python/sglang/srt/mem_cache/radix_cache.py)
- [belady_replay.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/sglang/python/sglang/srt/mem_cache/belady_replay.py)

This is an offline replay oracle for the chosen leaf-block eviction objective under deterministic FCFS. It is not a claim that throughput or latency must improve whenever misses improve.

## Files Used For The Strong Run

Remote run root on the GPU machine:

- `/workspace/kv_cache_research/runs/experiments/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z`

Local synced run root:

- [mc-128__20260420T034251Z](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z)

Files from that run:

- [benchmarks/lru.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/benchmarks/lru.jsonl)
- [benchmarks/belady.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/benchmarks/belady.jsonl)
- [traces/lru.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/traces/lru.jsonl)
- [traces/belady.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/traces/belady.jsonl)
- [plans/belady_plan.json](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/plans/belady_plan.json)
- [run_metadata.json](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/remote_sync/fcfs_static_prefix_strong/natural-hotset-one-shot-hbm-strong/mc-128__20260420T034251Z/run_metadata.json)

## How Cache Hits And Misses Were Computed

For the strong run, the cache comparison reported in chat was based on an offline replay over `request_lookup.block_hashes` from the synced traces.

Definition:

- `hit`: a requested block is already resident in simulated HBM
- `miss`: a requested block is not resident and must be brought in

The replay used:

- HBM capacity = `16000` blocks
- `LRU` simulation on the LRU trace
- `Belady` simulation on the Belady trace

This gives:

- `LRU hits = 828,584`
- `LRU misses = 399,524`
- `Belady hits = 843,843`
- `Belady misses = 380,432`

So the strong run showed:

- miss reduction = `19,092`
- relative miss reduction vs LRU = about `4.78%`

## Serving Results From The Strong Run

From the benchmark JSON outputs:

- `LRU request throughput = 8.8074 req/s`
- `Belady request throughput = 8.7909 req/s`
- `LRU output throughput = 183.8542 tok/s`
- `Belady output throughput = 183.5101 tok/s`
- `LRU median TTFT = 12782.74 ms`
- `Belady median TTFT = 12819.96 ms`
- `LRU median ITL = 20.65 ms`
- `Belady median ITL = 19.96 ms`

Interpretation:

- Belady improved the cache objective on this stronger test
- but did not improve overall serving throughput
- therefore, in this setup, eviction policy alone does not appear to be the dominant systems lever

## Plot Outputs

Plots for the stronger run:

- [fcfs_static_prefix_strong_20260420](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/plots/fcfs_static_prefix_strong_20260420)

Plots for the earlier three-workload FCFS panel:

- [fcfs_static_prefix_synced_20260419](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/plots/fcfs_static_prefix_synced_20260419)
