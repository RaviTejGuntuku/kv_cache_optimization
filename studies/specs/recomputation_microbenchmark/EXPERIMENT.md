# Recomputation vs Fetch Study

## Objective

The real question is not simply “what is the cost of a cache miss?”

It is:

- for a fixed KV block size `b`
- and for a miss involving `k` consecutive KV blocks
- when is it better to recompute those `k` blocks on GPU
- and when is it better to fetch them from a lower level of the KV memory hierarchy?

In other words, this study should estimate the crossover points between:

- recomputation cost
- fetch-from-`L_i` cost

where:

- `L0`: HBM-resident KV cache
- `L1`: host DRAM / pinned CPU memory
- `L2`: local SSD / NVMe spill
- optional future `L3`: remote KV store over network

The output of this study should be a set of curves whose intersections tell us:

- for small `k`, recomputation may be cheaper than fetching
- beyond some threshold `k*`, fetching from a given lower level is cheaper

That threshold is the systems quantity we care about.

## What We Are Trying To Learn

This experiment is meant to answer:

1. What is the marginal cost of recomputing `k` KV blocks?
2. How does that cost scale with `k`?
3. How does that compare against retrieving the same `k` blocks from host DRAM, SSD, or a slower tier?
4. Which lower-tier cache levels are even worth building, given realistic recomputation costs?

This is a headroom study for:

- CPU-DRAM KV spill
- SSD/NVMe KV spill
- multi-tier KV hierarchies
- prefetching from lower tiers
- selective recompute instead of fetch

It is not a study of eviction policy.

## Key Design Choice

Hold block size fixed.

Reason:

- attention/prefill compute cost is asymptotically proportional to the amount of recomputed sequence
- changing block size and number of blocks at the same time confounds the result
- the clean experiment is to choose one production-relevant block size `b` and only vary `k`

Recommended default:

- `b = 32` tokens per KV block

Optional follow-up:

- rerun the entire study at one second block size, e.g. `b = 64`, only if we later decide that block granularity itself is an important lever

## Memory Hierarchy Model

We will model the KV hierarchy as:

- `L0`: HBM
  - hit cost is effectively the baseline reused path
- `L1`: host DRAM
  - fetch requires host-to-device transfer
- `L2`: local SSD / NVMe
  - fetch requires storage read plus host/device movement
- `Recompute`
  - re-run the forward pass necessary to rebuild the missing KV blocks

The main comparison is:

- `cost_recompute(k)`
- `cost_fetch_L1(k)`
- `cost_fetch_L2(k)`

Potential future extension:

- `cost_fetch_L3(k)` for remote KV cache over network

## Workloads

Optimistic workload:

- [recompute_k_block_ladder.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/headroom_studies/recomputation_microbenchmark/recompute_k_block_ladder.jsonl)

Near-real workload:

- [recomputation_microbenchmark__realworld_sequence.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/processed/headroom_studies/recomputation_microbenchmark/recomputation_microbenchmark__realworld_sequence.jsonl)

Current canonical natural corpus:

- `ShareGPT` slice preserving original request order

Hypothesis:

- the synthetic ladder should give the cleanest estimate of `cost_recompute(k)`
- the near-real workload should be noisier because batching and overlap partially hide recomputation cost
- `L1` DRAM fetch should beat recompute only after some nontrivial `k`
- `L2` SSD fetch may only win for much larger `k`, or may not win at all under online serving constraints

## Independent Variables

- number of missing blocks `k`
- hierarchy level used for recovery:
  - `recompute`
  - `L1_dram_fetch`
  - `L2_nvme_fetch`
- workload:
  - optimistic
  - near-real
- optional concurrency regime:
  - low-concurrency isolated
  - moderate concurrency

## Fixed Controls

- model
- scheduler
- GPU type
- block size `b`
- prompt formatting
- memory fraction for the active run

Recommended default controls:

- scheduler: `fcfs`
- block size: `32`
- concurrency:
  - `1` for the pure micro-cost estimate
  - one moderate point later if we want overlap sensitivity

## Baselines

Retain:

- `L0` HBM-resident reuse baseline
- `LRU -> OPT` two-pass baseline for context
- compulsory misses from the sampled dataset

But the core comparison is:

- warm/HBM reuse
- forced recompute
- forced `L1` fetch
- forced `L2` fetch

## Required Measurement Quantities

For each sampled request and each `k`, measure:

- baseline reused path latency
- recompute path latency
- `L1` fetch path latency
- `L2` fetch path latency

Derived deltas:

- `delta_ttft_recompute(k)`
- `delta_itl_recompute(k)`
- `delta_ttft_L1(k)`
- `delta_itl_L1(k)`
- `delta_ttft_L2(k)`
- `delta_itl_L2(k)`

We should also record:

- estimated bytes fetched from each layer
- effective transfer bandwidth during the fetch
- recomputed tokens
- recomputed blocks
- per-block incremental slope

## User-Facing Impact Statistics

Even though this is a microbenchmark, the study should still report:

- output throughput (`tokens / sec`)
- request throughput (`requests / sec`)
- median / p99 `TTFT`
- median / p99 `ITL`

The main emphasis, though, is on deltas relative to the fully reused path.

## Procedure

### Phase 1: Recompute curve

1. Fix block size `b = 32`.
2. Use the synthetic ladder workload to construct prompts tagged with target missing-block counts:
   - `k in {1, 2, 4, 8, 16, 32, 64}`
3. For each sampled row:
   - run once warm with the KV already resident
   - run once cold so the same `k` blocks must be recomputed
4. Measure:
   - `delta_ttft`
   - `delta_itl`
   - throughput delta
5. Fit:
   - `cost_recompute(k)`

### Phase 2: Lower-tier fetch curve

1. Materialize the same KV payload in lower tiers:
   - `L1`: host DRAM buffer
   - `L2`: local SSD / NVMe file
2. For each sampled row and each `k`:
   - force recovery from `L1`
   - force recovery from `L2`
3. Measure:
   - `cost_fetch_L1(k)`
   - `cost_fetch_L2(k)`

### Phase 3: Crossover analysis

For each hierarchy level:

- plot `cost_recompute(k)` and `cost_fetch_Li(k)` on the same axes
- solve for the first `k` where:
  - `cost_fetch_Li(k) < cost_recompute(k)`

That `k*` is the crossover threshold for that layer.

## Main Graphs

The experiment should produce:

1. `delta_ttft` vs `k`
   - one curve for recompute
   - one for `L1`
   - one for `L2`

2. `delta_itl` vs `k`
   - same overlay

3. throughput degradation vs `k`
   - same overlay

4. effective recovery time vs `k`
   - same overlay

5. crossover summary bar/table:
   - `k*_L1`
   - `k*_L2`

Optional:

6. bytes moved vs recovery time
7. normalized cost per block

## How To Interpret The Results

If `cost_recompute(k)` is below `cost_fetch_L1(k)` for almost all realistic `k`, then:

- host DRAM KV spill is not especially compelling

If `L1` wins after a modest `k`, then:

- DRAM spill plus prefetch/recovery is a serious systems direction

If `L2` only wins at huge `k`, then:

- SSD spill is too slow except for very large prefixes or offline restoration

If recompute dominates both lower tiers, then:

- better eviction may still not matter much
- but selective recomputation becomes the right recovery primitive

## Relationship To The Rest Of The Headroom Study

This experiment complements the other two active headroom studies:

- effective residency sweep:
  - tells us how much more effective capacity helps
- critical-path attribution:
  - tells us how much miss cost is exposed
- this recomputation vs fetch study:
  - tells us how expensive each recovery method is once a miss happens

Together they answer:

- whether lower-tier caches are worth building
- whether recomputation is the right fallback
- when to fetch versus recompute

## What Must Change Relative To The Current Runner

The current runner is still too tied to the older framing.

It should be rewritten so that:

- block size is fixed by default
- `k` spans a much wider range
- the main sweep is over recovery method, not page size
- lower-tier recovery (`L1`, `L2`) is explicitly instrumented
- outputs directly include the crossover curves

The current runner can still be reused as a starting point for the warm-vs-cold recompute branch, but it does not yet implement the full memory-hierarchy experiment described here.
