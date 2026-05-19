# KV Recovery Crossover Study

## Objective

This study is about the recovery path **after a KV miss has already happened**.

For a request that needs `k` missing KV-cache blocks, we want to know:

- how long does recovery take if those `k` blocks are already in lower-tier DRAM?
- how long does recovery take if those `k` blocks must be recomputed on GPU?

The quantity of interest is:

- `recovery_latency(method, k)`

where:

- `method ∈ {HBM_reuse, DRAM_fetch, recompute}`
- `k` is the number of missing KV blocks
- block size is fixed at **16 tokens**

The main output is the crossover point:

- for what `k` does `DRAM_fetch(k)` become faster than `recompute(k)`?

That crossover tells us when a DRAM KV tier is worth using instead of recomputation.

## Recommended Framing

For bang-for-buck, this should be a **systems microbenchmark**, not an end-to-end LLM serving run.

We do **not** need a full model server to answer the first-order question.

The clean experiment is:

1. treat one KV block as a tensor payload of realistic size
2. generate tensor groups for `k` blocks
3. measure:
   - time to touch them when already resident in HBM
   - time to fetch them from pinned host DRAM into HBM
   - time to recompute them on the GPU
4. compare the resulting curves and identify the crossover region

This gives the raw recovery curves directly, without scheduler noise.

## What Counts As A Block

Fix:

- block size = **16 tokens**

Block payload size should be derived from:

- model hidden size
- number of KV heads
- head dimension
- dtype
- number of layers

For a given model and dtype, one KV block has a deterministic byte size:

- `bytes_per_block = 2 * layers * block_tokens * kv_heads * head_dim * bytes_per_element`

The `2` is for keys and values.

The experiment should record the exact block byte size used.

## Memory Hierarchy

For this repo, use the following practical hierarchy:

- `L0`: HBM-resident KV block
  - not an actionable miss-recovery tier
  - serves as the lower-bound baseline
- `L1`: host DRAM / pinned CPU memory
  - realistic first spill tier
- `Recompute`: regenerate the block on GPU

Important note:

- `HBM` is included as a baseline reference curve, not as a decision point
- if a block is already in HBM, that is simply a cache hit

## Why SSD Is Omitted

SSD is intentionally out of scope for the first pass.

Reason:

- the first real systems decision is almost certainly `recompute` vs `DRAM_fetch`
- SSD is more relevant for background staging or very large spills than for online critical-path recovery
- adding SSD now increases engineering complexity without improving the first-order answer

## Independent Variables

- `k`: number of missing blocks
- `method`:
  - `HBM_reuse`
  - `DRAM_fetch`
  - `recompute`
- optional follow-up `batch_regime`
  - primary: `1`
  - secondary sensitivity check: `8`

## Fixed Controls

- block size = `16` tokens
- block byte size formula
- dtype
- device
- tensor layout
- copy method
- pinned-memory configuration
- recovery tensor shape
- measurement harness

## Batch Size Choice

Primary batch size:

- **`1`**

Why:

- the question is about **critical-path request recovery**
- batch size `1` gives the cleanest marginal latency per request
- larger batch sizes would let transfer and compute overlap in ways that obscure the decision boundary

Secondary sensitivity point:

- optional **`8`**

Why:

- if we want one realism check, run a second small panel at batch size `8`
- but the canonical experiment and the headline crossover should come from batch size `1`

We should **not** sweep many batch sizes in the first version.

## Measurement Axes

Primary metric:

- wall-clock recovery latency in milliseconds

Secondary metric:

- CUDA-event recovery latency in milliseconds

Why both wall-clock and CUDA-event time:

- wall-clock time is the user-facing quantity
- CUDA-event time isolates device-side copy/compute cost
- if they disagree materially, that reveals host submission / synchronization overhead

This is **not** a decode-step-indexed experiment. The main question is raw recovery cost by tier for one miss episode.

## Procedure

### Phase 1: Parameterize realistic KV block size

1. Choose a reference model config.
2. Compute `bytes_per_block` for one 16-token KV block.
3. Record this in the manifest.

### Phase 2: Build synthetic block payloads

1. Allocate synthetic KV tensors matching the per-block byte size.
2. Materialize collections of:
   - primary grid covering `k = 1` through `256`
3. Store those payloads in:
   - GPU HBM
   - pinned host DRAM

### Phase 3: Measure HBM baseline

For each `k`:

1. Touch `k` blocks already resident in HBM.
2. Synchronize.
3. Measure:
   - host wall-clock latency
   - CUDA-event latency

This is the best-case lower bound.

### Phase 4: Measure DRAM fetch

For each `k`:

1. Copy `k` blocks from pinned host DRAM to GPU memory.
2. Synchronize the device.
3. Measure:
   - host wall-clock latency
   - CUDA-event latency

### Phase 5: Measure recomputation

For each `k`:

1. Run a synthetic compute kernel that regenerates `k` KV blocks of the same size.
2. Synchronize the device.
3. Measure:
   - host wall-clock latency
   - CUDA-event latency

The compute should approximate the amount of work needed to regenerate the payload, rather than simply writing random bytes.

### Phase 6: Adaptive refinement near the crossover

Use a two-stage sweep.

Stage A: broad-but-dense first pass

- `k = 1..32` at every integer
- `k = 40, 48, 56, ..., 128` in steps of `8`
- `k = 144, 160, 176, ..., 256` in steps of `16`

Stage B: local refinement

If Stage A reveals a crossover interval, for example between `k = 72` and `k = 80`, then run a second pass only in that interval:

- refinement grid: every integer `k` in the first interval where the ordering changes

If no crossover appears by `k = 256`, then the experiment should explicitly report:

- `no crossover observed in [1, 256]`

and the next run can extend the upper limit rather than changing the lower region.

This avoids:

- missing the critical point because the sweep is too coarse
- wasting time on dense measurement far from the crossover

## Outputs

### Required tables

- `recovery_times.csv`
  - columns:
    - `batch_size`
    - `k_blocks`
    - `method`
    - `wall_mean_ms`
    - `wall_p50_ms`
    - `wall_p95_ms`
    - `wall_std_ms`
    - `cuda_mean_ms`
    - `cuda_p50_ms`
    - `cuda_p95_ms`
    - `cuda_std_ms`

- `crossover_points.json`
  - `dram_vs_recompute`
  - coarse interval
  - refined estimate

### Required graphs

1. recovery time vs `k`
   - one line for `HBM_reuse`
   - one for `DRAM_fetch`
   - one for `recompute`

2. crossover summary
   - vertical markers or annotated intersection table

3. optional wall-clock vs CUDA-event comparison
   - to show how much overhead is outside raw device work

## Pilot

Pilot goal:

- validate tensor sizing
- validate timing harness
- validate device synchronization
- validate output schema

Pilot settings:

- `k = {1, 4, 16}`
- `methods = {HBM_reuse, DRAM_fetch, recompute}`
- `batch_size = 1`
- `20` repetitions each

The pilot should finish in a few minutes.

## Full Run

Full settings:

- primary pass:
  - `k = 1..32`
  - `k = 40..128` in steps of `8`
  - `k = 144..256` in steps of `16`
  - `methods = {HBM_reuse, DRAM_fetch, recompute}`
  - `batch_size = 1`
  - `100` repetitions per point
- adaptive refinement:
  - integer `k` only in the first crossover interval
  - `100` repetitions per point
- optional secondary realism check:
  - rerun only the refined crossover interval at `batch_size = 8`

## Interpretation

This experiment answers:

- whether recomputation is cheap enough to prefer over lower-tier fetch for small or moderate misses
- whether a DRAM spill layer is likely worthwhile
- at what miss size the system should switch from recompute to DRAM fetch

If recompute beats DRAM until very large `k`, then:

- elaborate KV spill systems may be lower-value than expected

If DRAM beats recompute quickly, then:

- DRAM-backed KV caching, prefetching, and lower-tier recovery become much more compelling
