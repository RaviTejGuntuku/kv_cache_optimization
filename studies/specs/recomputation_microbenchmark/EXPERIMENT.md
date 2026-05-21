# KV Recovery Crossover Study

## Objective

This study asks a simple systems question:

- if a request needs `k` missing KV-cache blocks, is it faster to:
  - reuse those blocks if they are already in `HBM`
  - fetch those blocks from host `DRAM`
  - fetch those blocks from local `SSD`
  - or recompute those blocks through the real model?

The quantity of interest is:

- `request_latency(method, k)`

where:

- `method ∈ {HBM_reuse, DRAM_fetch, SSD_fetch, recompute}`
- `k` is the number of missing KV blocks
- block size is fixed at **16 tokens**

The main output is the crossover region:

- for what `k` does `DRAM_fetch(k)` become faster than `recompute(k)`?
- for what `k` does `SSD_fetch(k)` become faster than `recompute(k)`?

This tells us when a DRAM KV tier is worth using instead of rebuilding missing blocks on GPU.

## Current Canonical Result Bundle

The current synced full run for this study is:

- [recomputation_microbenchmark_ssd_full__20260521](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/results/recomputation_microbenchmark_ssd_full__20260521)

Key result tables:

- [recovery_times.csv](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/results/recomputation_microbenchmark_ssd_full__20260521/metrics/recovery_times.csv)
- [crossover_points.json](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/results/recomputation_microbenchmark_ssd_full__20260521/metrics/crossover_points.json)

## Why The Synthetic Proxy Was Not Enough

The earlier synthetic tensor microbenchmark was useful as a pipeline smoke test, but not as the canonical experiment.

Reason:

- its recompute path stayed too close to fixed kernel-launch overhead
- the recompute curve did not scale convincingly with `k`
- therefore the measured crossover was not trustworthy

So the canonical experiment should use a **real model execution path**, not a toy compute proxy.

## Canonical Design

This is now a **real-model recovery experiment**.

For one prompt family with a long reusable prefix, we evaluate three regimes:

1. `HBM_reuse`
   - the needed `k` KV blocks are already resident in HBM
   - this is the hit baseline / lower bound

2. `DRAM_fetch`
   - the needed `k` KV blocks have already been computed earlier
   - they are no longer in HBM
   - they are present in host DRAM and must be restored before the request proceeds

3. `SSD_fetch`
   - the needed `k` KV blocks have already been computed earlier
   - they are persisted to local SSD and must be restored through host memory before the request proceeds

4. `recompute`
   - the needed `k` KV blocks are absent from both HBM and DRAM
   - the system must rebuild them from tokens through the model

All three regimes must use:

- the same model
- the same prompt family
- the same output length
- the same server configuration

Only the recovery mode changes.

## Fixed Controls

- model: `Qwen/Qwen2.5-7B-Instruct`
- block size: `16` tokens
- scheduler: `fcfs`
- batch size: `1`
- page size: `16`
- output length: fixed across runs
- request shape: identical except for the number of reusable prefix blocks `k`

Primary batch size:

- **`1`**

Why:

- this is a critical-path recovery study
- batch size `1` gives the clearest per-request latency signal
- larger batches can be used only as a later sensitivity panel

## Workload Construction

Construct a family of prompts such that:

- each prompt has a reusable prefix of exactly `k * 16` tokens
- the suffix after that reusable prefix is held fixed
- the output length is held fixed

For each `k`, create a request whose serving cost depends on recovering exactly `k` reusable blocks.

The prompt family should be synthetic but **served through the real model**.

That is:

- the content can be cooked up
- but the recovery path must go through real LLM serving, not a fake tensor kernel

Current workload interpretation:

- the independent variable is the reusable-prefix length, expressed as `k` KV blocks
- each prompt is a synthetic long-prefix request whose prefix length is exactly `k * 16` tokens
- the suffix and output shape are held fixed so only the recovery mode changes
- the model is real, the KV objects are real, and the restore/recompute path goes through the real model stack

So this is a synthetic prompt family, but not a synthetic compute proxy.

## Independent Variable

The independent variable is:

- `k = number of missing KV blocks`

Use a denser sweep than powers of two.

### Recommended full sweep

- `k = 1, 10, 20, 30, ..., 800`

This gives a wide, uniform sweep across small and large recovery sizes while staying easy to interpret.

### Recommended pilot sweep

- `k ∈ {1, 10, 50, 100, 200, 400, 800}`

If no crossover appears by `k = 800`, report:

- `no crossover observed in [1, 800]`

## Measurement

### Primary metrics

- wall-clock request latency `p50` in milliseconds
- wall-clock request latency `p95` in milliseconds

### Secondary metrics

- wall-clock trimmed mean in milliseconds
- CUDA-event recovery latency `p50` in milliseconds
- CUDA-event recovery latency `p95` in milliseconds

Why:

- wall-clock request latency is the actual systems objective
- `p50` and `p95` are much more stable than raw means for this benchmark
- trimmed mean helps detect whether a crossover is robust or just an outlier artifact
- CUDA-event latency is useful only as a supporting diagnostic

We do **not** report:

- bytes recovered
- effective GB/s

Those can be derived later if needed, but they are not central to the question.

## Procedure

### Phase 1: HBM hit baseline

For each `k`:

1. warm the reusable prefix so that its `k` blocks are present in HBM
2. serve the request
3. record:
   - wall-clock latency
   - `TTFT`
   - CUDA-event latency if available

This is the lower-bound reference.

### Phase 2: DRAM restore

For each `k`:

1. precompute the same reusable prefix
2. ensure the corresponding `k` blocks are resident in host DRAM rather than HBM
3. serve the request and force restoration from DRAM
4. record:
   - wall-clock latency
   - CUDA-event latency if available

This measures the cost of a lower-tier host-memory fetch.

### Phase 3: SSD restore

For each `k`:

1. precompute the same reusable prefix
2. persist the corresponding `k` blocks to local SSD
3. serve the request and force restoration from SSD through host memory
4. record:
   - wall-clock latency
   - CUDA-event latency if available

This measures the cost of a lower-tier SSD-backed restore.

### Phase 4: Recompute

For each `k`:

1. ensure the corresponding `k` blocks are absent
2. serve the same request
3. let the system rebuild the reusable prefix through the actual model path
4. record:
   - wall-clock latency
   - `TTFT`
   - CUDA-event latency if available

This measures the true rebuild cost through the model.

## Interpretation

The curves should be read as:

- `HBM_reuse(k)`: best-case hit baseline
- `DRAM_fetch(k)`: host-memory restore path
- `SSD_fetch(k)`: SSD-backed restore path
- `recompute(k)`: rebuild path

The key question is:

- when does the `DRAM_fetch` curve drop below the `recompute` curve?
- when does the `SSD_fetch` curve drop below the `recompute` curve?

That is the decision boundary for whether a DRAM KV tier is worth using.

## Outputs

### Required tables

- `recovery_times.csv`
  - columns:
    - `phase`
    - `batch_size`
    - `k_blocks`
    - `method`
    - `wall_mean_ms`
    - `wall_trimmed_mean_ms`
    - `wall_p50_ms`
    - `wall_p95_ms`
    - `wall_std_ms`
    - `cuda_mean_ms`
    - `cuda_trimmed_mean_ms`
    - `cuda_p50_ms`
    - `cuda_p95_ms`

- `crossover_points.json`
  - first observed crossover interval
  - refined estimated crossover if refinement is run

### Required graphs

1. wall-clock `p50` latency vs `k`
2. wall-clock `p95` latency vs `k`
3. CUDA-event `p50` latency vs `k`
4. CUDA-event `p95` latency vs `k`

## Success Criterion

The experiment is successful if:

- the `recompute` curve scales meaningfully upward with `k`
- the `DRAM_fetch` curve also scales with `k`
- the relative ordering is stable enough to identify:
  - a crossover interval
  - or the explicit absence of a crossover up to `k = 256`

If recompute remains nearly flat as `k` grows, the experiment is still not measuring the intended recovery path and must be fixed before interpretation.
