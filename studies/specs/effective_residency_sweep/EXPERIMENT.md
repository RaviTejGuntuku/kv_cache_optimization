# Effective Residency Headroom Study

## Objective

This study asks one question:

- if reusable KV blocks can stay resident in HBM until miss rate approaches the compulsory-miss floor, how much do user-facing serving metrics improve?

This is the clean headroom study for:

- better eviction
- better prefetching
- better compression
- better cache-aware scheduling

The study is **not** about DRAM offload. DRAM recovery is a separate question handled by the recomputation study.

## Why This Workload

Earlier runs on:

- [natural_tenant_rotation_gap.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/adversarial_fcfs/natural_tenant_rotation_gap.jsonl)
- [natural_periodic_refinement_gap.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/adversarial_fcfs/natural_periodic_refinement_gap.jsonl)

showed that:

- some workloads have a low compulsory floor and therefore meaningful reusable-residency headroom
- others have a high compulsory floor and therefore much weaker headroom

Those results motivated a new synthetic workload that is explicitly engineered to make the compulsory floor reachable on an `80 GB` HBM GPU when the reusable-cache slice is large enough.

Primary workload for this study:

- [residency_compulsory_reachable_hotset.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/headroom_studies/effective_residency_sweep/residency_compulsory_reachable_hotset.jsonl)

Why this workload is appropriate:

- it contains a **finite reusable hotset**
- hot families recur many times
- there is no second reusable working set that keeps growing over time
- therefore, once the reusable HBM slice is large enough to hold that hotset, the run is guaranteed to approach the compulsory-miss floor

The manifest for this workload is:

- [residency_compulsory_reachable_hotset.manifest.json](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/headroom_studies/effective_residency_sweep/residency_compulsory_reachable_hotset.manifest.json)

At `page_size = 16`, this workload has:

- `12` reusable hot families
- `32` rounds
- about `360` blocks per full request
- a reusable hotset on the order of `12 * 320 = 3840` shared-prefix blocks

So a reusable-cache sweep through approximately `500 -> 6000` blocks is expected to move from:

- clearly capacity-constrained

to:

- large enough to hold the reusable hotset and therefore approach compulsory misses

## Core Design

The GPU has `80 GB` HBM, but we are **not** sweeping total GPU memory and we are **not** sweeping total KV memory blindly.

Instead, partition KV-resident HBM into:

- `L_live`: fixed live-KV budget for active prefill + decode state
- `C_reuse`: swept reusable-KV budget for cached reusable blocks

We hold `L_live` fixed and only sweep `C_reuse`.

That is the cleanest way to test:

- what happens as reusable KV capacity gets larger

without simultaneously changing:

- live request working space
- request admission stability
- decode feasibility
- scheduler operating regime

This is **not** a bypass cache experiment. Every reusable block is still cache-eligible under normal policy; only the size of the reusable region changes.

## Fixed Serving Setup

Recommended first pass:

- model: `Qwen/Qwen2.5-7B-Instruct`
- scheduler: `fcfs`
- page size: `16`
- request rate: `inf`
- concurrency: `96`
- GPU: `80 GB` HBM
- `mem_fraction_static`: start at `0.70` for `Qwen/Qwen2.5-7B-Instruct` on `H100 80GB`

Keep fixed across all sweep points:

- model
- scheduler
- request rate
- concurrency
- page size
- workload order
- server launch settings other than reusable-KV budget

## Step 1: Calibrate the Fixed Live-KV Budget

Before the sweep, run one high-capacity calibration pass per workload.

Goal:

- determine a safe fixed `L_live` that is large enough for active requests, independent of reusable-cache pressure

Procedure:

1. Launch the server with a large KV budget so reusable pressure is minimal.
2. Run the workload once with the target scheduler and concurrency.
3. Record the peak live-KV residency required by active requests.
4. Set:
   - `L_live = 1.25 x peak_live_kv`
5. Round `L_live` up to block granularity.

Interpretation:

- `L_live` is the fixed safety budget for active requests
- the remaining KV-eligible HBM becomes the maximum reusable-cache budget `C_reuse_max`

## Step 2: Sweep the Reusable KV Budget

After fixing `L_live`, sweep only `C_reuse`.

Recommended full-run sweep points for the primary workload:

- `C_reuse_blocks in {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000}`

Rationale:

- the low-capacity region is sampled coarsely because the curve changes monotonically and steeply there
- the upper region is sampled more densely because that is where the run should start flattening toward the compulsory floor
- those denser points make the headroom curve much more interpretable near saturation

Recommended pilot sweep:

- `C_reuse_blocks in {500, 2000, 4000}`

The live server knob should be set directly via:

- `--max-total-tokens = C_reuse_blocks * page_size`

while keeping:

- `mem_fraction_static`

fixed and large enough that SGLang's profiled token capacity exceeds the largest
requested sweep point.

Important implementation note:

- in SGLang, `--max-total-tokens` is only an **upper bound**
- it cannot increase token capacity above the value profiled from
  `mem_fraction_static`
- therefore, before running the sweep, the runner must perform a preflight launch
  at the largest requested capacity point and verify:
  - profiled token capacity `>= max(C_reuse_blocks) * page_size`
  - profiled `max_req_input_len` exceeds the workload's maximum prompt length

If either condition fails, the study is invalid and must be stopped before the
sweep begins.

If the curve is still far above compulsory at `6000` blocks, then either:

- the live-KV budget is still leaking into the experiment
- or the synthetic workload is mis-sized

because the workload is intended to make the compulsory floor reachable.

The independent variable for plotting should still be:

- `distance_above_compulsory = miss_rate - compulsory_miss_rate`

not raw budget fraction.

## Policies

At each sweep point compute:

- compulsory misses
- `LRU`
- `OPT` / offline Belady

How to use them:

- compulsory = lower bound
- `LRU` = practical baseline
- `OPT` = reference showing best possible eviction-only behavior at the same reusable capacity

The main headroom curve should be the `LRU` curve as reusable capacity grows.

## Metrics

Primary metrics:

- output throughput
- request throughput
- median / p99 `TTFT`
- median / p99 `ITL`

Cache metrics:

- total miss count / miss rate
- compulsory miss count / miss rate
- reuse miss count / miss rate
- `distance_above_compulsory`

## Success Criterion

The study is only complete if the sweep reaches the compulsory-floor regime for at least one tested workload, and ideally for both.

Recommended completion criterion:

- `distance_above_compulsory <= 0.02`

for at least one workload, with the preferred outcome being that both workloads are pushed as close as hardware permits to their compulsory floors.

If that does not happen, the run is not yet a full headroom study. It is only a partial residency curve.

## Procedure

1. Choose one workload.
2. Run the calibration pass and determine `L_live`.
3. Compute `C_reuse_max`.
4. Sweep `C_reuse` from very small to `C_reuse_max`.
5. At each point, run:
   - `LRU`
   - `OPT`
6. Verify that the live serving regime is still stable:
   - no OOM
   - no scheduler instability
   - no live-KV starvation
7. Stop only when:
   - `LRU` approaches the compulsory floor
   - or the final planned reusable-capacity point has been reached

## Graphs

Required plots:

1. output throughput vs `distance_above_compulsory`
2. median `TTFT` vs `distance_above_compulsory`
3. p99 `TTFT` vs `distance_above_compulsory`
4. median `ITL` vs `distance_above_compulsory`
5. p99 `ITL` vs `distance_above_compulsory`
6. miss rate vs reusable-cache fraction
7. compulsory miss rate as a horizontal reference line

Also include one summary table per workload:

- smallest `C_reuse`
- largest `C_reuse`
- closest-to-compulsory point
- absolute and percent deltas for throughput / `TTFT` / `ITL`

## How to Read the Result

### Larger-gain regime

If a workload shows large user-facing gains as `distance_above_compulsory` goes to zero, then there is real headroom in better reusable-residency management for that workload pattern.

### Smaller-gain regime

If a workload improves only slightly near the compulsory floor, then reusable-KV miss reduction is probably not the main systems lever for that workload pattern.

### Historical interpretation

The older `natural_tenant_rotation_gap` and `natural_periodic_refinement_gap` results still matter as motivation:

- they show that headroom is workload-dependent
- they justify constructing a workload where the compulsory floor is definitely reachable
- they should not be confused with the primary workload used in this specific residency-sweep experiment

## Notes

- This study is intentionally about **HBM-resident reusable capacity only**.
- It should not be mixed with DRAM fetch or recomputation policies.
- Queue-information usefulness should be a separate future study.
- The synthetic workload is deliberately cooked so that the compulsory floor is reachable; this is a feature, not a bug, because the whole point is to measure the full headroom curve.

## Implemented Runner

- [run_effective_residency_sweep.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_effective_residency_sweep.py)

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
