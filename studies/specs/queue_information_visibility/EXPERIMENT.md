# Queue Information Visibility Study

## Objective

This study asks one simple headroom question:

- if the controller can see the next `Q` requests in the queue, what fraction of future reuse is already visible?

This is **not** a policy experiment.

It does **not** implement:

- queue-aware eviction
- queue-aware prefetching
- queue-aware scheduling

Instead, it measures how informative the queue is as a source of future-reuse information.

That makes it a clean headroom study for future directions such as:

- queue-aware prefetching
- queue-aware admission / retention
- queue-aware scheduling

## Core Idea

At request position `i`, define the visible queue as:

- the next `Q` requests after position `i`

Then ask:

- how much of the future reuse associated with request `i` is already visible in those next `Q` requests?

Sweep `Q` from very small to large, and observe how quickly queue visibility saturates.

## What Counts As Reuse

Use the same block-level view as the rest of the repo:

- one trie node corresponds to one KV block
- a reuse occurs when a future request uses a KV block that was previously computed by an earlier request

Important:

- count reused blocks along the reused prefix path
- do **not** reduce this to only the leaf request object

So the study is fundamentally about:

- future reuse events
- future reused KV-block volume

## Workloads

Use the two existing synthetic workloads:

- [natural_tenant_rotation_gap.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/adversarial_fcfs/natural_tenant_rotation_gap.jsonl)
- [natural_periodic_refinement_gap.jsonl](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/adversarial_fcfs/natural_periodic_refinement_gap.jsonl)

Why:

- they already produce very different reuse structures
- they are already familiar from the earlier residency / eviction studies
- they let us see whether queue visibility is workload-dependent

This experiment is offline and trace-like, so we do not need to involve a live serving loop in the first implementation.

## Independent Variable

Primary independent variable:

- `Q = visible queue size`

Recommended sweep:

- `Q in {1, 10, 20, 30, ..., 800, full}`

Interpretation:

- `Q = 1`: only the next queued request is visible
- `Q = full`: the entire future queue is visible

Why this is denser than a powers-of-two sweep:

- the purpose of this study is to see how quickly queue information saturates
- a coarse geometric sweep hides where that saturation actually starts
- the denser `10`-request increments make it much easier to answer:
  - “how large does the queue need to be before most reuse is already visible?”

## What Not To Vary

Do **not** make the primary x-axis:

- request rate
- max queue size
- scheduler knobs

Those only indirectly affect queue visibility by producing different realized queue sizes.

The clean experiment is to vary queue size `Q` directly.

If needed later, a separate calibration study can map:

- request rate / offered load

to:

- typical realized queue sizes

But that is not the main experiment.

## Metrics

### 1. Reuse Event Visibility Fraction

Definition:

- among all future reuse events, what fraction are already visible within the next `Q` requests?

Formula:

- `reuse_event_visibility_fraction(Q) = visible_reuse_events(Q) / total_reuse_events`

This is the cleanest event-level metric.

### 2. Reuse Block Visibility Fraction

Definition:

- among all future reused KV blocks, what fraction are already visible within the next `Q` requests?

Formula:

- `reuse_block_visibility_fraction(Q) = visible_reuse_blocks(Q) / total_reuse_blocks`

This is likely the most important metric, because it reflects reusable KV mass rather than just counting reuse events equally.

### 3. Next-Reuse Visibility Fraction

Definition:

- for each reused block, is its **next** reuse already visible within the next `Q` requests?

Formula:

- `next_reuse_visibility_fraction(Q) = visible_next_reuses(Q) / total_next_reuses`

This is a stricter and more actionable metric than generic future reuse.

## Exact Visibility Logic

For each request position `i`:

1. Identify the set of KV blocks that request `i` computes and that are later reused.
2. Look at the next `Q` requests:
   - positions `i+1` through `i+Q`
3. Mark a future reuse as visible if the corresponding future reuser appears in that lookahead window.

Apply this separately for:

- reuse events
- reused blocks
- next reuse

The `full` horizon means:

- all future requests in the stream are visible

This serves as the upper bound.

## Recommended Procedure

1. Load one workload in its fixed order.
2. Tokenize / blockize it with the chosen page size.
3. For each request position `i`:
   - determine future reuses induced by that request’s blocks
4. For each queue horizon `Q`:
   - measure which of those future reuses are visible
5. Aggregate over the full workload.
6. Repeat for the second workload.

Recommended fixed settings:

- scheduler assumption: `fcfs`
- page size: `16`

The scheduler is fixed only to define the queue order cleanly. This is not a scheduling experiment.

## Graphs

Required plots:

1. `Q` vs reuse-event visibility fraction
2. `Q` vs reuse-block visibility fraction
3. `Q` vs next-reuse visibility fraction

Recommended additional plot:

4. cumulative saturation plot showing how quickly visibility plateaus as `Q` grows

## Tables

Required table per workload:

- one row per `Q`
- columns:
  - `queue_size_q`
  - `reuse_event_visibility_fraction`
  - `reuse_block_visibility_fraction`
  - `next_reuse_visibility_fraction`

Recommended summary table:

- smallest `Q` achieving:
  - `50%`
  - `75%`
  - `90%`
  - `95%`

for each of the three visibility metrics

## Interpretation

### If visibility saturates at very small `Q`

Then the queue already exposes most useful future reuse information with only shallow lookahead.

That would strongly motivate:

- queue-aware prefetching
- queue-aware retention
- queue-aware request ordering

### If visibility grows slowly and needs a very large `Q`

Then useful reuse information is diffuse in the future queue, and queue-aware methods may be harder to exploit in practice.

### If the two workloads diverge substantially

Then queue usefulness is strongly workload-dependent and should be discussed in terms of request-pattern structure rather than as a universal statement.

## Scope Boundaries

This study intentionally does **not** answer:

- what is the best queue-aware policy?
- how much throughput gain a specific queue-aware mechanism will produce?

It only answers:

- how much future reuse is already visible in the queue?

That is why it is valuable as a headroom study.

## Suggested Output Location

- `studies/results/queue_information_visibility_<timestamp>/`

with:

- `procedure/`
- `inputs/`
- `metrics/`
- `graphs/`

## Implemented Runner

- [run_queue_information_visibility.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_queue_information_visibility.py)

This is intentionally an offline analysis runner, not a live serving benchmark.

## Pilot Command

```bash
python3 benchmarking/runners/run_queue_information_visibility.py \
  --output-root studies/results/queue_information_visibility_pilot \
  --mode pilot
```

Pilot settings:

- workloads:
  - `natural_tenant_rotation_gap`
  - `natural_periodic_refinement_gap`
- `page_size = 16`
- `Q in {1, 4, 16, full}`
- first `64` requests from each workload

## Full Command

```bash
python3 benchmarking/runners/run_queue_information_visibility.py \
  --output-root studies/results/queue_information_visibility_full \
  --mode full
```

Full settings:

- workloads:
  - `natural_tenant_rotation_gap`
  - `natural_periodic_refinement_gap`
- `page_size = 16`
- `Q in {1, 2, 4, 8, 16, 32, 64, 128, full}`
- full workload length
