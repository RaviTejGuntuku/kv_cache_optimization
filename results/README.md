# Results Catalog

Each experiment now lives in a single self-contained directory under
[results/experiments](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments).

## Experiment layout

Every experiment directory should contain the same top-level buckets:

- `procedure/`: how the experiment was run
- `inputs/`: workload files or symlinks to the workload files
- `raw/`: raw synced run bundles, traces, benchmark JSONL dumps, plans
- `metrics/`: derived CSV/JSON summaries
- `graphs/`: rendered plots
- `archive/`: redundant or intermediate artifacts kept only for safety

## Most recent experiment

The most recent experiment is:

- [adversarial_fcfs_page_size_20260420](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/adversarial_fcfs_page_size_20260420)

That is the run with:

- `LRU` vs `Belady`
- compulsory misses
- throughput
- median / p99 `TTFT`
- median / p99 `ITL`

Start with:

- [adversarial_fcfs_page_size_20260420/EXPERIMENT.md](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/adversarial_fcfs_page_size_20260420/EXPERIMENT.md)
- [adversarial_fcfs_page_size_20260420/metrics/consolidated_metrics.csv](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/adversarial_fcfs_page_size_20260420/metrics/consolidated_metrics.csv)
- [adversarial_fcfs_page_size_20260420/graphs](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/adversarial_fcfs_page_size_20260420/graphs)

## Canonical experiment folders

- [adversarial_fcfs_page_size_20260420](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/adversarial_fcfs_page_size_20260420)
- [fcfs_static_prefix_panel_20260419](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/fcfs_static_prefix_panel_20260419)
- [fcfs_static_prefix_strong_20260420](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/fcfs_static_prefix_strong_20260420)
- [sharegpt_sweep_rr16_mcgrid_v2](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/sharegpt_sweep_rr16_mcgrid_v2)
- [synth_shared_prefix_sweep](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/synth_shared_prefix_sweep)
- [synth_shared_prefix_sweep_recomputed](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/synth_shared_prefix_sweep_recomputed)
- [synthetic_natural_panel_live](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/results/experiments/synthetic_natural_panel_live)
