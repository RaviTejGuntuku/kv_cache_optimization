# Experiment Specs

This directory contains experiment definitions and procedures, not executed results.

Canonical note:

- `studies/specs/` is the source-of-truth location for experiment designs.
- `experiments/` remains as a compatibility alias.

Each experiment has its own folder with an `EXPERIMENT.md` that defines:

- the question being answered
- optimistic and near-real workloads
- baselines
- metrics
- procedure
- interpretation

Optimistic synthetic workloads live under:

- [datasets/synthetic/headroom_studies](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/synthetic/headroom_studies)

Near-real workloads should be extracted from a naturally occurring request sequence, preserving
the original request order. For the current repo state, the canonical near-real source is the
local `ShareGPT` subset. `LMSYS-Chat-1M` is an acceptable future replacement, but is not required
for these experiment designs. The slicer is:

- [build_headroom_realworld_slices.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/datasets/build_headroom_realworld_slices.py)

Generated near-real slices live under:

- [datasets/processed/headroom_studies](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/datasets/processed/headroom_studies)

Experiments:

- [effective_residency_sweep](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/specs/effective_residency_sweep/EXPERIMENT.md)
- [critical_path_miss_attribution](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/specs/critical_path_miss_attribution/EXPERIMENT.md)
- [recomputation_microbenchmark](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/studies/specs/recomputation_microbenchmark/EXPERIMENT.md)

## Runner Entry Points

These experiment plans are now implemented by the following runners:

- [run_effective_residency_sweep.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_effective_residency_sweep.py)
- [run_critical_path_miss_attribution.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_critical_path_miss_attribution.py)
- [run_recomputation_microbenchmark.py](/Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking/runners/run_recomputation_microbenchmark.py)

Each runner supports:

- `--mode pilot` for a short pipeline-validation run
- `--mode full` for the intended experiment panel
- `--model-path`
- `--output-root`

Recommended model for the current stack:

- `Qwen/Qwen2.5-7B-Instruct`

Recommended execution order:

1. Run all three pilots.
2. Inspect the generated `run_manifest.json`, `reports/`, and benchmark JSONLs.
3. If the pilots look sane, run the three full experiments.

Current estimated wall-clock runtime on one H100-class GPU:

- effective residency sweep full: about `1.75 h`
- critical-path miss attribution full: about `1.25 h`
- recomputation microbenchmark full: about `1.5 h`

Total full-suite estimate:

- about `4.5 h`

Total pilot-suite estimate:

- about `0.32 h`

Pilot mode is a smoke test only:

- it is meant to validate serving, trace capture, plan compilation, analysis, and output layout
- it is not meant to produce publishable or decision-quality measurements
