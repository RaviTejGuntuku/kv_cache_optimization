# KV Cache Belady Benchmark

This repository is now organized around one supported workflow:

- run `SGLang` with `LRU`
- collect a deterministic radix-cache trace
- compile an offline `Belady` plan from that trace
- rerun the same request stream with oracle-guided `Belady`
- compare `LRU` vs `Belady`

The abandoned predictive / ML eviction path has been removed from the benchmark code and from the preserved experiment artifacts.

## Repository Layout

```text
README.md
docs/
  benchmark_plan.pdf
  proposal_rough.pdf
benchmarking/
  analysis_scripts/
  datasets/
  launchers/
  runners/
  setup/
data/
  raw/
  processed/
  synthetic/
    shared_prefix/
    recency_trap/
  generators/
results/
  plots/
runs/
  logs/
  summaries/
  <run directories>
sglang/
```

What each area is for:

- `docs/`: project PDFs and planning docs.
- `benchmarking/analysis_scripts/`: scripts that analyze traces, compare runs, estimate memory pressure, and generate plots.
- `benchmarking/datasets/`: dataset preparation and subset-selection scripts.
- `benchmarking/launchers/`: shell entrypoints that start servers and run the serving benchmark.
- `benchmarking/runners/`: top-level experiment drivers.
- `benchmarking/setup/`: environment setup.
- `data/raw/`: raw source data, currently including the ShareGPT dump.
- `data/processed/`: prepared runnable datasets such as the ShareGPT subset.
- `data/synthetic/shared_prefix/`: synthetic shared-prefix workloads.
- `data/synthetic/recency_trap/`: synthetic workloads designed to expose LRU-vs-OPT gaps.
- `data/generators/`: scripts that generate synthetic datasets.
- `results/plots/`: exported plot bundles and summary views that are meant to be inspected locally.
- `runs/`: canonical experiment artifacts. `runs_remote_sync/` has been removed.

## Supported Experiment

The supported benchmark is a deterministic two-pass experiment.

Run 1:
- serve with `LRU`
- collect the traced lookup stream
- benchmark throughput and latency

Run 2:
- compile an offline `Belady` plan from the Run 1 lookup stream
- rerun the same external request stream
- make oracle-guided leaf-eviction decisions during runtime
- benchmark throughput and latency again

The timing axis used by the oracle and trace analysis is logical lookup progress, not wall-clock time. Wall-clock metrics such as throughput, `TTFT`, and `ITL` are still collected as outputs of the serving benchmark.

## Quick Start

### 1. Set up the environment

```bash
cd ~/kv_cache_research
bash benchmarking/setup/setup_sglang_env.sh
source .venv-sglang/bin/activate
```

### 2. Prepare a ShareGPT subset

Raw ShareGPT should live at:

```text
data/raw/sharegpt_raw.json
```

Build the runnable subset:

```bash
python benchmarking/datasets/select_benchmark_subset.py \
  --dataset-type sharegpt \
  --input data/raw/sharegpt_raw.json \
  --output data/processed/sharegpt_subset.jsonl \
  --target-size 1500 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct
```

### 3. Estimate memory pressure

```bash
python benchmarking/analysis_scripts/estimate_memory_pressure.py \
  --dataset data/processed/sharegpt_subset.jsonl \
  --page-size 16 \
  --gpu-kv-capacity-blocks 16000 \
  --concurrency 64 96 128 160 192 224 256
```

Use this to pick concurrency levels that actually push the KV working set over capacity. If pressure is below 1.0, eviction policy usually will not matter much.

### 4. Run a single two-pass benchmark

```bash
python benchmarking/runners/run_two_pass_benchmark.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/processed/sharegpt_subset.jsonl \
  --output-root runs/debug/sharegpt__smoke \
  --page-size 16 \
  --num-prompts 64 \
  --request-rate 16 \
  --max-concurrency 64 \
  --schedule-policy fcfs \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 16000
```

This creates one run directory with:

- `traces/`
- `plans/`
- `benchmarks/`
- `analysis/` for per-policy trace summaries
- `reports/`
- `run_metadata.json`

### 5. Run a sweep

```bash
python benchmarking/runners/run_benchmark_sweep.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --dataset-path data/processed/sharegpt_subset.jsonl \
  --output-root runs/sweeps/sharegpt__fcfs-sweep \
  --request-rates 16 \
  --max-concurrencies 64 96 128 192 256 \
  --page-size 16 \
  --num-prompts 1024 \
  --schedule-policy fcfs \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 16000 \
  --auto-version
```

### 6. Plot a completed sweep

```bash
python benchmarking/analysis_scripts/plot_benchmark_results.py \
  --sweep-manifest runs/sweeps/sharegpt__fcfs-sweep__<timestamp>/sweep_manifest.json \
  --output-dir results/plots/sharegpt__fcfs-sweep \
  --x-axis max_concurrency
```

## Synthetic Workloads

Synthetic data now lives under `data/synthetic/`.

Shared-prefix workloads:

- `data/synthetic/shared_prefix/`

Recency-trap workloads:

- `data/synthetic/recency_trap/`

To generate a recency-trap workload:

```bash
python data/generators/generate_recency_trap_workloads.py \
  --family hotset-one-shot \
  --output data/synthetic/recency_trap/hotset_one_shot_demo.jsonl \
  --hot-set-size 3 \
  --rounds 4 \
  --interference-per-round 10 \
  --prefix-tokens 512 \
  --suffix-tokens 96 \
  --output-len 64
```

The current recency-trap families are:

- `grouped-baseline`
- `bursty-return`
- `hotset-one-shot`
- `zipf-bursty`
- `adversarial-recency-trap`

## Reading Run Artifacts

Every completed run is self-contained under `runs/<category>/<run_name>/`.

The main files to inspect are:

- `reports/comparison.json`: top-level `LRU` vs `Belady` summary
- `reports/memory_pressure.json`: estimated KV pressure for that run
- `analysis/lru/summary.json`: LRU trace summary
- `analysis/belady/summary.json`: Belady trace summary
- `benchmarks/lru.jsonl`: raw serving results for the LRU pass
- `benchmarks/belady.jsonl`: raw serving results for the Belady pass

## Notes

- `FCFS` remains the default scheduling policy for the benchmark.
- `prefix-coverage` can still be used by passing `--schedule-policy prefix-coverage` to the runners.
- The canonical run tree is `runs/`. There is no longer a separate `runs_remote_sync/`.
- Top-level summary artifacts and plots live under `results/`, with plots directly under `results/plots/` rather than `results/plots/local_plots/`.
- Old predictive / three-policy outputs were removed so the repository only reflects the active Belady-vs-LRU study.
