# KV Cache Eviction Benchmark

This repository contains a practical benchmark scaffold for measuring how much performance is left on the table by `LRU` KV-cache eviction in `SGLang`, and how much of that gap an oracle-style `Belady` policy could recover.

The current implementation is centered on three things:

- instrumenting `SGLang`'s radix cache so it emits benchmark-grade traces,
- running real serving workloads against that traced runtime,
- analyzing the collected traces offline to estimate the benefit of `Belady` over `LRU`.

This repository now supports an oracle-replay Belady benchmark path for deterministic two-pass experiments. It is benchmark infrastructure, not a deployable online policy.

Current recommended runnable setup:

- primary workload: `ShareGPT`
- excluded from the current runnable path: `LMSYS-Chat-1M`

`LMSYS-Chat-1M` is not part of the current runnable setup because the dataset is gated and adds setup friction without being necessary to validate the benchmark pipeline.

## Quick Start

If you already have:

- an SSH key for a remote NVIDIA GPU box,
- a working Ubuntu-like image with NVIDIA drivers,
- a model you can access,

then this is the shortest path from local machine to completed two-pass benchmark.

### 1. Get the repository onto the GPU machine

If your remote box can access your git remote, prefer `git clone` or `git pull` instead of `scp`.

From the GPU machine:

```bash
git clone <your_repo_url> ~/kv_cache_research
cd ~/kv_cache_research
```

If the repo is already there:

```bash
cd ~/kv_cache_research
git pull
```

Use `scp` only if the remote box cannot reach your git remote. From your local machine:

```bash
KEY=~/.ssh/my_gpu_key.pem
USER=ubuntu
HOST=<public_ip_or_dns>

scp -i "$KEY" -r /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research "$USER@$HOST:~/"
```

### 2. SSH in and create the environment

```bash
ssh -i "$KEY" "$USER@$HOST"
cd ~/kv_cache_research
nvidia-smi
bash benchmarking/setup_sglang_env.sh
source .venv-sglang/bin/activate
```

### 3. Build a workload subset

Example for ShareGPT:

```bash
python benchmarking/select_benchmark_subset.py \
  --dataset-type sharegpt \
  --input data/sharegpt_raw.json \
  --output data/sharegpt_subset.jsonl \
  --target-size 1500 \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

### 4. Estimate memory pressure before running

```bash
python benchmarking/estimate_memory_pressure.py \
  --dataset data/sharegpt_subset.jsonl \
  --page-size 16 \
  --gpu-kv-capacity-blocks 20000 \
  --concurrency 64 128 256 384 512
```

### 5. Run the full experiment

```bash
python benchmarking/run_two_pass_benchmark.py \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-path data/sharegpt_subset.jsonl \
  --output-root runs/sharegpt_run1 \
  --page-size 16 \
  --num-prompts 1000 \
  --request-rate 4 \
  --max-concurrency 256 \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 20000
```

If you want a quick smoke test before renting more GPU time, start with a much smaller run:

```bash
python benchmarking/run_two_pass_benchmark.py \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-path data/sharegpt_subset.jsonl \
  --output-root runs/sharegpt_smoke \
  --page-size 16 \
  --num-prompts 64 \
  --request-rate 2 \
  --max-concurrency 32 \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 20000
```

This smoke test is mainly for pipeline validation:

- does the traced `LRU` run finish,
- does Belady-plan compilation succeed,
- does the replay run start and complete,
- do `comparison.json` and trace summaries appear at the end.

It is not the run you should use for a final conclusion about `Belady` vs `LRU`.

This single command runs:

1. traced `LRU` baseline
2. deterministic serving benchmark
3. Belady-plan compilation
4. oracle-replay `Belady` server
5. deterministic rerun
6. per-run trace analysis
7. consolidated `LRU vs Belady` comparison report
8. memory-pressure report

### 6. Sweep the key knobs

```bash
python benchmarking/run_benchmark_sweep.py \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-path data/sharegpt_subset.jsonl \
  --output-root sweep/sharegpt \
  --request-rates 2 4 8 \
  --max-concurrencies 128 256 384 \
  --page-size 16 \
  --num-prompts 1000 \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 20000
```

### 7. Bring results back to your local machine

If you do not want the raw benchmark outputs in git, copy them back separately. From your local machine:

```bash
scp -i "$KEY" -r "$USER@$HOST:~/kv_cache_research/runs" /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/
scp -i "$KEY" -r "$USER@$HOST:~/kv_cache_research/sweep" /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/
```

## What This Benchmark Measures

The benchmark is designed to answer:

> When the GPU KV cache is under pressure, how often does `LRU` make the wrong eviction choice, and how much room is there for a better policy?

For this benchmark, request scheduling is fixed to `FCFS`.
Scheduling is intentionally not varied here.
The goal is to isolate eviction effects from scheduler effects.

The implementation traces these radix-cache events:

- `node_created`
- `node_split`
- `node_access`
- `node_lock`
- `eviction_frontier`
- `node_evicted`

The most important analysis outputs are:

- how often `LRU` and offline `Belady` would choose different victims on the same eviction frontier,
- which leaves or page-blocks account for the reuse gap,
- an optional page-level cache simulation that compares `LRU` and `Belady` at a fixed block capacity.

## How The Experiment Works

This benchmark is a two-run offline-oracle experiment.

The key idea is:

- `LRU` is the real baseline policy we want to beat.
- `Belady` needs future knowledge, so we cannot run it as a normal online policy.
- because the request stream is deterministic, we can learn the future from one run and use it in a second run.

### Run 1: Baseline `LRU`

In Run 1, the server runs with:

- the fixed `ShareGPT` subset,
- fixed `FCFS` scheduling,
- fixed request-rate / concurrency / seed,
- real `LRU` eviction inside `SGLang`.

During this run, the modified radix cache logs:

- request lookups into the cache,
- node accesses,
- eviction frontiers,
- actual evictions,
- per-request match summaries.

The most important event for the Belady construction is `request_lookup`.
This tells us, for each logical lookup step in the workload, which block hashes are being requested.

From those `request_lookup` events, we build an offline oracle:

- block `A` is requested at lookup steps `3, 9, 20`
- block `B` is requested at lookup steps `4, 5, 18`
- block `C` is requested at lookup steps `7, 25`

That oracle is exactly the future-access information that Belady needs.

### Run 2: Dynamic Offline-Oracle `Belady`

In Run 2, we rerun the exact same workload:

- same dataset,
- same request order,
- same seed,
- same request rate,
- same max concurrency,
- same scheduler,
- same model,
- same hardware.

The only thing that changes is the eviction policy.

At each actual eviction frontier in Run 2, the runtime:

1. looks at the live eviction candidates that exist in the current cache state,
2. asks the offline oracle when each candidate will next be used,
3. evicts the candidate whose next use is farthest in the future.

This is the important point:

- Run 2 does **not** replay a fixed victim sequence from Run 1.
- Run 2 makes Belady's choice dynamically over the live frontier in the second run.

That is what makes this a meaningful `LRU` vs `Belady` comparison.

### Why This Works

Belady is defined in terms of future reuse.
Normally, that future is unavailable online.
In this benchmark, we intentionally fix the request stream, so the future access pattern can be learned from Run 1 and reused in Run 2.

The cache state in Run 2 is allowed to differ from Run 1.
That is expected.
If Belady is better, it should produce a different cache state.

What is held fixed between the two runs is:

- workload,
- scheduler,
- arrival process,
- model,
- hardware.

So the comparison isolates the effect of eviction quality.

### Tiny Example

Suppose the deterministic workload induces these logical cache lookups:

- lookup `0`: blocks `A, B`
- lookup `1`: blocks `B, C`
- lookup `2`: blocks `D`
- lookup `3`: blocks `A`
- lookup `4`: blocks `E`
- lookup `5`: blocks `B`

From Run 1, the oracle becomes:

- `A -> [0, 3]`
- `B -> [0, 1, 5]`
- `C -> [1]`
- `D -> [2]`
- `E -> [4]`

Now imagine Run 2 reaches an eviction frontier after lookup step `2`, and the live candidates are:

- candidate ending in `A`
- candidate ending in `B`
- candidate ending in `C`

The oracle says:

- next use of `A` after step `2` is `3`
- next use of `B` after step `2` is `5`
- next use of `C` after step `2` is never

So Belady evicts `C`, because it is needed farthest in the future.
`LRU` could make a different choice based only on recency.

### What We Compare At The End

For both runs we collect:

- throughput,
- median / p99 `TTFT`,
- median / p99 `ITL`,
- server-reported cache hit rate if available,
- trace-derived token and block hit / miss rates.

The final comparison is therefore:

- `LRU` on the real workload
- versus dynamic offline-oracle `Belady` on the exact same real workload

## Hardware Guidance

### What to use from the Lambda options you showed

From your screenshot, these are the relevant options:

- `1x A10 (24 GB PCIe)`: not recommended for the main benchmark. `24 GB` is too tight for the intended `Llama-3.1 8B` serving setup plus meaningful KV-cache pressure.
- `1x A100 (40 GB SXM4)`: usable for small-scale dry runs, trace validation, code debugging, and reduced-pressure experiments.
- `8x A100 (40 GB SXM4)`: good if you want parallel sweeps, but overkill for the first pass if you are still validating correctness.
- `8x Tesla V100 (16 GB)`: not recommended for this benchmark.

If you stay on Lambda for now:

- Use `1x A100 40 GB` for validation and tooling bring-up.
- Move to `H100 80 GB` once you want the benchmark to match the original plan more closely.

### Minimum practical recommendation

For the benchmark as currently designed:

- Preferred: `1x H100 80 GB`
- Acceptable for dry runs: `1x A100 40 GB`
- Avoid: `A10 24 GB`, `V100 16 GB`

The reason is simple: the benchmark only becomes meaningful when model weights fit comfortably enough that KV pressure, not model loading, is the bottleneck.

## Repository Layout

Relevant files:

- `benchmarking/setup_sglang_env.sh`
- `benchmarking/launch_sglang_server.sh`
- `benchmarking/launch_belady_server.sh`
- `benchmarking/run_serving_benchmark.sh`
- `benchmarking/run_two_pass_benchmark.py`
- `benchmarking/run_benchmark_sweep.py`
- `benchmarking/select_benchmark_subset.py`
- `benchmarking/prepare_custom_dataset.py`
- `benchmarking/compile_belady_plan.py`
- `benchmarking/compare_benchmark_runs.py`
- `benchmarking/estimate_memory_pressure.py`
- `benchmarking/analyze_kv_trace.py`
- `benchmarking/README.md`

## Subset Construction Policy

You should not run the full raw datasets as the first benchmark pass.

The right goal is:

- preserve the access-pattern property that makes the workload scientifically useful,
- reach sustained KV pressure,
- collect enough eviction events for `LRU` vs `Belady` to stabilize.

This repository now includes `benchmarking/select_benchmark_subset.py` for that purpose.

### ShareGPT policy

What matters:

- realistic heavy-tailed prompt lengths,
- mixed short and long conversations,
- enough longer prompts to create real cache pressure.

Sampling policy:

- bucket requests by prompt length,
- preserve the short/medium mass,
- explicitly reserve budget for long and extra-long prompts so the heavy tail survives sampling.

Recommended first-pass subset size:

- `1000-2000` requests

### Example subset commands

ShareGPT:

```bash
source .venv-sglang/bin/activate
python benchmarking/select_benchmark_subset.py \
  --dataset-type sharegpt \
  --input data/sharegpt_raw.json \
  --output data/sharegpt_subset.jsonl \
  --target-size 1500 \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

Each run also writes a manifest JSON next to the subset output so you can inspect:

- input size,
- selected size,
- prompt length distribution,
- shared-prefix coverage,
- top selected prefix groups.

## Experiment Knobs

The main knobs for this benchmark are:

- `page_size` / block size: the number of tokens per cached page.
- `request_rate`: controls how quickly requests arrive.
- `max_concurrency`: controls how many requests can be active at once.
- `num_prompts`: controls total benchmark length and statistical stability.
- `mem_fraction_static`: affects how much HBM remains for KV cache after model/runtime allocation.
- dataset choice and subset policy: determines access structure and context-length distribution.
- prompt/output length distribution: directly drives KV working-set size.
- GPU KV capacity in blocks: the denominator for memory-pressure estimates.
- scheduler policy: fixed here to `FCFS`.

The most important derived quantity is:

- memory pressure, roughly `working_set_blocks / gpu_kv_capacity_blocks`

Use `benchmarking/estimate_memory_pressure.py` before a run to sanity-check whether a subset and concurrency setting are likely to exceed the KV budget.

## Full Workflow Notes

The instructions below assume:

- your local machine is your Mac,
- the remote machine is a Linux NVIDIA GPU instance,
- you have SSH access via a private key,
- the remote image already has NVIDIA drivers working.

### 1. Launch the remote instance

Choose one of:

- Lambda: `1x A100 40 GB` for dry runs or `1x H100 80 GB` if available.
- RunPod: `1x H100 PCIe`, `1x H100 SXM`, or `1x A100`.
- CoreWeave / Fluidstack / Nebius: any single-GPU `H100 80 GB` or `A100 40/80 GB` instance.

Use an Ubuntu `22.04` or similar CUDA-ready image when possible.

### 2. Test SSH locally

Replace these values:

- `KEY=~/.ssh/my_gpu_key.pem`
- `USER=ubuntu`
- `HOST=<public_ip_or_dns>`

Run:

```bash
ssh -i "$KEY" "$USER@$HOST"
```

If that works, continue.

### 3. Copy this repository to the remote machine

From your local machine:

```bash
KEY=~/.ssh/my_gpu_key.pem
USER=ubuntu
HOST=<public_ip_or_dns>

scp -i "$KEY" -r /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research "$USER@$HOST:~/"
```

That copies the full workspace to:

```bash
~/kv_cache_research
```

If you later want to update the remote copy after local edits, use:

```bash
scp -i "$KEY" -r /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/benchmarking "$USER@$HOST:~/kv_cache_research/"
```

For repeated syncs, `rsync` is better than `scp`, but `scp` is enough to get started.

### 4. SSH into the instance

```bash
ssh -i "$KEY" "$USER@$HOST"
```

Then:

```bash
cd ~/kv_cache_research
```

### 5. Sanity-check the GPU environment

On the remote machine:

```bash
nvidia-smi
python3 --version
```

You want to see the expected NVIDIA GPU and a working Python installation.

### 6. Create the Python environment

On the remote machine:

```bash
cd ~/kv_cache_research
bash benchmarking/setup_sglang_env.sh
source .venv-sglang/bin/activate
```

This script creates a virtualenv and installs the local `SGLang` checkout in editable mode.

### 7. Log into Hugging Face if needed

If you are using gated models:

```bash
export HF_TOKEN=<your_token>
huggingface-cli login --token "$HF_TOKEN"
```

### 8. Prepare a workload dataset

If you already have a local JSONL with conversations:

```bash
source .venv-sglang/bin/activate
python benchmarking/prepare_custom_dataset.py \
  --source-path /path/to/your/input.jsonl \
  --conversation-field conversations \
  --output data/workload_custom.jsonl
```

If you want to convert a Hugging Face dataset:

```bash
source .venv-sglang/bin/activate
python benchmarking/prepare_custom_dataset.py \
  --hf-dataset <dataset_name> \
  --hf-split train \
  --input-field input \
  --output-field output \
  --output data/workload_custom.jsonl
```

For datasets where the response field is a list, the converter takes the first usable string.

If you want a benchmark-grade subset rather than a straight conversion, use:

```bash
source .venv-sglang/bin/activate
python benchmarking/select_benchmark_subset.py \
  --dataset-type sharegpt \
  --input data/sharegpt_raw.json \
  --output data/sharegpt_subset.jsonl \
  --target-size 1500 \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

### 9. Full end-to-end run with one command

Once your subset file exists, the easiest path is:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

python benchmarking/run_two_pass_benchmark.py \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-path data/sharegpt_subset.jsonl \
  --output-root runs/sharegpt_run1 \
  --page-size 16 \
  --num-prompts 1000 \
  --request-rate 4 \
  --max-concurrency 256 \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 20000
```

This runs:

1. traced `LRU` server launch
2. deterministic serving benchmark
3. Belady plan compilation
4. Belady replay server launch
5. deterministic replay benchmark
6. trace analysis for both runs
7. consolidated comparison report
8. memory-pressure estimate for the dataset and block size

The key outputs will land under:

- `runs/sharegpt_run1/benchmarks`
- `runs/sharegpt_run1/traces`
- `runs/sharegpt_run1/analysis`
- `runs/sharegpt_run1/reports/comparison.json`
- `runs/sharegpt_run1/reports/memory_pressure.json`

### 10. Manual step-by-step run

If you prefer to control each phase manually:

### 10a. Start the traced LRU baseline server

On the remote machine:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export RUN_LABEL=baseline_lru_run1
export TRACE_DIR=$PWD/artifacts/traces

bash benchmarking/launch_sglang_server.sh
```

What this does:

- launches `SGLang`,
- forces `LRU` eviction,
- fixes scheduling to `FCFS`,
- enables the benchmark trace logger through environment variables,
- writes the trace to:

```bash
artifacts/traces/baseline_lru_run1.jsonl
```

### 10b. Run the serving benchmark

Open a second SSH session to the same instance and run:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export DATASET_PATH=$PWD/data/workload_custom.jsonl
export NUM_PROMPTS=1000
export REQUEST_RATE=4
export MAX_CONCURRENCY=256
export OUTPUT_FILE=$PWD/artifacts/benchmarks/run1.jsonl

bash benchmarking/run_serving_benchmark.sh
```

You should sweep:

- `REQUEST_RATE`
- `MAX_CONCURRENCY`
- workload type
- prompt length regime

Those are your main pressure knobs on a single GPU.

### 10c. Compile the Belady replay plan

After the baseline run:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

python benchmarking/compile_belady_plan.py \
  --trace artifacts/traces/baseline_lru_run1.jsonl \
  --output artifacts/plans/baseline_lru_run1_belady.json
```

### 10d. Launch the Belady replay server

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export RUN_LABEL=belady_replay_run1
export TRACE_DIR=$PWD/artifacts/traces
export BELADY_PLAN_PATH=$PWD/artifacts/plans/baseline_lru_run1_belady.json

bash benchmarking/launch_belady_server.sh
```

Run the exact same workload again with the same `BENCH_SEED`, `REQUEST_RATE`, `MAX_CONCURRENCY`, and dataset.

### 10e. Analyze the traces offline on the remote machine

After the runs:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

python benchmarking/analyze_kv_trace.py \
  --trace artifacts/traces/baseline_lru_run1.jsonl \
  --output-dir artifacts/analysis/baseline_lru_run1 \
  --block-capacity 20000

python benchmarking/analyze_kv_trace.py \
  --trace artifacts/traces/belady_replay_run1.jsonl \
  --output-dir artifacts/analysis/belady_replay_run1 \
  --block-capacity 20000
```

This produces:

- `artifacts/analysis/baseline_lru_run1/summary.json`
- `artifacts/analysis/baseline_lru_run1/frontier_decisions.csv`
- `artifacts/analysis/belady_replay_run1/summary.json`
- `artifacts/analysis/belady_replay_run1/frontier_decisions.csv`

### 11. Sweep the knobs

To sweep request rate and concurrency:

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

python benchmarking/run_benchmark_sweep.py \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-path data/sharegpt_subset.jsonl \
  --output-root sweep/sharegpt \
  --request-rates 2 4 8 \
  --max-concurrencies 128 256 384 \
  --page-size 16 \
  --num-prompts 1000 \
  --bench-seed 1 \
  --gpu-kv-capacity-blocks 20000
```

This writes one run directory per knob setting plus `sweep_manifest.json`.

Then aggregate and plot the results:

```bash
pip install matplotlib
python benchmarking/plot_benchmark_results.py \
  --sweep-manifest sweep/sharegpt/sweep_manifest.json \
  --output-dir sweep/sharegpt/plots \
  --x-axis memory_pressure
```

This produces:

- `aggregated_metrics.csv`
- `aggregated_metrics.json`
- `summary.json`
- `throughput_vs_memory_pressure_full_request.png`
- `median_ttft_ms_vs_memory_pressure_full_request.png`
- `p99_ttft_ms_vs_memory_pressure_full_request.png`
- `median_itl_ms_vs_memory_pressure_full_request.png`
- `p99_itl_ms_vs_memory_pressure_full_request.png`
- `block_miss_rate_vs_memory_pressure_full_request.png`
- `transfer_proxy_bytes_vs_memory_pressure_full_request.png`

### 12. Estimate memory pressure directly

```bash
cd ~/kv_cache_research
source .venv-sglang/bin/activate

python benchmarking/estimate_memory_pressure.py \
  --dataset data/sharegpt_subset.jsonl \
  --page-size 16 \
  --gpu-kv-capacity-blocks 20000 \
  --concurrency 64 128 256 384 512
```

## Repository Status

### What is working now

The repository is at the stage where you can collect meaningful `LRU` baseline data on a GPU.

Specifically, it already supports:

- running a local `SGLang` checkout,
- traced radix-cache access and eviction logging,
- benchmark subset construction for the ShareGPT workload,
- Belady-plan compilation from a deterministic baseline trace,
- oracle-replay victim selection inside SGLang,
- serving benchmarks against selected subsets,
- offline `LRU` vs oracle-style `Belady` analysis from the collected trace,
- one-command two-pass orchestration,
- multi-point knob sweeps,
- memory-pressure estimation and consolidated report generation.

### What that means in practice

You can do this now:

1. select a ShareGPT subset,
2. launch the traced `LRU` server,
3. drive requests through the server,
4. compile a Belady replay plan from the baseline trace,
5. relaunch the server with oracle-replay Belady,
6. rerun the exact same request stream,
7. compare traces and serving metrics.
8. estimate memory pressure and repeat across knob settings.

### What is still missing

The repository is now at the stage where it can run an oracle-replay `Belady` victim sequence inside the runtime.

Secondary missing pieces:

- workload-specific subset tuning after the first real data pass,
- repeated runs / confidence intervals,
- a cleaner automation layer for pressure sweeps.

### Bottom-line status

- Ready now for: deterministic `LRU` baseline collection, Belady-plan compilation, oracle-replay reruns, metric comparison across the two runs, and sweep automation.
- Not yet polished for: large automated sweeps or paper-quality repeatability.

## GPU Recommendations

If you want runs to land in roughly the `15-30 minute` range for a useful single benchmark point, these are the GPUs to watch for.

### Best targets

- `H100 80 GB`: best overall target for the benchmark and the overall project.
- `H200`: also excellent if available.
- `A100 80 GB`: very strong fallback if H100 is unavailable.

### Good bring-up / validation target

- `A100 40 GB`: good for bring-up, debugging, and reduced-scale real experiments.

### Acceptable but not ideal

- `L40S 48 GB`: usable for engineering and some smaller-scale experiments, but not my preferred hardware for the main result.
- `MI300X`: technically viable if you intentionally go down the ROCm path, but not the easiest path for this project.

### Avoid for the main benchmark

- `A10 24 GB`
- `V100 16 GB`

Those are too constrained for the intended memory-pressure regime.

### 13. Copy results back to your local machine

From your local machine:

```bash
KEY=~/.ssh/my_gpu_key.pem
USER=ubuntu
HOST=<public_ip_or_dns>

scp -i "$KEY" -r "$USER@$HOST:~/kv_cache_research/artifacts" /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/
```

If you only want the analysis output:

```bash
scp -i "$KEY" -r "$USER@$HOST:~/kv_cache_research/artifacts/analysis" /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research/
```

## Recommended Experiment Sequence

If you want the shortest path to useful signal:

1. Bring up the stack on `1x A100 40 GB`.
2. Run a small synthetic or custom workload to confirm traces are generated correctly.
3. Run one real workload at moderate concurrency.
4. Inspect `summary.json` and `frontier_decisions.csv`.
5. Move to `1x H100 80 GB` once the pipeline is stable.
6. Sweep concurrency or request rate to find the pressure point where `LRU` diverges from the oracle.

## Interpreting the Results

The most useful signals right now are:

- `frontier_same_choice_rate`
- `frontier_belady_diff_rate`
- page-level `lru_hit_rate` vs `belady_hit_rate`

Interpretation:

- If `frontier_same_choice_rate` stays very high, `LRU` is already close to oracle behavior in that regime.
- If `frontier_belady_diff_rate` rises under pressure, eviction quality likely matters.
- If the page-level Belady simulation shows a real hit-rate gap, you have evidence that better eviction policy can plausibly improve end-to-end serving.

## What This Does Not Yet Do

This benchmark scaffold does **not** yet:

- run an oracle-replay Belady eviction policy inside SGLang,
- produce final end-to-end `Belady` TTFT / ITL / throughput measurements,
- handle every dataset automatically without minor schema adaptation.

It does give you the instrumentation and analysis needed to justify that next step.
