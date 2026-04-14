# KV Cache Benchmark Scaffold

This directory is the practical scaffold for the `SGLang`-based benchmark.

What is implemented here:

- A traced `SGLang` radix cache that emits JSONL events for node creation, node access, node splitting, lock changes, eviction frontiers, and chosen eviction victims.
- A dataset preparation utility that converts local or Hugging Face data into `SGLang`'s `custom` benchmark format.
- A serving benchmark wrapper around `sglang.bench_serving`.
- An offline trace analyzer that estimates Belady headroom from the collected radix events.

What this gives you today:

- You can run an `LRU` baseline on a real H100 box.
- You can collect benchmark-specific cache traces, separate from workload traces.
- You can measure how often `LRU`'s actual leaf choices differ from the oracle choice on the same eviction frontier.
- You can optionally run a page-level `LRU` vs `Belady` cache simulation over the traced access stream.

Current limitation:

- This is an offline oracle analysis scaffold, not yet a fully online Belady replay inside the SGLang server.
- The trace is rich enough to support the next step, but the current analyzer focuses on frontier-quality comparisons and page-level upper bounds.

## Files

- `prepare_custom_dataset.py`: convert datasets into SGLang `custom` JSONL.
- `setup_sglang_env.sh`: create a Python environment and install the local SGLang checkout.
- `launch_sglang_server.sh`: start SGLang with radix tracing enabled.
- `run_serving_benchmark.sh`: drive SGLang with a custom dataset.
- `analyze_kv_trace.py`: compute trace summaries and Belady-vs-LRU diagnostics.

## Trace events

The traced SGLang runtime now emits JSONL rows with these event types:

- `node_created`
- `node_split`
- `node_access`
- `node_lock`
- `eviction_frontier`
- `node_evicted`

The stable fields for analysis are:

- `seq`: trace-local event sequence number.
- `node_id`: radix-tree node id.
- `parent_id`: parent node id.
- `block_hashes`: page-level stable identifiers for the tokens covered by that node.
- `lock_ref`: whether a node is protected from eviction.
- `priority`: priority propagated through the radix tree.

## Local workflow

Assuming the repo root is this workspace:

```bash
cd /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research
bash benchmarking/setup_sglang_env.sh
```

Prepare a dataset. For a local JSONL file that already has `conversations`:

```bash
source .venv-sglang/bin/activate
python benchmarking/prepare_custom_dataset.py \
  --source-path /path/to/input.jsonl \
  --conversation-field conversations \
  --output data/sharegpt_custom.jsonl
```

Or for a Hugging Face dataset with separate input/output fields:

```bash
source .venv-sglang/bin/activate
python benchmarking/prepare_custom_dataset.py \
  --hf-dataset THUDM/LongBench \
  --hf-split train \
  --input-field input \
  --output-field answers \
  --output data/longbench_custom.jsonl
```

Launch the server:

```bash
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export RUN_LABEL=sharegpt_lru
export TRACE_DIR=$PWD/artifacts/traces
bash benchmarking/launch_sglang_server.sh
```

In another shell, run the serving benchmark:

```bash
source .venv-sglang/bin/activate
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export DATASET_PATH=$PWD/data/sharegpt_custom.jsonl
export NUM_PROMPTS=1000
export REQUEST_RATE=8
export MAX_CONCURRENCY=256
bash benchmarking/run_serving_benchmark.sh
```

Analyze the trace:

```bash
source .venv-sglang/bin/activate
python benchmarking/analyze_kv_trace.py \
  --trace artifacts/traces/sharegpt_lru.jsonl \
  --output-dir artifacts/analysis/sharegpt_lru \
  --block-capacity 20000
```

Outputs:

- `summary.json`: aggregate metrics.
- `frontier_decisions.csv`: one row per traced eviction frontier.

## AWS H100 runbook

Recommended target:

- `p5.48xlarge`
- Run the benchmark on a single GPU for the 8B case.
- Use the other GPUs only for parallel sweeps after the single-GPU path is stable.

### 1. Instance bootstrap

Use an NVIDIA-ready AMI or DLAMI. After SSH:

```bash
git clone <your repo or workspace sync> kv_cache_research
cd kv_cache_research
bash benchmarking/setup_sglang_env.sh
source .venv-sglang/bin/activate
nvidia-smi
```

You want to confirm the instance is actually exposing `H100 80GB` devices before attempting any install debugging.

### 2. Model access

If the model is gated, configure Hugging Face credentials:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

### 3. Start the traced LRU baseline

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export RUN_LABEL=lmsys_lru_c256
export TRACE_DIR=$PWD/artifacts/traces
bash benchmarking/launch_sglang_server.sh
```

Useful knobs you will likely sweep:

- `PORT`
- `PAGE_SIZE`
- `MEM_FRACTION_STATIC`
- extra `sglang.launch_server` flags appended to `launch_sglang_server.sh`

### 4. Run a workload sweep

For each workload and concurrency / request-rate setting:

```bash
source .venv-sglang/bin/activate
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export DATASET_PATH=$PWD/data/lmsys_custom.jsonl
export NUM_PROMPTS=1000
export REQUEST_RATE=4
export MAX_CONCURRENCY=256
export OUTPUT_FILE=$PWD/artifacts/benchmarks/lmsys_c256.jsonl
bash benchmarking/run_serving_benchmark.sh
```

Repeat this across pressure levels. In practice, you will vary:

- `REQUEST_RATE`
- `MAX_CONCURRENCY`
- dataset mix and prompt lengths

### 5. Analyze Belady headroom

```bash
source .venv-sglang/bin/activate
python benchmarking/analyze_kv_trace.py \
  --trace artifacts/traces/lmsys_lru_c256.jsonl \
  --output-dir artifacts/analysis/lmsys_lru_c256 \
  --block-capacity 20000
```

Interpretation:

- `frontier_same_choice_rate` near `1.0` means LRU often matches the oracle on the observed frontier.
- A high `frontier_belady_diff_rate` means there is likely recoverable headroom.
- The page-cache simulation gives an optimistic block-level upper bound on hit-rate improvement at the chosen capacity.

## Recommended next step

Once the frontier-analysis results show a meaningful gap, the next iteration should be:

1. Add a replayable Belady policy implementation to SGLang using this trace format.
2. Re-run the exact same request stream with `Belady` enabled.
3. Compare end-to-end throughput, TTFT, and ITL directly rather than only through trace-derived diagnostics.
