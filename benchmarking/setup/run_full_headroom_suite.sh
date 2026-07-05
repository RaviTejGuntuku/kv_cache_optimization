#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-empirical-headroom}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PREFIX="${MODEL_PREFIX:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_BROAD="${MODEL_BROAD:-Qwen/Qwen2.5-7B-Instruct}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/studies/results/full_headroom_suite}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  exit 1
fi

source "$VENV_DIR/bin/activate"

mkdir -p "$OUTPUT_ROOT"

echo "[1/4] Oracle 0 + marginal: prefix"
SYSTEM="vllm_apc" \
MODEL="$MODEL_PREFIX" \
BUNDLE_ROOT="datasets/processed/empirical_headroom/shared_prefix_64x16" \
OUTPUT_ROOT="$OUTPUT_ROOT/prefix_exact_main" \
REQUEST_LIMIT="${PREFIX_REQUEST_LIMIT:-96}" \
MAX_COUNTERFACTUALS_PER_REQUEST="${PREFIX_MAX_COUNTERFACTUALS:-1}" \
MAX_MODEL_LEN="${PREFIX_MAX_MODEL_LEN:-4096}" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
WARMUP=1 \
REPEAT_COUNT=1 \
ISOLATED=0 \
"$ROOT_DIR/benchmarking/setup/run_empirical_headroom_pilot.sh"

echo "[2/4] Oracle 0 + marginal: mixed reusable objects"
SYSTEM="lmcache_cacheblend" \
MODEL="$MODEL_BROAD" \
BUNDLE_ROOT="datasets/processed/empirical_headroom/mixed_reuse_1024req" \
OUTPUT_ROOT="$OUTPUT_ROOT/mixed_reuse_main" \
REQUEST_LIMIT="${MIXED_REQUEST_LIMIT:-64}" \
MAX_COUNTERFACTUALS_PER_REQUEST="${MIXED_MAX_COUNTERFACTUALS:-5}" \
MAX_MODEL_LEN="${MIXED_MAX_MODEL_LEN:-4096}" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
WARMUP=0 \
REPEAT_COUNT=1 \
ISOLATED=0 \
"$ROOT_DIR/benchmarking/setup/run_empirical_headroom_pilot.sh"

echo "[3/4] Oracle 0 + marginal: real RAG"
SYSTEM="lmcache_cacheblend" \
MODEL="$MODEL_BROAD" \
BUNDLE_ROOT="datasets/processed/empirical_headroom/hotpotqa_rag_main_32req" \
OUTPUT_ROOT="$OUTPUT_ROOT/rag_main" \
REQUEST_LIMIT="${RAG_REQUEST_LIMIT:-32}" \
MAX_COUNTERFACTUALS_PER_REQUEST="${RAG_MAX_COUNTERFACTUALS:-4}" \
MAX_MODEL_LEN="${RAG_MAX_MODEL_LEN:-4096}" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
WARMUP=0 \
REPEAT_COUNT=1 \
ISOLATED=0 \
"$ROOT_DIR/benchmarking/setup/run_empirical_headroom_pilot.sh"

run_concurrency() {
  local name="$1"
  local system="$2"
  local model="$3"
  local bundle="$4"
  local out="$OUTPUT_ROOT/$name"
  echo "[4/4] Concurrency study: $name"
  "$PYTHON_BIN" "$ROOT_DIR/benchmarking/runners/run_concurrency_tension_pilot.py" \
    --system "$system" \
    --model "$model" \
    --bundle-root "$bundle" \
    --output-root "$out" \
    --concurrency-levels ${CONCURRENCY_LEVELS:-1 4 8 16} \
    --target-count-per-level "${TARGET_COUNT_PER_LEVEL:-4}" \
    --max-counterfactuals-per-request "${CONCURRENCY_MAX_COUNTERFACTUALS:-3}" \
    --max-model-len "${CONCURRENCY_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --warmup
  "$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/plot_concurrency_tension_pilot.py" \
    --pilot-root "$out" \
    --output-dir "$out/plots"
}

run_concurrency \
  "concurrency_prefix_control" \
  "vllm_apc" \
  "$MODEL_PREFIX" \
  "datasets/processed/empirical_headroom/shared_prefix_64x16"

run_concurrency \
  "concurrency_mixed_reuse" \
  "lmcache_cacheblend" \
  "$MODEL_BROAD" \
  "datasets/processed/empirical_headroom/mixed_reuse_1024req"

run_concurrency \
  "concurrency_rag" \
  "lmcache_cacheblend" \
  "$MODEL_BROAD" \
  "datasets/processed/empirical_headroom/hotpotqa_rag_main_32req"

echo "Full headroom suite complete at: $OUTPUT_ROOT"
