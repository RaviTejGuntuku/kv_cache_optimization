#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-empirical-headroom}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/studies/results/headroom_main}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  exit 1
fi

source "$VENV_DIR/bin/activate"

run_workload() {
  local exp_name="$1"
  local system="$2"
  local bundle_root="$3"
  local request_limit="$4"
  local max_counterfactuals="$5"
  local max_model_len="$6"
  local warmup="$7"

  local exp_root="$OUTPUT_ROOT/$exp_name"
  rm -rf "$exp_root"

  VENV_DIR="$VENV_DIR" \
  SYSTEM="$system" \
  MODEL="$MODEL" \
  BUNDLE_ROOT="$bundle_root" \
  OUTPUT_ROOT="$exp_root" \
  REQUEST_LIMIT="$request_limit" \
  MAX_COUNTERFACTUALS_PER_REQUEST="$max_counterfactuals" \
  PROFILE_ORACLE0_EVERY=0 \
  PROFILE_BASELINE_EVERY=0 \
  PROFILE_COUNTERFACTUAL_EVERY=0 \
  MAX_MODEL_LEN="$max_model_len" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  WARMUP="$warmup" \
  REPEAT_COUNT=1 \
  ISOLATED=0 \
  "$ROOT_DIR/benchmarking/setup/run_empirical_headroom_pilot.sh"

  "$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/plot_empirical_headroom_pilot.py" \
    --pilot-root "$exp_root" \
    --output-dir "$exp_root/plots"

  "$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/summarize_empirical_headroom_pilot.py" \
    --pilot-name "$exp_name" \
    --pilot-root "$exp_root" \
    --output-dir "$exp_root/plots"
}

mkdir -p "$OUTPUT_ROOT"

run_workload \
  "prefix_exact_main" \
  "vllm_apc" \
  "datasets/processed/empirical_headroom/shared_prefix_64x16" \
  96 \
  1 \
  4096 \
  1

run_workload \
  "mixed_reuse_main" \
  "lmcache_cacheblend" \
  "datasets/processed/empirical_headroom/mixed_reuse_1024req" \
  64 \
  5 \
  4096 \
  0

run_workload \
  "rag_main" \
  "lmcache_cacheblend" \
  "datasets/processed/empirical_headroom/hotpotqa_rag_main_32req" \
  32 \
  4 \
  4096 \
  0

echo "Full empirical headroom matrix complete at: $OUTPUT_ROOT"
