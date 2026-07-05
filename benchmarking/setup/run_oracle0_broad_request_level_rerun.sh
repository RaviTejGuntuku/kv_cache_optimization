#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
RESULT_ROOT="${RESULT_ROOT:-studies/results/oracle0_broad_request_level_$(date -u +%Y%m%dT%H%M%SZ)}"

run_case() {
  local case_name="$1"
  local bundle_root="$2"
  local request_limit="$3"
  local case_root="$RESULT_ROOT/$case_name"

  mkdir -p "$case_root/raw/baseline_replay_fcfs" "$case_root/raw/oracle0_fcfs"

  python benchmarking/runners/run_baseline_replay_empirical.py \
    --system lmcache_cacheblend \
    --model "$MODEL" \
    --bundle-root "$bundle_root" \
    --output-root "$case_root/raw/baseline_replay_fcfs" \
    --request-limit "$request_limit" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --repeat-count "$REPEAT_COUNT" \
    --isolated \
    --no-timestamp

  python benchmarking/runners/run_oracle0_empirical.py \
    --system lmcache_cacheblend \
    --model "$MODEL" \
    --bundle-root "$bundle_root" \
    --output-root "$case_root/raw/oracle0_fcfs" \
    --request-limit "$request_limit" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --repeat-count "$REPEAT_COUNT" \
    --isolated \
    --no-timestamp

  python benchmarking/analysis_scripts/plot_oracle0_headroom.py \
    --baseline-root "$case_root/raw/baseline_replay_fcfs" \
    --oracle-root "$case_root/raw/oracle0_fcfs" \
    --output-dir "$case_root/analysis/plots"

  python benchmarking/analysis_scripts/audit_oracle0_measurements.py \
    --case-root "$case_root" \
    --output-dir "$case_root/analysis/audit"
}

run_case \
  "oracle0_fcfs_broad_headroom__mixed_reuse__lmcache_cacheblend__request_level" \
  "datasets/processed/empirical_headroom/mixed_reuse_aligned_pilot_24req" \
  "${MIXED_REQUEST_LIMIT:-24}"

run_case \
  "oracle0_fcfs_broad_headroom__real_rag__lmcache_cacheblend__request_level" \
  "datasets/processed/empirical_headroom/hotpotqa_rag_aligned_pilot_16req" \
  "${RAG_REQUEST_LIMIT:-16}"

echo "Broad LMCache request-level rerun complete: $RESULT_ROOT"
