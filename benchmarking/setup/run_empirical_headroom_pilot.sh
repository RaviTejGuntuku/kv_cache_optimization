#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-empirical-headroom}"

SYSTEM="${SYSTEM:-}"
MODEL="${MODEL:-}"
BUNDLE_ROOT="${BUNDLE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/studies/results/empirical_headroom_pilot}"
REQUEST_LIMIT="${REQUEST_LIMIT:-32}"
MAX_COUNTERFACTUALS_PER_REQUEST="${MAX_COUNTERFACTUALS_PER_REQUEST:-4}"
PROFILE_ORACLE0_EVERY="${PROFILE_ORACLE0_EVERY:-0}"
PROFILE_BASELINE_EVERY="${PROFILE_BASELINE_EVERY:-0}"
PROFILE_COUNTERFACTUAL_EVERY="${PROFILE_COUNTERFACTUAL_EVERY:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
WARMUP="${WARMUP:-1}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
ISOLATED="${ISOLATED:-0}"

if [[ -z "$SYSTEM" || -z "$MODEL" || -z "$BUNDLE_ROOT" ]]; then
  cat <<EOF
Usage:
  SYSTEM=<vllm_apc|lmcache_exact|lmcache_cacheblend> \\
  MODEL=<model-name-or-path> \\
  BUNDLE_ROOT=<bundle-dir> \\
  benchmarking/setup/run_empirical_headroom_pilot.sh

Optional env vars:
  OUTPUT_ROOT=$OUTPUT_ROOT
  REQUEST_LIMIT=$REQUEST_LIMIT
  MAX_COUNTERFACTUALS_PER_REQUEST=$MAX_COUNTERFACTUALS_PER_REQUEST
  PROFILE_ORACLE0_EVERY=$PROFILE_ORACLE0_EVERY
  PROFILE_BASELINE_EVERY=$PROFILE_BASELINE_EVERY
  PROFILE_COUNTERFACTUAL_EVERY=$PROFILE_COUNTERFACTUAL_EVERY
  MAX_MODEL_LEN=$MAX_MODEL_LEN
  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
  WARMUP=$WARMUP
  REPEAT_COUNT=$REPEAT_COUNT
  ISOLATED=$ISOLATED
EOF
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  echo "Run benchmarking/setup/setup_empirical_headroom_env.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

RAW_ROOT="$OUTPUT_ROOT/raw"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"
ORACLE0_ROOT="$RAW_ROOT/oracle0_fcfs"
COUNTERFACTUAL_ROOT="$RAW_ROOT/marginal_counterfactuals_fcfs"
ACCOUNTING_ROOT="$ANALYSIS_ROOT/missed_opportunity_accounting"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/benchmarking/runners/run_oracle0_empirical.py" \
  --system "$SYSTEM" \
  --model "$MODEL" \
  --bundle-root "$BUNDLE_ROOT" \
  --output-root "$ORACLE0_ROOT" \
  --request-limit "$REQUEST_LIMIT" \
  --profile-every "$PROFILE_ORACLE0_EVERY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --repeat-count "$REPEAT_COUNT" \
  $([[ "$ISOLATED" == "1" ]] && printf '%s' "--isolated") \
  $([[ "$WARMUP" == "1" ]] && printf '%s' "--warmup")

"$PYTHON_BIN" "$ROOT_DIR/benchmarking/runners/run_marginal_counterfactuals_empirical.py" \
  --system "$SYSTEM" \
  --model "$MODEL" \
  --bundle-root "$BUNDLE_ROOT" \
  --output-root "$COUNTERFACTUAL_ROOT" \
  --request-limit "$REQUEST_LIMIT" \
  --max-counterfactuals-per-request "$MAX_COUNTERFACTUALS_PER_REQUEST" \
  --profile-baseline-every "$PROFILE_BASELINE_EVERY" \
  --profile-counterfactual-every "$PROFILE_COUNTERFACTUAL_EVERY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --repeat-count "$REPEAT_COUNT" \
  $([[ "$ISOLATED" == "1" ]] && printf '%s' "--isolated") \
  $([[ "$WARMUP" == "1" ]] && printf '%s' "--warmup")

LATEST_ORACLE0_DIR="$(find "$RAW_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'oracle0_fcfs__*' | sort | tail -n 1)"
LATEST_COUNTERFACTUAL_DIR="$(find "$RAW_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'marginal_counterfactuals_fcfs__*' | sort | tail -n 1)"

if [[ -z "$LATEST_ORACLE0_DIR" || -z "$LATEST_COUNTERFACTUAL_DIR" ]]; then
  LATEST_ORACLE0_DIR="$(find "$OUTPUT_ROOT" -type d \\( -name 'oracle0_fcfs__*' -o -name 'oracle0__*' \\) | sort | tail -n 1)"
  LATEST_COUNTERFACTUAL_DIR="$(find "$OUTPUT_ROOT" -type d \\( -name 'marginal_counterfactuals_fcfs__*' -o -name 'marginal_counterfactuals__*' \\) | sort | tail -n 1)"
fi

if [[ -z "$LATEST_ORACLE0_DIR" || -z "$LATEST_COUNTERFACTUAL_DIR" ]]; then
  echo "Failed to resolve latest Oracle 0 or counterfactual output directories."
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/analyze_empirical_missed_opportunities.py" \
  --bundle-root "$BUNDLE_ROOT" \
  --oracle0-root "$LATEST_ORACLE0_DIR" \
  --counterfactual-root "$LATEST_COUNTERFACTUAL_DIR" \
  --output-root "$ACCOUNTING_ROOT"

PLOT_ROOT="$ANALYSIS_ROOT/plots"
SUMMARY_ROOT="$ANALYSIS_ROOT/summary"

"$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/plot_empirical_headroom_pilot.py" \
  --pilot-root "$OUTPUT_ROOT" \
  --output-dir "$PLOT_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/benchmarking/analysis_scripts/summarize_empirical_headroom_pilot.py" \
  --pilot-name "$(basename "$OUTPUT_ROOT")" \
  --pilot-root "$OUTPUT_ROOT" \
  --output-dir "$SUMMARY_ROOT"

cat <<EOF
Empirical headroom pilot complete.

Oracle 0 outputs:
  $LATEST_ORACLE0_DIR

Marginal counterfactual outputs:
  $LATEST_COUNTERFACTUAL_DIR

Missed-opportunity accounting outputs:
  $ACCOUNTING_ROOT

Plots:
  $PLOT_ROOT

Summary:
  $SUMMARY_ROOT
EOF
