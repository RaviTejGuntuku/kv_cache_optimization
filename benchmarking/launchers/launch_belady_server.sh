#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${VENV_DIR:-$ROOT_DIR/.venv-sglang}/bin/activate"

: "${MODEL_PATH:?Set MODEL_PATH to a local or Hugging Face model id}"
: "${BELADY_PLAN_PATH:?Set BELADY_PLAN_PATH to a compiled Belady plan JSON}"

TRACE_DIR="${TRACE_DIR:-$ROOT_DIR/artifacts/traces}"
RUN_LABEL="${RUN_LABEL:-belady_replay}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-fcfs}"

mkdir -p "$TRACE_DIR"

export SGLANG_BENCH_TRACE_PATH="$TRACE_DIR/${RUN_LABEL}.jsonl"
export SGLANG_BENCH_TRACE_RUN_LABEL="$RUN_LABEL"
export SGLANG_BELADY_PLAN_PATH="$BELADY_PLAN_PATH"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --page-size "$PAGE_SIZE" \
  --schedule-policy "$SCHEDULE_POLICY" \
  --radix-eviction-policy belady \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  "$@"
