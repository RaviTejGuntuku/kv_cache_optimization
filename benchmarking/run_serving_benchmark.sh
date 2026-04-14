#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VENV_DIR:-$ROOT_DIR/.venv-sglang}/bin/activate"

: "${MODEL_PATH:?Set MODEL_PATH to the served model id/path}"
: "${DATASET_PATH:?Set DATASET_PATH to a custom JSONL dataset}"

PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
OUTPUT_FILE="${OUTPUT_FILE:-$ROOT_DIR/artifacts/benchmarks/$(date +%Y%m%d_%H%M%S)_serving.jsonl}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

CMD=(
  python -m sglang.bench_serving
  --backend sglang
  --base-url "$BASE_URL"
  --dataset-name custom
  --dataset-path "$DATASET_PATH"
  --model "$MODEL_PATH"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --output-file "$OUTPUT_FILE"
  --output-details
)

if [[ -n "$MAX_CONCURRENCY" ]]; then
  CMD+=(--max-concurrency "$MAX_CONCURRENCY")
fi

"${CMD[@]}" "$@"
