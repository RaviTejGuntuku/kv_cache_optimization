#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNPOD_HOST="${RUNPOD_HOST:-64.247.201.31}"
RUNPOD_PORT="${RUNPOD_PORT:-18800}"
RUNPOD_USER="${RUNPOD_USER:-root}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/kv_cache_research}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-$REMOTE_ROOT/studies/results}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-$ROOT_DIR/studies/results/runpod_pulled}"
RUN_NAME="${1:-}"

if [[ -z "$RUN_NAME" ]]; then
  cat <<EOF
Usage:
  benchmarking/setup/pull_runpod_results.sh <remote-results-subdir>

Example:
  benchmarking/setup/pull_runpod_results.sh runpod_small_pilot
  benchmarking/setup/pull_runpod_results.sh runpod_warmup_prefix_pilot
EOF
  exit 1
fi

mkdir -p "$LOCAL_RESULTS_ROOT"

rsync -avh --progress \
  -e "ssh -i $RUNPOD_KEY -p $RUNPOD_PORT" \
  "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_RESULTS_ROOT/$RUN_NAME/" \
  "$LOCAL_RESULTS_ROOT/$RUN_NAME/"

cat <<EOF
Pulled RunPod results:
  remote: $REMOTE_RESULTS_ROOT/$RUN_NAME/
  local:  $LOCAL_RESULTS_ROOT/$RUN_NAME/
EOF
