#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found on this machine." >&2
  exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<EOF
Usage:
  benchmarking/setup/sync_to_vista.sh <tacc_username> [remote_dir]

Examples:
  benchmarking/setup/sync_to_vista.sh tejg
  benchmarking/setup/sync_to_vista.sh tejg /scratch/12345/tejg/kv_cache_research

Default remote_dir:
  \$SCRATCH/kv_cache_research

Notes:
  - This script runs locally on your Mac, not on Vista.
  - It syncs only the files needed for the empirical headroom pilot.
  - It intentionally excludes local virtualenvs, .git history, local result dumps,
    processed datasets, external clones, and the large sglang tree.
EOF
  exit 1
fi

TACC_USER="$1"
REMOTE_DIR_INPUT="${2:-__USE_REMOTE_SCRATCH__}"
REMOTE_HOST="${REMOTE_HOST:-vista.tacc.utexas.edu}"

if [[ "$REMOTE_DIR_INPUT" == "__USE_REMOTE_SCRATCH__" ]]; then
  REMOTE_SCRATCH="$(ssh "${TACC_USER}@${REMOTE_HOST}" 'printf %s "$SCRATCH"' 2>/dev/null || true)"
  if [[ -z "$REMOTE_SCRATCH" ]]; then
    echo "Failed to resolve \$SCRATCH on ${REMOTE_HOST}. Try passing an explicit remote directory." >&2
    exit 1
  fi
  REMOTE_DIR="${REMOTE_SCRATCH}/kv_cache_research"
else
  REMOTE_DIR="$REMOTE_DIR_INPUT"
fi

echo "Syncing empirical headroom pilot files to ${TACC_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avh --progress \
  --prune-empty-dirs \
  --filter='+ /.gitignore' \
  --filter='+ /README.md' \
  --filter='+ /EXPERIMENT.md' \
  --filter='+ /benchmarking/' \
  --filter='+ /benchmarking/***' \
  --filter='+ /docs/' \
  --filter='+ /docs/empirical_headroom_setup.md' \
  --filter='+ /studies/' \
  --filter='+ /studies/specs/' \
  --filter='+ /studies/specs/prefix_perfect_prefetch_oracle/' \
  --filter='+ /studies/specs/prefix_perfect_prefetch_oracle/EXPERIMENT.md' \
  --filter='+ /studies/specs/marginal_counterfactuals/' \
  --filter='+ /studies/specs/marginal_counterfactuals/EXPERIMENT.md' \
  --filter='+ /studies/specs/missed_opportunity_accounting/' \
  --filter='+ /studies/specs/missed_opportunity_accounting/EXPERIMENT.md' \
  --filter='+ /data/' \
  --filter='+ /data/README.md' \
  --filter='+ /datasets/' \
  --filter='+ /datasets/README.md' \
  --filter='+ /external/' \
  --filter='+ /external/README.md' \
  --filter='- /.git/' \
  --filter='- /.claude/' \
  --filter='- /.codex_state/' \
  --filter='- /.venv*/' \
  --filter='- /external/vllm/' \
  --filter='- /external/lmcache/' \
  --filter='- /sglang/' \
  --filter='- /studies/results/' \
  --filter='- /studies/runs/' \
  --filter='- /results/' \
  --filter='- /runs/' \
  --filter='- /datasets/raw/' \
  --filter='- /datasets/processed/' \
  --filter='- /datasets/synthetic/' \
  --filter='- /tacc_kv_cache_headroom_abstract.pdf' \
  --filter='- /.DS_Store' \
  --filter='- **/__pycache__/' \
  --filter='- **/*.pyc' \
  --filter='- *' \
  "$ROOT_DIR/" "${TACC_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

cat <<EOF

Sync complete.

Next on Vista:
  ssh ${TACC_USER}@${REMOTE_HOST}
  cd ${REMOTE_DIR}
  benchmarking/setup/setup_empirical_headroom_baselines.sh
  benchmarking/setup/setup_empirical_headroom_env.sh
  source .venv-empirical-headroom/bin/activate
  python benchmarking/workload_generators/generate_empirical_headroom_workloads.py

Then get a GPU node with idev and run a pilot.
EOF
