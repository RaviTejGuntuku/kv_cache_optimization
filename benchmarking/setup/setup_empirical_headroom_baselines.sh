#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${EXTERNAL_DIR:-$ROOT_DIR/external}"

mkdir -p "$EXTERNAL_DIR"

clone_or_update() {
  local repo_url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" fetch --depth 1 origin
    git -C "$dest" reset --hard origin/HEAD
  else
    git clone --depth 1 "$repo_url" "$dest"
  fi
}

clone_or_update "https://github.com/vllm-project/vllm" "$EXTERNAL_DIR/vllm"
clone_or_update "https://github.com/LMCache/LMCache" "$EXTERNAL_DIR/lmcache"

echo "Baseline repos ready under $EXTERNAL_DIR"
echo "  vLLM commit:   $(git -C "$EXTERNAL_DIR/vllm" rev-parse --short HEAD)"
echo "  LMCache commit: $(git -C "$EXTERNAL_DIR/lmcache" rev-parse --short HEAD)"
