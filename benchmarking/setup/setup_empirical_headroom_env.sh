#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-empirical-headroom}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SOURCE_PACKAGES="${INSTALL_SOURCE_PACKAGES:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "setuptools>=77.0.3,<81.0.0" "packaging>=24.2" ninja

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "Python 3.10+ is required for the empirical headroom environment. "
        "Please load a newer Python module on Vista before rerunning this script."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ "$INSTALL_SOURCE_PACKAGES" == "1" ]]; then
  if [[ -d "$ROOT_DIR/external/vllm" ]]; then
    python -m pip install --no-build-isolation -e "$ROOT_DIR/external/vllm"
  fi

  if [[ -d "$ROOT_DIR/external/lmcache" ]]; then
    python -m pip install --no-build-isolation -e "$ROOT_DIR/external/lmcache"
  fi
else
  python -m pip install --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" "torchvision" "torchaudio"
  python -m pip install vllm openai transformers requests
  python -m pip install --no-build-isolation lmcache
fi

cat <<EOF
Empirical headroom environment ready.

Activate with:
  source "$VENV_DIR/bin/activate"

Notes:
  - Default mode installs PyTorch from:
      $TORCH_INDEX_URL
    using torch==$TORCH_VERSION
  - Default mode then installs stable \`vllm\` from PyPI and \`lmcache\` with
    \`--no-build-isolation\` so the LMCache extension build reuses that same Torch/CUDA stack.
  - To force editable source installs from external clones, rerun with:
      INSTALL_SOURCE_PACKAGES=1 benchmarking/setup/setup_empirical_headroom_env.sh
  - LMCache + CacheBlend may still require the documented vLLM patch from:
      $ROOT_DIR/external/lmcache/examples/blend_kv_v1/README.md
  - Nsight Systems profiling requires \`nsys\` to be installed separately.
EOF
