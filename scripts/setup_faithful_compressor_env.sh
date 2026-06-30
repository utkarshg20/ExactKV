#!/usr/bin/env bash
# One-time setup for faithful external compressor panels (RunPod or local GPU).
#
# Installs:
#   - jy-yuan/KIVI clone (PYTHONPATH, simulate quant only — no CUDA build)
#   - kvpress (SnapKV experimental adapter)
#
# Does NOT install KVQuant (requires isolated transformers~=4.44 venv).
# See docs/FAITHFUL_COMPRESSOR_INTEGRATION.md for KVQuant Llama calibration path.
set -euo pipefail

ROOT="${EXACTKV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
KIVI_DIR="${KIVI_DIR:-/tmp/kivi_research}"
PY="${PYTHON:-python3}"

echo "==> ExactKV root: $ROOT"
echo "==> KIVI clone target: $KIVI_DIR"

if [[ ! -d "$KIVI_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/jy-yuan/KIVI.git "$KIVI_DIR"
else
  echo "==> KIVI repo already present at $KIVI_DIR"
fi

export PYTHONPATH="$KIVI_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import models.utils_quant; print('KIVI utils_quant OK:', models.utils_quant.__file__)"

echo "==> Installing kvpress (SnapKV experimental)"
"$PY" -m pip install -q kvpress
"$PY" -c "import kvpress; print('kvpress OK')"

echo ""
echo "==> Setup complete. Add to your shell before running panels:"
echo "    export PYTHONPATH=$KIVI_DIR"
echo "    bash scripts/run_faithful_compressor_panel.sh"
