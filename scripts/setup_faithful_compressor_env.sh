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
if [[ "$PY" == "python3" && -x /workspace/.venv-faithful/bin/python3 ]]; then
  PY=/workspace/.venv-faithful/bin/python3
fi

echo "==> ExactKV root: $ROOT"
echo "==> KIVI clone target: $KIVI_DIR"

if [[ ! -d "$KIVI_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/jy-yuan/KIVI.git "$KIVI_DIR"
else
  echo "==> KIVI repo already present at $KIVI_DIR"
fi

export PYTHONPATH="$KIVI_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import models.utils_quant; print('KIVI utils_quant OK:', models.utils_quant.__file__)"

echo "==> Installing kvpress (SnapKV + KnormPress experimental)"
"$PY" -m pip install -q kvpress
"$PY" -c "import kvpress; print('kvpress OK')"

TURBO_DIR="${EXACTKV_TURBOQUANT_ROOT:-/tmp/turboquant_plus}"
if [[ "${INSTALL_TURBOQUANT:-0}" == "1" ]]; then
  if [[ ! -d "$TURBO_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/TheTom/turboquant_plus "$TURBO_DIR"
  fi
  export PYTHONPATH="$TURBO_DIR${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -m pip install -q scipy 2>/dev/null || true
  if ! "$PY" -c "import turboquant; print('turboquant OK:', turboquant.__file__)" 2>/dev/null; then
    echo "WARN: TurboQuant import failed (wave-2 may skip turboquant_experimental cells)"
  fi
else
  echo "==> TurboQuant: skip (set INSTALL_TURBOQUANT=1 to clone TheTom/turboquant_plus)"
fi

echo ""
echo "==> Setup complete. Add to your shell before running panels:"
echo "    export EXACTKV_KIVI_ROOT=$KIVI_DIR"
echo "    export EXACTKV_TURBOQUANT_ROOT=$TURBO_DIR   # if TurboQuant installed"
echo "    bash scripts/run_faithful_compressor_panel.sh"
echo "    bash scripts/run_faithful_external_wave2_smoke.sh  # KnormPress + TurboQuant compare"
