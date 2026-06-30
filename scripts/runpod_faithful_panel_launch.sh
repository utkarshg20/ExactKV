#!/usr/bin/env bash
# RunPod launcher: setup env + faithful compressor panel in tmux.
# Usage (from local): ssh runpod-a5000 'bash /workspace/ExactKV/scripts/runpod_faithful_panel_launch.sh'
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
cd "$ROOT"

VENV="${FAITHFUL_VENV:-/workspace/.venv-faithful}"
if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "==> Creating faithful panel venv at $VENV (system-site-packages for CUDA torch)"
  python3.12 -m venv --system-site-packages "$VENV"
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export TMPDIR="${TMPDIR:-/workspace/tmp}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
  "$VENV/bin/pip" install -q kvpress
  "$VENV/bin/pip" install -q -e "$ROOT" --no-deps
fi
PY="$VENV/bin/python3"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export EXACTKV_KIVI_ROOT="${KIVI_DIR:-/tmp/kivi_research}"

echo "==> ExactKV faithful panel launcher $(date -u -Iseconds)"

# KIVI clone (simulate quant only)
if [[ ! -f "${KIVI_DIR:-/tmp/kivi_research}/models/utils_quant.py" ]]; then
  git clone --depth 1 https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
fi
"$PY" -c "from exactkv.compressors.kivi_adapter import _import_kivi_utils; _import_kivi_utils(); print('KIVI OK')"

# kvpress for SnapKV experimental adapter
if ! "$PY" -c "import kvpress" 2>/dev/null; then
  "$PY" -m pip install -q kvpress
fi
"$PY" -c "import kvpress; print('kvpress OK')"

# ExactKV editable install if missing
if ! "$PY" -c "import exactkv" 2>/dev/null; then
  "$PY" -m pip install -q -e ".[dev]"
fi

rm -rf "$ROOT/platform" 2>/dev/null || true

"$PY" -c "
import torch
from exactkv.benchmarks.evidence_plus_panel import resolve_evidence_plus_compressor
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
for c in ['int8','kivi_offline_r32','snapkv_experimental']:
    r = resolve_evidence_plus_compressor(c)
    print(c, r.backend_tier, r.adapter_available)
"

mkdir -p reports/external_panels/faithful reports/external_panels/logs

if tmux has-session -t faithful 2>/dev/null; then
  echo "==> tmux session 'faithful' already running — attach with: tmux attach -t faithful"
  exit 0
fi

tmux new-session -d -s faithful \
  "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' HF_HOME='${HF_HOME:-/workspace/.cache/huggingface}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}' TMPDIR='${TMPDIR:-/workspace/tmp}' && \
   bash '$ROOT/scripts/run_faithful_compressor_panel.sh' 2>&1 | tee '$ROOT/reports/faithful_panel.log'; \
   echo FAITHFUL_PANEL_DONE >> '$ROOT/reports/faithful_panel.log'"

echo "==> Launched tmux session 'faithful'"
echo "    Monitor: tmux attach -t faithful"
echo "    Log:     tail -f $ROOT/reports/faithful_panel.log"
