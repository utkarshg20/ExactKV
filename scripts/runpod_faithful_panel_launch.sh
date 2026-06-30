#!/usr/bin/env bash
# RunPod launcher: setup env + faithful compressor panel in tmux (Mistral then Llama queue).
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
export EXACTKV_RUNDIR="${EXACTKV_RUNDIR:-/tmp/exactkv_panel_run}"

echo "==> ExactKV faithful panel launcher $(date -u -Iseconds)"

if [[ ! -f "${KIVI_DIR:-/tmp/kivi_research}/models/utils_quant.py" ]]; then
  git clone --depth 1 https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
fi
"$PY" -c "from exactkv.compressors.kivi_adapter import _import_kivi_utils; _import_kivi_utils(); print('KIVI OK')"

if ! "$PY" -c "import kvpress" 2>/dev/null; then
  "$PY" -m pip install -q kvpress
fi
"$PY" -c "import kvpress; print('kvpress OK')"

if ! "$PY" -c "import exactkv.benchmarks.external_panel" 2>/dev/null; then
  "$PY" -m pip install -q -e "$ROOT" --no-deps
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

panel_running() {
  pgrep -f "run_faithful_compressor_panel.sh" >/dev/null 2>&1 \
    || pgrep -f "run_external_panel.py.*faithful" >/dev/null 2>&1
}

if panel_running; then
  echo "==> Faithful panel process already running"
  pgrep -af "run_faithful_compressor_panel|run_external_panel.py" || true
  exit 0
fi

# Remove false-positive FAITHFUL_PANEL_DONE from prior failed launches
if [[ -f "$ROOT/reports/faithful_panel.log" ]] && ! pgrep -f "run_external_panel.py.*Mistral-7B" >/dev/null 2>&1; then
  if ! ls "$ROOT/reports/external_panels/faithful/longbench_Mistral_7B_Instruct_v0_3_raw.json" >/dev/null 2>&1; then
    sed -i '/^FAITHFUL_PANEL_DONE/d' "$ROOT/reports/faithful_panel.log" 2>/dev/null || true
  fi
fi

# Kill stale tmux sessions without live panel processes
for sess in faithful faithful_llama; do
  if tmux has-session -t "$sess" 2>/dev/null && ! panel_running; then
    tmux kill-session -t "$sess" 2>/dev/null || true
  fi
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

tmux new-session -d -s faithful \
  "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_RUNDIR='$EXACTKV_RUNDIR' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' TMPDIR='${TMPDIR}' FAITHFUL_MODELS='$MISTRAL' FAITHFUL_PANEL_LOG='$ROOT/reports/faithful_panel.log' && \
   echo '=== faithful tmux start $STAMP ===' >> '$ROOT/reports/faithful_panel.log' && \
   bash '$ROOT/scripts/run_faithful_compressor_panel.sh'"

tmux new-session -d -s faithful_llama \
  "export PYTHON='$PY' EXACTKV_ROOT='$ROOT' EXACTKV_RUNDIR='/tmp/exactkv_panel_run_llama' EXACTKV_KIVI_ROOT='${KIVI_DIR:-/tmp/kivi_research}' HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' TMPDIR='${TMPDIR}' && \
   echo '=== faithful_llama tmux start $STAMP ===' >> '$ROOT/reports/faithful_llama_panel.log' && \
   bash '$ROOT/scripts/run_faithful_llama_after_mistral.sh' 2>&1 | tee -a '$ROOT/reports/faithful_llama_panel.log'"

echo "==> Launched tmux sessions: faithful (Mistral), faithful_llama (queued)"
echo "    Monitor:  tmux attach -t faithful"
echo "    Mistral:  tail -f $ROOT/reports/faithful_panel.log"
echo "    Llama:    tail -f $ROOT/reports/faithful_llama_panel.log"
