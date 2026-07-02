#!/usr/bin/env bash
# Run inside RunPod web terminal (or SSH once connected).
# Pulls latest main from GitHub and launches wave-3 TurboQuant panel in tmux.
#
# Paste into RunPod web terminal:
#   curl -fsSL https://raw.githubusercontent.com/utkarshg20/ExactKV/main/scripts/runpod_wave3_web_bootstrap.sh | bash
#
# Or if /workspace/ExactKV already exists:
#   cd /workspace/ExactKV && bash scripts/runpod_wave3_web_bootstrap.sh
set -euo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
REPO_URL="${EXACTKV_REPO_URL:-https://github.com/utkarshg20/ExactKV.git}"

echo "==> ExactKV wave-3 web bootstrap $(date -u -Iseconds)"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "==> Cloning repo to $ROOT"
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$REPO_URL" "$ROOT"
fi

cd "$ROOT"
git fetch origin main 2>/dev/null || true
git checkout main 2>/dev/null || true
git pull origin main 2>/dev/null || echo "WARN: git pull failed; using on-disk tree"

chmod +x scripts/run_faithful_turboquant_wave3_panel.sh \
         scripts/runpod_faithful_wave3_launch.sh \
         scripts/setup_faithful_compressor_env.sh 2>/dev/null || true

export INSTALL_TURBOQUANT=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

# Create faithful venv if missing
if [[ ! -x /workspace/.venv-faithful/bin/python3 ]]; then
  echo "==> Creating /workspace/.venv-faithful"
  python3 -m venv --system-site-packages /workspace/.venv-faithful
  /workspace/.venv-faithful/bin/pip install -q -e "$ROOT" scipy 2>/dev/null || \
    /workspace/.venv-faithful/bin/pip install -q scipy
fi

bash "$ROOT/scripts/runpod_faithful_wave3_launch.sh"

echo ""
echo "==> Wave-3 queued. Monitor: tmux attach -t faithful_wave3"
echo "==> Log: tail -f $ROOT/reports/faithful_wave3_*.log"
