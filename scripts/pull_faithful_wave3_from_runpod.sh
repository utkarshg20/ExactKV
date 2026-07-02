#!/usr/bin/env bash
# Pull wave-3 TurboQuant faithful panel from RunPod.
set -euo pipefail

HOST="${RUNPOD_HOST:-203.57.40.101}"
PORT="${RUNPOD_PORT:-10003}"
USER="${RUNPOD_USER:-root}"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_exactkv}"
REMOTE="${RUNPOD_REMOTE:-/workspace/ExactKV}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$ROOT/reports/external_panels/faithful/wave3"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"

mkdir -p "$LOCAL"

echo "==> Pulling wave-3 $USER@$HOST:$REMOTE/reports/external_panels/faithful/wave3/"
rsync -avz \
  -e "$RSYNC_SSH" \
  "$USER@$HOST:$REMOTE/reports/external_panels/faithful/wave3/" \
  "$LOCAL/"

echo "==> Pulling wave-3 logs"
rsync -avz \
  -e "$RSYNC_SSH" \
  "$USER@$HOST:$REMOTE/reports/faithful_wave3_"*.log \
  "$ROOT/reports/" 2>/dev/null || true

echo "==> Integrating wave-3 summary"
python3 "$ROOT/scripts/integrate_faithful_panel_results.py" --dir "$LOCAL" --write

echo "==> Done."
ls -lah "$LOCAL" | tail -15
