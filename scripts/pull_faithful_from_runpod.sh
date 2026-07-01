#!/usr/bin/env bash
# Pull faithful panel artifacts from RunPod to local reports/.
set -euo pipefail

HOST="${RUNPOD_HOST:-203.57.40.101}"
PORT="${RUNPOD_PORT:-10003}"
USER="${RUNPOD_USER:-root}"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_exactkv}"
REMOTE="${RUNPOD_REMOTE:-/workspace/ExactKV}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$ROOT/reports/external_panels/faithful"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"

mkdir -p "$LOCAL"

echo "==> Pulling $USER@$HOST:$REMOTE/reports/external_panels/faithful/ -> $LOCAL"
rsync -avz \
  -e "$RSYNC_SSH" \
  "$USER@$HOST:$REMOTE/reports/external_panels/faithful/" \
  "$LOCAL/"

echo "==> Integrating faithful summary"
python3 "$ROOT/scripts/integrate_faithful_panel_results.py" --dir "$LOCAL" --write

echo "==> Done. Files:"
ls -lah "$LOCAL" | tail -20
