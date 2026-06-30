#!/usr/bin/env bash
# Sync ExactKV to RunPod volume over direct TCP SSH (SCP/rsync).
# Usage:
#   RUNPOD_HOST=203.57.40.169 RUNPOD_PORT=10113 bash scripts/sync_to_runpod.sh
set -euo pipefail

HOST="${RUNPOD_HOST:-203.57.40.169}"
PORT="${RUNPOD_PORT:-10113}"
USER="${RUNPOD_USER:-root}"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_exactkv}"
REMOTE="${RUNPOD_REMOTE:-/workspace/ExactKV}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"

echo "==> Syncing $ROOT -> $USER@$HOST:$REMOTE"
rsync -avz --delete \
  -e "$RSYNC_SSH" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'reports/scale_7b' \
  "$ROOT/" "$USER@$HOST:$REMOTE/"

echo "==> Remote bootstrap"
$SSH "$USER@$HOST" "cd $REMOTE && chmod +x scripts/setup_runpod_evidence_plus.sh && bash scripts/setup_runpod_evidence_plus.sh"
