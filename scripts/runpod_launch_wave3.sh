#!/usr/bin/env bash
# Discover RunPod SSH endpoint and launch wave-3.
# Usage:
#   RUNPOD_HOST=203.57.40.101 RUNPOD_PORT=10003 bash scripts/runpod_launch_wave3.sh
#   bash scripts/runpod_launch_wave3.sh   # tries defaults + ssh config alias
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_exactkv}"
USER="${RUNPOD_USER:-root}"
REMOTE="${RUNPOD_REMOTE:-/workspace/ExactKV}"

try_ssh() {
  local host="$1" port="$2"
  ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -p "$port" -i "$KEY" "$USER@$host" 'echo OK' 2>/dev/null
}

HOST="${RUNPOD_HOST:-}"
PORT="${RUNPOD_PORT:-}"

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "==> Probing known RunPod endpoints..."
  for spec in \
    "203.57.40.101:10003" \
    "194.68.245.16:22108" \
    "203.57.40.169:10113"; do
    h="${spec%:*}"; p="${spec#*:}"
    if [[ "$(try_ssh "$h" "$p")" == "OK" ]]; then
      HOST="$h"; PORT="$p"; break
    fi
  done
fi

if [[ -z "$HOST" || -z "$PORT" ]]; then
  if ssh -o ConnectTimeout=8 -o BatchMode=yes runpod-a5000 'echo OK' 2>/dev/null | grep -q OK; then
    echo "==> Using ssh config alias runpod-a5000"
    RUNPOD_HOST= RUNPOD_PORT= bash "$ROOT/scripts/sync_to_runpod.sh"
    ssh runpod-a5000 "cd $REMOTE && bash scripts/runpod_faithful_wave3_launch.sh"
    exit 0
  fi
  echo "ERROR: Cannot reach RunPod. Set RUNPOD_HOST and RUNPOD_PORT from dashboard." >&2
  echo "  Example: RUNPOD_HOST=x.x.x.x RUNPOD_PORT=10xxx bash $0" >&2
  echo "  Or paste in web terminal:" >&2
  echo "    curl -fsSL https://raw.githubusercontent.com/utkarshg20/ExactKV/main/scripts/runpod_wave3_web_bootstrap.sh | bash" >&2
  exit 1
fi

export RUNPOD_HOST="$HOST" RUNPOD_PORT="$PORT" RUNPOD_SSH_KEY="$KEY"
echo "==> Connected via $USER@$HOST:$PORT"

# Update local ssh config alias for next time
if grep -q "Host runpod-a5000" "$HOME/.ssh/config" 2>/dev/null; then
  perl -i -pe "s/^    HostName .*/    HostName $HOST/; s/^    Port .*/    Port $PORT/" "$HOME/.ssh/config" 2>/dev/null || true
fi

bash "$ROOT/scripts/sync_to_runpod.sh"
ssh -p "$PORT" -i "$KEY" "$USER@$HOST" "cd $REMOTE && bash scripts/runpod_faithful_wave3_launch.sh"

echo "==> Wave-3 launched. Monitor:"
echo "    ssh -p $PORT -i $KEY $USER@$HOST -t 'tmux attach -t faithful_wave3'"
