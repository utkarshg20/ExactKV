#!/usr/bin/env bash
# Local watcher: waits up to 6h for NEW pod SSH, then launches GPU marathon.
#
# Credentials ONLY from:
#   1) env RUNPOD_HOST RUNPOD_PORT
#   2) file launch/pod_ssh.env  (RUNPOD_HOST=... / RUNPOD_PORT=... or HOST=/PORT=)
#   3) a NEW user Connect paste after watch start (SSH over exposed TCP line)
set -uo pipefail
cd /Users/utkarshgupta/Documents/ExactKV
KEY="${RUNPOD_KEY:-$HOME/.ssh/runpod_exactkv}"
TOKEN_FILE="$HOME/.cache/huggingface/token"
START=$(date +%s)
export WATCH_START_EPOCH=$START
DEADLINE=$((START + 6 * 3600))
WATCHLOG=/tmp/exactkv_pod_watch_6h.log
: >"$WATCHLOG"
echo "watch_start $(date -u) epoch=$START deadline_epoch=$DEADLINE" | tee -a "$WATCHLOG"

extract_fresh_from_transcript() {
  python3 - <<'PY'
import json, re, os
from pathlib import Path
start = float(os.environ.get("WATCH_START_EPOCH", "0"))
cutoff = start - 600  # 10 min before watch
root = Path.home() / ".cursor/projects/Users-utkarshgupta-Documents-ExactKV/agent-transcripts"
pat = re.compile(r"ssh root@([0-9.]+)\s+-p\s+(\d+)")
best = None
for f in root.rglob("*.jsonl"):
    if f.stat().st_mtime < cutoff:
        continue
    for line in f.open(errors="ignore"):
        if '"role":"user"' not in line or "ssh root@" not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        text = ""
        c = o.get("message", {}).get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    text += part.get("text", "")
        if "SSH over exposed TCP" not in text and "Direct TCP" not in text:
            continue
        # Require this paste to be "new" relative to watch: must mention being gone /
        # marathon intent OR file mtime after watch start.
        fresh = f.stat().st_mtime >= start - 60
        intent = ("6 hours" in text.lower()) or ("pod is up" in text.lower())
        if not (fresh or intent):
            continue
        for m in pat.finditer(text):
            score = f.stat().st_mtime
            if intent:
                score += 1e6
            cand = (score, m.group(1), m.group(2))
            if best is None or cand > best:
                best = cand
if best:
    print(f"{best[1]} {best[2]}")
PY
}

resolve_creds() {
  if [[ -n "${RUNPOD_HOST:-}" && -n "${RUNPOD_PORT:-}" ]]; then
    echo "$RUNPOD_HOST $RUNPOD_PORT"
    return 0
  fi
  if [[ -f launch/pod_ssh.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source launch/pod_ssh.env
    set +a
    if [[ -n "${RUNPOD_HOST:-}${HOST:-}" && -n "${RUNPOD_PORT:-}${PORT:-}" ]]; then
      echo "${RUNPOD_HOST:-$HOST} ${RUNPOD_PORT:-$PORT}"
      return 0
    fi
  fi
  extract_fresh_from_transcript
}

try_ssh() {
  local host=$1 port=$2
  ssh -p "$port" -i "$KEY" -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=accept-new \
    root@"$host" 'echo POD_OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' 2>/dev/null
}

launch_marathon() {
  local host=$1 port=$2
  local RSYNC_SSH="ssh -p $port -i $KEY -o StrictHostKeyChecking=accept-new"
  echo "CONNECTED $host:$port — syncing + launching" | tee -a "$WATCHLOG"

  python3 - <<PY
from pathlib import Path
import re
p = Path.home() / ".ssh" / "config"
block = """Host runpod-marathon
    HostName $host
    Port $port
    User root
    IdentityFile ~/.ssh/runpod_exactkv
    StrictHostKeyChecking accept-new
"""
text = p.read_text() if p.exists() else ""
text = re.sub(r"Host runpod-marathon\n(?: .*\n)*", "", text)
p.write_text(text.rstrip() + "\n\n" + block)
PY

  rsync -az --no-owner --no-group --delete \
    --exclude '.venv/' --exclude '.venv-*/' --exclude '__pycache__/' --exclude '.git/' \
    --exclude 'reports/exp*/' --exclude 'reports/phase*/' --exclude 'reports/scale_7b/' \
    --exclude 'paper/*.pdf' --exclude 'dist/' --exclude '*.egg-info/' \
    /Users/utkarshgupta/Documents/ExactKV/ \
    root@"$host":/workspace/ExactKV/ \
    -e "$RSYNC_SSH"

  if [[ -f "$TOKEN_FILE" ]]; then
    ssh -p "$port" -i "$KEY" root@"$host" \
      "mkdir -p /workspace/hf; cat > /workspace/hf/token; chmod 600 /workspace/hf/token" \
      <"$TOKEN_FILE"
  fi

  ssh -p "$port" -i "$KEY" root@"$host" bash -s <<REMOTE
set -euo pipefail
export HF_HOME=/workspace/hf
export HF_TOKEN="\$(tr -d '[:space:]' </workspace/hf/token)"
export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
if [[ ! -x /workspace/.venv-runpod/bin/python3 ]]; then
  python3 -m venv --system-site-packages /workspace/.venv-runpod
fi
source /workspace/.venv-runpod/bin/activate
cd /workspace/ExactKV
pip install -q --no-deps -e . || true
pip install -q 'transformers>=4.40' 'accelerate>=0.29' 'safetensors>=0.4' 'numpy>=1.24' 'tqdm>=4.66' 'huggingface_hub' || true
chmod +x scripts/run_six_hour_gpu_marathon.sh
export DEADLINE_EPOCH=$DEADLINE
tmux has-session -t marathon 2>/dev/null && tmux kill-session -t marathon || true
tmux new-session -d -s marathon "export DEADLINE_EPOCH=$DEADLINE; export HF_HOME=/workspace/hf; export HF_TOKEN=\$(cat /workspace/hf/token); export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN; bash /workspace/ExactKV/scripts/run_six_hour_gpu_marathon.sh"
tmux ls
echo LAUNCHED_MARATHON
REMOTE

  # Separate aggressive puller in tmux on local side via frequent loop
  while (( $(date +%s) < DEADLINE )); do
    out=$(ssh -p "$port" -i "$KEY" root@"$host" \
      'tail -n 8 /workspace/marathon_6h.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; test -f /workspace/MARATHON_6H_DONE && echo ALL_DONE; ls reports/external_panels/serving_microbench/*_raw.json 2>/dev/null | wc -l; ls reports/external_panels/serving_microbench/*.bak.*.json 2>/dev/null | wc -l' 2>&1) || out="SSH_FAIL"
    echo "==== $(date -u +%H:%M:%S) ====" | tee -a "$WATCHLOG"
    echo "$out" | tee -a "$WATCHLOG"
    # Pull every cycle (~60s) — live copy + timestamped backup under reports/remote_backups/
    RUNPOD_HOST="$host" RUNPOD_PORT="$port" bash scripts/pull_runpod_artifacts.sh \
      2>&1 | tee -a "$WATCHLOG" || true
    if echo "$out" | grep -q ALL_DONE; then
      break
    fi
    if echo "$out" | grep -q SSH_FAIL; then
      echo "SSH_FAIL — will retry pull next cycle" | tee -a "$WATCHLOG"
    fi
    sleep 60
  done

  # Final catch-all pull
  RUNPOD_HOST="$host" RUNPOD_PORT="$port" bash scripts/pull_runpod_artifacts.sh || true
  rsync -az -e "$RSYNC_SSH" \
    root@"$host":/workspace/ExactKV/reports/ \
    reports/ 2>/dev/null || true
  python3 scripts/build_serving_microbench_pack.py 2>/dev/null || true
  python3 scripts/build_systems_diagnostic_pack.py 2>/dev/null || true
  echo "WATCH_COMPLETE $(date -u)" | tee -a "$WATCHLOG"
}

while (( $(date +%s) < DEADLINE )); do
  creds=$(resolve_creds || true)
  if [[ -n "${creds:-}" ]]; then
    host=$(echo "$creds" | awk '{print $1}')
    port=$(echo "$creds" | awk '{print $2}')
    echo "$(date -u +%H:%M:%S) candidate $host:$port" | tee -a "$WATCHLOG"
    if try_ssh "$host" "$port" | tee -a "$WATCHLOG" | grep -q POD_OK; then
      launch_marathon "$host" "$port"
      exit 0
    fi
  else
    echo "$(date -u +%H:%M:%S) waiting — paste SSH TCP in chat OR write launch/pod_ssh.env" | tee -a "$WATCHLOG"
  fi
  sleep 45
done
echo "WATCH_TIMEOUT no reachable pod SSH within 6h" | tee -a "$WATCHLOG"
exit 1
