#!/usr/bin/env bash
# Poll RunPod wave-3 faithful panel every 3 minutes. Logs locally.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONITOR="$ROOT/reports/faithful_wave3_monitor.log"
SSH_HOST="${RUNPOD_SSH:-runpod-a5000}"
INTERVAL="${WAVE3_MONITOR_INTERVAL:-180}"

mkdir -p "$(dirname "$MONITOR")"

poll_once() {
  ssh -o ConnectTimeout=20 -o BatchMode=yes "$SSH_HOST" '
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
    RUN=$(pgrep -fc "run_external_panel.py" 2>/dev/null || echo 0)
    HF=$(test -f /workspace/.cache/huggingface/token && echo yes || echo no)
    HF_OK=$(HF_HOME=/workspace/.cache/huggingface /workspace/.venv-faithful/bin/python3 -c "
from huggingface_hub import get_token, whoami
try:
    assert get_token()
    whoami()
    print(\"ok\")
except Exception:
    print(\"fail\")
" 2>/dev/null || echo fail)
    LAST=$(grep -hE "\[longbench\]|\[bfcl\]|\[mbpp\]|cell [0-9]+/|SUCCESS|FAILED|WAVE3_TURBOQUANT_DONE|ERROR|Traceback|OOM|CUDA out of memory" \
      /workspace/ExactKV/reports/faithful_wave3_launch_*.log \
      /workspace/ExactKV/reports/faithful_wave3_*.log 2>/dev/null | tail -5)
    PROG=$(python3 -c "
import json, glob, os
ok=0; total=576
for p in sorted(glob.glob(\"/workspace/ExactKV/reports/external_panels/faithful/wave3/*_wave3_raw.json\")):
    d=json.load(open(p))
    n=sum(1 for c in d.get(\"cells\",[]) if c.get(\"status\")==\"ok\")
    ok+=n
    print(f\"{os.path.basename(p)}:{n}\", end=\" \")
print(f\"| total_ok={ok}/{total}\")
" 2>/dev/null)
    TMUX=$(tmux has-session -t faithful_wave3 2>/dev/null && echo alive || echo dead)
    echo "tmux=$TMUX gpu=$GPU procs=$RUN hf_token=$HF hf_auth=$HF_OK"
    echo "progress=$PROG"
    echo "$LAST"
  ' 2>&1
}

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] wave3_monitor_start" >> "$MONITOR"

while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  OUT=$(poll_once) || OUT="SSH_FAIL: ${OUT:-connection error}"
  {
    echo "[$TS]"
    echo "$OUT"
    echo "---"
  } >> "$MONITOR"

  if echo "$OUT" | grep -qiE "SSH_FAIL|CUDA out of memory|Traceback|FAILED"; then
    echo "[$TS] ALERT check log" >> "$MONITOR"
  fi
  if echo "$OUT" | grep -q "WAVE3_TURBOQUANT_DONE"; then
    echo "[$TS] wave3_complete" >> "$MONITOR"
    exit 0
  fi
  if echo "$OUT" | grep -q "tmux=dead" && ! echo "$OUT" | grep -q "WAVE3_TURBOQUANT_DONE"; then
    echo "[$TS] ALERT tmux_dead" >> "$MONITOR"
  fi

  sleep "$INTERVAL"
done
