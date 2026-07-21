#!/usr/bin/env bash
# 6-hour ExactKV GPU marathon for friend-feedback serving metrics + depth.
#
# Requires env:
#   RUNPOD_HOST  RUNPOD_PORT  (TCP SSH)
#   HF_TOKEN
# Optional:
#   RUNPOD_KEY=~/.ssh/runpod_exactkv
#   EXACTKV_ROOT=/workspace/ExactKV
#
# Phases (designed for ~6h on 7B/8B + RTX 24GB once models cached):
#   1) Setup + prefetch models
#   2) Serving microbench STRONG  (~2.5–3.5h)
#   3) Systems diagnostic expand / refresh if needed (~1–1.5h)
#   4) Pack build + integrity checks
#   5) Idle keepalive / optional extra n_requests until deadline
set -uo pipefail

ROOT="${EXACTKV_ROOT:-/workspace/ExactKV}"
PY="${PYTHON:-/workspace/.venv-runpod/bin/python3}"
[[ -x "$PY" ]] || PY=python3
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 6*3600 ))}"
LOG=/workspace/marathon_6h.log
mkdir -p /workspace "$ROOT/reports/external_panels/serving_microbench/logs"
mkdir -p "$ROOT/reports/external_panels/systems_diagnostic/logs"
mkdir -p "$HF_HOME"

exec > >(tee -a "$LOG") 2>&1
echo "==> MARATHON START $(date -u +%Y%m%dT%H%M%SZ) deadline_epoch=$DEADLINE_EPOCH"

remaining() { echo $(( DEADLINE_EPOCH - $(date +%s) )); }

cd "$ROOT"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# Prefetch is best-effort; models may already be cached. Never abort the marathon.
"$PY" - <<'PY' || echo "WARN prefetch failed — continuing with cache"
import os
from huggingface_hub import snapshot_download
token = open("/workspace/hf/token").read().strip() if os.path.exists("/workspace/hf/token") else os.environ.get("HF_TOKEN")
os.environ["HF_HOME"] = "/workspace/hf"
for mid in ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B"]:
    print("PREFETCH", mid, flush=True)
    try:
        print(snapshot_download(mid, token=token), flush=True)
    except Exception as e:
        print("PREFETCH_WARN", mid, type(e).__name__, e, flush=True)
print("PREFETCH_DONE", flush=True)
PY

# After each major phase, snapshot on the volume too (pod-local redundancy)
snapshot_reports() {
  local tag="$1"
  local dest="/workspace/report_snapshots/${tag}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dest"
  cp -a reports/external_panels/serving_microbench "$dest/" 2>/dev/null || true
  cp -a reports/external_panels/systems_diagnostic "$dest/" 2>/dev/null || true
  cp -a reports/systems "$dest/" 2>/dev/null || true
  echo "SNAPSHOT $dest"
}

run_serving_strong() {
  echo "==> PHASE serving_microbench STRONG $(date -u +%H:%M:%S) rem=$(remaining)s"
  for model in "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B"; do
    tag="$(echo "${model##*/}" | tr '.-' '_')"
    out="reports/external_panels/serving_microbench/${tag}_raw.json"
    "$PY" scripts/run_serving_microbench_panel.py \
      --device cuda --dtype float16 \
      --models "$model" \
      --compressors "noop,int8,int4_sim" \
      --context-buckets "2048,4096" \
      --max-new-tokens "64,128" \
      --n-requests "1,4,8" \
      --output-json "$out" \
      --checkpoint-json "$out" \
      --resume-json "$out" \
      || echo "WARN serving $model rc=$?"
    snapshot_reports "serving_${tag}"
    if (( $(remaining) < 600 )); then echo "NEAR_DEADLINE stop serving"; return; fi
  done
  "$PY" scripts/build_serving_microbench_pack.py || true
  # Only mark DONE if we actually have raw cells
  if ls reports/external_panels/serving_microbench/*_raw.json >/dev/null 2>&1; then
    echo DONE > reports/external_panels/serving_microbench/_DONE
    snapshot_reports serving_done
  else
    echo "ERROR: no serving *_raw.json produced" >&2
    return 1
  fi
}

run_systems_refresh() {
  echo "==> PHASE systems_diagnostic refresh $(date -u +%H:%M:%S) rem=$(remaining)s"
  if (( $(remaining) < 1800 )); then echo "skip systems — low time"; return; fi
  bash scripts/run_systems_diagnostic_panel.sh || echo "WARN systems rc=$?"
}

extra_load_until_deadline() {
  echo "==> PHASE extra serial_16 load $(date -u +%H:%M:%S) rem=$(remaining)s"
  while (( $(remaining) > 900 )); do
    for model in "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B"; do
      tag="$(echo "${model##*/}" | tr '.-' '_')"
      out="reports/external_panels/serving_microbench/${tag}_extra16_raw.json"
      "$PY" scripts/run_serving_microbench_panel.py \
        --device cuda --dtype float16 \
        --models "$model" \
        --compressors "int8,int4_sim" \
        --context-buckets "2048" \
        --max-new-tokens "64" \
        --n-requests "16" \
        --output-json "$out" \
        --checkpoint-json "$out" \
        --resume-json "$out" \
        || true
      if (( $(remaining) < 900 )); then break 2; fi
    done
    # rebuild pack merging all raws
    "$PY" scripts/build_serving_microbench_pack.py || true
    break  # one extra pass is enough; then idle heartbeat
  done
}

heartbeat_until_deadline() {
  echo "==> PHASE heartbeat $(date -u +%H:%M:%S)"
  while (( $(remaining) > 0 )); do
    echo "HEARTBEAT $(date -u +%H:%M:%S) rem=$(remaining)s gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    # light CUDA alloc touch so pod isn't idle-killed and logs stay alive
    "$PY" - <<'PY' || true
import torch, time
if torch.cuda.is_available():
    x=torch.randn(1024,1024, device='cuda', dtype=torch.float16)
    for _ in range(50):
        x=x@x
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()
print('heartbeat_ok')
PY
    sleep 120
  done
  echo "==> MARATHON COMPLETE $(date -u +%Y%m%dT%H%M%SZ)"
  echo DONE > /workspace/MARATHON_6H_DONE
}

run_serving_strong
run_systems_refresh
extra_load_until_deadline
"$PY" scripts/build_serving_microbench_pack.py || true
"$PY" scripts/build_systems_diagnostic_pack.py || true
heartbeat_until_deadline
