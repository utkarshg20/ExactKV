#!/usr/bin/env bash
# Run Experiment 056 CUDA restored-verifier runtime gate on RunPod GPU.
# Diagnostic exactness gate only — not a performance benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${EXP056_VENV:-/workspace/kivi_exp024/.venv-kivi}"
PYTHON="${EXP056_PYTHON:-$VENV/bin/python}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
LOG="${EXP056_LOG:-/workspace/exp056_cuda_restored_verifier_runtime_gate.log}"
exec > >(tee -a "$LOG") 2>&1
echo "=== Exp 056 CUDA restored-verifier runtime gate ==="
echo "date: $(date -Iseconds)"
echo "python: $PYTHON"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON" - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16_supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
print("torch_version:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable — cannot collect Exp 056 evidence")
PY
if ! "$PYTHON" -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B', local_files_only=True)" 2>/dev/null; then
  echo "caching Qwen/Qwen2.5-0.5B (one-time download)..."
  unset TRANSFORMERS_OFFLINE
  "$PYTHON" -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m='Qwen/Qwen2.5-0.5B'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')"
  export TRANSFORMERS_OFFLINE=1
fi
"$PYTHON" scripts/research/run_exp056_cuda_restored_verifier_runtime_gate.py
EXACTKV_RUN_CUDA_SMOKE=1 "$PYTHON" -m pytest tests/test_exp056_cuda_restored_verifier_runtime_gate.py -q
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from exactkv.runtime.experimental import validate_exp056_report
p = Path("reports/experiment_056_cuda_restored_verifier_runtime_gate.json")
r = json.loads(p.read_text())
errs = validate_exp056_report(r)
gate = (
    r.get("cuda_available") is True
    and r.get("total_cells", 0) > 0
    and r.get("exactkv_failures") == 0
    and r.get("token_exact_match_count") == r.get("total_cells")
    and r.get("status") == "pass"
    and not r.get("cuda_blockers")
)
print("schema_errors:", errs)
print("gate_passed:", gate)
print("status:", r.get("status"))
print("dtypes:", r.get("dtype_configs"))
print("exact:", r.get("token_exact_match_count"), "/", r.get("total_cells"))
if not gate:
    raise SystemExit("Exp 056 CUDA exactness gate FAILED — do not proceed to Exp 057")
PY
echo "exp056_complete exit=0"
