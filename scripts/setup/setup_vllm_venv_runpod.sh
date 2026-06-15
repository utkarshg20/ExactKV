#!/usr/bin/env bash
# Phase 15B: isolated vLLM venv setup on RunPod (install-safe).
# Does NOT modify /usr/bin/python3 or system torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VLLM_VENV_PATH:-$ROOT/.venv-vllm}"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
LOG="${VLLM_SETUP_LOG:-$ROOT/reports/exp060_vllm_venv_setup.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== Exp 060 vLLM venv setup ==="
echo "date: $(date -Iseconds)"
echo "system_python: /usr/bin/python3"
echo "venv: $VENV"

if [[ ! -x /usr/bin/python3 ]]; then
  echo "ERROR: /usr/bin/python3 not found"
  exit 1
fi

/usr/bin/python3 - <<'PY'
import torch
print("system_torch:", torch.__version__)
print("system_cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("system_gpu:", torch.cuda.get_device_name(0))
PY

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating venv at $VENV ..."
  /usr/bin/python3 -m venv "$VENV"
fi

"$PIP" install -q --upgrade pip wheel setuptools

echo "Installing torch cu128 into venv (match system driver) ..."
"$PIP" install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing vLLM into venv ..."
if ! "$PIP" install -q vllm; then
  echo "ERROR: vLLM pip install failed"
  exit 1
fi

echo "=== venv verification ==="
echo "venv_python: $PYTHON"
"$PYTHON" - <<'PY'
import sys
print("python_version:", sys.version.split()[0])
import torch
print("venv_torch:", torch.__version__)
print("venv_cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable inside vLLM venv")
print("venv_gpu:", torch.cuda.get_device_name(0))
import vllm
print("vllm_version:", getattr(vllm, "__version__", "unknown"))
from vllm import LLM, SamplingParams
print("vllm_import: ok LLM=%s SamplingParams=%s" % (LLM, SamplingParams))
PY

echo "exp060_vllm_venv_setup_complete exit=0"
