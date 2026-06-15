#!/usr/bin/env bash
# Phase 15B-unblock: isolated vLLM version compatibility sweep on RunPod.
# Does NOT modify /usr/bin/python3 or system torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SWEEP_ROOT="${VLLM_SWEEP_ROOT:-$ROOT/.venv-vllm-sweep}"
LOG_ROOT="${VLLM_SWEEP_LOG_DIR:-$ROOT/reports/vllm_sweep_logs}"
MANIFEST="${VLLM_SWEEP_MANIFEST:-$ROOT/reports/vllm_sweep_manifest.json}"
MAX_CANDIDATES="${VLLM_SWEEP_MAX:-5}"
EXPLICIT_VERSIONS="${VLLM_SWEEP_VERSIONS:-}"
INCLUDE_KNOWN_BAD=0
CLEANUP_FAILED_VENVS=1

usage() {
  echo "Usage: $0 [--versions v1,v2,...] [--max-candidates N] [--include-known-bad] [--no-cleanup]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --versions)
      EXPLICIT_VERSIONS="${2:-}"
      shift 2
      ;;
    --max-candidates)
      MAX_CANDIDATES="${2:-5}"
      shift 2
      ;;
    --include-known-bad)
      INCLUDE_KNOWN_BAD=1
      shift
      ;;
    --no-cleanup)
      CLEANUP_FAILED_VENVS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_ROOT" "$SWEEP_ROOT"

echo "=== Exp 061 vLLM version sweep ==="
echo "date: $(date -Iseconds)"
echo "system_python: /usr/bin/python3"
echo "sweep_root: $SWEEP_ROOT"
echo "log_root: $LOG_ROOT"

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

readarray -t CANDIDATE_META < <(
  /usr/bin/python3 - "$ROOT" "$MAX_CANDIDATES" "$EXPLICIT_VERSIONS" "$INCLUDE_KNOWN_BAD" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_candidates = int(sys.argv[2])
explicit = sys.argv[3].strip()
include_known_bad = sys.argv[4] == "1"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from exactkv.integrations.vllm_probe import select_sweep_candidates

explicit_versions = [v.strip() for v in explicit.split(",") if v.strip()] if explicit else None
candidates, excluded = select_sweep_candidates(
    explicit_versions=explicit_versions,
    max_candidates=max_candidates,
    include_known_bad=include_known_bad,
)
print(json.dumps({"candidates": candidates, "excluded_versions": excluded}))
PY
)

META_JSON="${CANDIDATE_META[0]}"
readarray -t CANDIDATES < <(
  /usr/bin/python3 - "$META_JSON" <<'PY'
import json
import sys
meta = json.loads(sys.argv[1])
for version in meta.get("candidates", []):
    print(version)
PY
)

EXCLUDED_JSON="$(/usr/bin/python3 - "$META_JSON" <<'PY'
import json, sys
meta = json.loads(sys.argv[1])
print(json.dumps(meta.get("excluded_versions", [])))
PY
)"

echo "candidates: ${CANDIDATES[*]:-<none>}"
echo "excluded_versions: $EXCLUDED_JSON"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "ERROR: no sweep candidates selected"
  exit 1
fi

WINNING=""
STOPPED_EARLY=0

for VERSION in "${CANDIDATES[@]}"; do
  SAFE="${VERSION//./_}"
  VENV="$SWEEP_ROOT/vllm-$SAFE"
  PYTHON="$VENV/bin/python"
  PIP="$VENV/bin/pip"
  LOG_DIR="$LOG_ROOT/$VERSION"
  INSTALL_LOG="$LOG_DIR/install.log"
  mkdir -p "$LOG_DIR"

  echo "--- candidate vLLM==$VERSION ---"
  echo "venv: $VENV"

  if [[ ! -x "$PYTHON" ]]; then
    echo "Creating venv ..."
    /usr/bin/python3 -m venv "$VENV"
  fi

  set +e
  {
    echo "Installing torch cu128 ..."
    "$PIP" install -q --upgrade pip wheel setuptools
    "$PIP" install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo "Installing vllm==$VERSION ..."
    "$PIP" install -q "vllm==$VERSION"
  } >"$INSTALL_LOG" 2>&1
  INSTALL_RC=$?
  set -e

  if [[ $INSTALL_RC -ne 0 ]]; then
    echo "install failed for vLLM==$VERSION (see $INSTALL_LOG)"
    /usr/bin/python3 - "$ROOT" "$VERSION" "$VENV" "$LOG_DIR" "$INSTALL_LOG" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from exactkv.integrations.vllm_probe import probe_sweep_candidate
install_log = Path(sys.argv[5]).read_text(encoding="utf-8", errors="replace")
probe_sweep_candidate(
    version=sys.argv[2],
    venv_python=Path(sys.argv[3]) / "bin" / "python",
    log_dir=Path(sys.argv[4]),
    install_success=False,
    install_error=f"pip install vllm=={sys.argv[2]} failed",
    stdout=install_log,
    stderr="",
)
PY
    continue
  fi

  /usr/bin/python3 - "$ROOT" "$VERSION" "$PYTHON" "$LOG_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from exactkv.integrations.vllm_probe import probe_sweep_candidate
result = probe_sweep_candidate(
    version=sys.argv[2],
    venv_python=Path(sys.argv[3]),
    log_dir=Path(sys.argv[4]),
    install_success=True,
    run_generation_smoke=True,
)
print("classification:", result.get("classification"))
print("generation_smoke_passed:", result.get("generation_smoke_passed"))
PY

  CLASSIFICATION="$(/usr/bin/python3 - "$LOG_DIR/candidate_result.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("classification", ""))
PY
)"

  echo "candidate $VERSION => $CLASSIFICATION"
  if [[ "$CLASSIFICATION" == "pass" ]]; then
    WINNING="$VERSION"
    STOPPED_EARLY=1
    echo "winning candidate found: $VERSION — stopping sweep"
    break
  fi
  if [[ "$CLEANUP_FAILED_VENVS" -eq 1 ]]; then
    echo "removing failed candidate venv to free disk: $VENV"
    rm -rf "$VENV"
  fi
done

/usr/bin/python3 - "$MANIFEST" "$META_JSON" "$WINNING" "$STOPPED_EARLY" "$LOG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
meta = json.loads(sys.argv[2])
winning = sys.argv[3] or None
stopped_early = sys.argv[4] == "1"
log_root = Path(sys.argv[5])

candidate_results = []
for version in meta.get("candidates", []):
    result_path = log_root / str(version) / "candidate_result.json"
    if result_path.is_file():
        candidate_results.append(json.loads(result_path.read_text(encoding="utf-8")))

payload = {
    "candidates": meta.get("candidates", []),
    "excluded_versions": meta.get("excluded_versions", []),
    "candidate_results": candidate_results,
    "winning_candidate": winning,
    "stopped_early": stopped_early,
    "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", manifest_path)
PY

echo "exp061_vllm_version_sweep_complete winning=${WINNING:-none} stopped_early=$STOPPED_EARLY"
