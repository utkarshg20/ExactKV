#!/usr/bin/env bash
# Record ExactKV live case-study demo as asciinema cast or script transcript.
#
# Usage (from repo root):
#   bash scripts/record_terminal_demo.sh
#   bash scripts/record_terminal_demo.sh --fast
#   bash scripts/record_terminal_demo.sh --single
#
# Output:
#   docs/assets/exactkv_live_demo.cast
#   docs/assets/exactkv_live_demo.typescript
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="$ROOT/docs/assets"
mkdir -p "$OUT_DIR"

SPEED="cinematic"
MODE="carousel"
if [[ "${1:-}" == "--fast" ]]; then
  SPEED="fast"
fi
if [[ "${1:-}" == "--single" ]]; then
  MODE="single"
fi

if [[ "$MODE" == "single" ]]; then
  DEMO_CMD="$PYTHON scripts/exactkv_live_demo.py --speed $SPEED --mode cases --case p02_p2_json_tool"
else
  DEMO_CMD="$PYTHON scripts/exactkv_live_demo.py --speed launch"
fi

CAST_PATH="$OUT_DIR/exactkv_live_demo.cast"
TYPESCRIPT_PATH="$OUT_DIR/exactkv_live_demo.typescript"

echo "ExactKV live demo recorder"
echo "  Command: $DEMO_CMD"
echo "  Output dir: $OUT_DIR"
echo ""

export COLUMNS="${COLUMNS:-110}"

if command -v asciinema >/dev/null 2>&1; then
  echo "Using asciinema → $CAST_PATH"
  asciinema rec -c "$DEMO_CMD" "$CAST_PATH"
  echo "Recorded: $CAST_PATH"
  exit 0
fi

if command -v script >/dev/null 2>&1; then
  echo "asciinema not found — falling back to script → $TYPESCRIPT_PATH"
  script -q "$TYPESCRIPT_PATH" bash -c "$DEMO_CMD"
  echo "Recorded transcript: $TYPESCRIPT_PATH"
  exit 0
fi

echo "ERROR: Neither 'asciinema' nor 'script' is available." >&2
echo "Use screen capture per docs/assets/exactkv_terminal_demo_recording_plan.md" >&2
exit 1
