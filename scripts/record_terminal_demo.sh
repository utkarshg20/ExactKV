#!/usr/bin/env bash
# Record ExactKV terminal demo as asciinema cast or script transcript.
# Does NOT require GUI screen capture. Does NOT produce MP4 by itself.
#
# Usage (from repo root):
#   bash scripts/record_terminal_demo.sh
#   bash scripts/record_terminal_demo.sh --fast
#
# Output:
#   docs/assets/exactkv_crash_test_demo.cast   (asciinema)
#   docs/assets/exactkv_crash_test_demo.typescript (script fallback)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="$ROOT/docs/assets"
mkdir -p "$OUT_DIR"

SPEED="cinematic"
if [[ "${1:-}" == "--fast" ]]; then
  SPEED="fast"
fi

DEMO_CMD="$PYTHON scripts/exactkv_terminal_crash_test.py --speed $SPEED --plain"
CAST_PATH="$OUT_DIR/exactkv_crash_test_demo.cast"
TYPESCRIPT_PATH="$OUT_DIR/exactkv_crash_test_demo.typescript"

echo "ExactKV terminal demo recorder"
echo "  Command: $DEMO_CMD"
echo "  Output dir: $OUT_DIR"
echo ""

if command -v asciinema >/dev/null 2>&1; then
  echo "Using asciinema → $CAST_PATH"
  echo "  Tip: convert with 'agg' or upload to asciinema.org; MP4 requires separate encode step."
  asciinema rec -c "$DEMO_CMD" "$CAST_PATH"
  echo "Recorded: $CAST_PATH"
  exit 0
fi

if command -v script >/dev/null 2>&1; then
  echo "asciinema not found — falling back to BSD/GNU script → $TYPESCRIPT_PATH"
  # shellcheck disable=SC2034
  export COLUMNS="${COLUMNS:-110}"
  script -q "$TYPESCRIPT_PATH" bash -c "$DEMO_CMD"
  echo "Recorded transcript: $TYPESCRIPT_PATH"
  echo ""
  echo "Neither asciinema nor GUI capture was used. For MP4, use screen capture or:"
  echo "  python3 scripts/render_exactkv_crash_test_video.py"
  exit 0
fi

echo "ERROR: Neither 'asciinema' nor 'script' is available." >&2
echo "Install asciinema (recommended) or use manual screen capture per:" >&2
echo "  docs/assets/exactkv_terminal_demo_recording_plan.md" >&2
exit 1
