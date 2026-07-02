#!/usr/bin/env bash
# ExactKV prelaunch smoke test (CPU-safe, no GPU, no model download).
# Run from repository root: bash scripts/smoke_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
FAILURES=0

section() {
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "══════════════════════════════════════════════════════════════"
}

pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; FAILURES=$((FAILURES + 1)); }

section "1. Environment"
if command -v "$PYTHON" >/dev/null 2>&1; then
  pass "Python: $($PYTHON --version 2>&1)"
else
  fail "Python not found ($PYTHON)"
fi

section "2. Import exactkv"
if "$PYTHON" -c "import exactkv; assert exactkv.__version__; print(exactkv.__version__)" >/dev/null 2>&1; then
  pass "import exactkv (__version__=$("$PYTHON" -c 'import exactkv; print(exactkv.__version__)'))"
else
  fail "import exactkv (run: pip install -e '.[dev]')"
fi

section "3. Terminal crash-test demo (replay)"
if "$PYTHON" scripts/exactkv_terminal_crash_test.py --no-delay --plain | grep -q "EXACTKV CRASH TEST"; then
  pass "exactkv_terminal_crash_test.py"
else
  fail "exactkv_terminal_crash_test.py"
fi

section "4. Leaderboard"
if "$PYTHON" scripts/exactkv_leaderboard.py --plain | grep -q "EXACTKV CRASH-TEST LEADERBOARD"; then
  pass "exactkv_leaderboard.py"
else
  fail "exactkv_leaderboard.py"
fi

section "5. Prelaunch audit scripts"
if "$PYTHON" scripts/audit_public_claims.py; then
  pass "audit_public_claims.py"
else
  fail "audit_public_claims.py"
fi

if "$PYTHON" scripts/check_docs_links.py; then
  pass "check_docs_links.py"
else
  fail "check_docs_links.py"
fi

if "$PYTHON" scripts/check_report_hygiene.py; then
  pass "check_report_hygiene.py"
else
  fail "check_report_hygiene.py"
fi

section "6. Pytest subset (no GPU / no model download)"
PYTEST_ARGS=(
  tests/test_exactkv_terminal_crash_test.py
  tests/test_exactkv_leaderboard.py
  tests/test_public_claims_audit.py
  tests/test_docs_links.py
  tests/test_report_hygiene.py
  tests/test_acceptance_logic.py
  tests/test_capture_divergence_topk.py
  -q
)
if "$PYTHON" -m pytest "${PYTEST_ARGS[@]}"; then
  pass "pytest subset"
else
  fail "pytest subset"
fi

section "SUMMARY"
if [[ "$FAILURES" -eq 0 ]]; then
  echo "  SMOKE TEST PASSED"
  exit 0
else
  echo "  SMOKE TEST FAILED ($FAILURES section(s))"
  exit 1
fi
