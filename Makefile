# ExactKV developer shortcuts (prelaunch — not a release Makefile)

.PHONY: smoke demo leaderboard audit test-smoke install help

help:
	@echo "ExactKV prelaunch targets:"
	@echo "  make smoke        - CPU-safe smoke test (bash scripts/smoke_test.sh)"
	@echo "  make demo         - Terminal crash-test demo (fast pacing)"
	@echo "  make leaderboard  - Terminal + md + html leaderboard"
	@echo "  make audit        - Claims, links, and report hygiene audits"
	@echo "  make test-smoke   - Pytest subset only (no bash wrapper)"
	@echo "  make install      - pip install -e '.[dev]'"

install:
	pip install -U pip
	pip install -e ".[dev]"

smoke:
	bash scripts/smoke_test.sh

demo:
	python3 scripts/exactkv_terminal_crash_test.py --speed fast

leaderboard:
	python3 scripts/exactkv_leaderboard.py

audit:
	python3 scripts/audit_public_claims.py
	python3 scripts/check_docs_links.py
	python3 scripts/check_report_hygiene.py --require-public

test-smoke:
	python3 -m pytest tests/test_exactkv_terminal_crash_test.py tests/test_exactkv_leaderboard.py \
		tests/test_public_claims_audit.py tests/test_docs_links.py tests/test_report_hygiene.py \
		tests/test_acceptance_logic.py -q
