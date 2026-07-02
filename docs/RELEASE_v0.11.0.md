# ExactKV v0.11.0 (research release v3.0)

**Canonical package version:** `0.11.0` (`exactkv.__version__`, `pyproject.toml`)

**Research bundle label:** v3.0 (1,500-cell core panel + 8,132-cell external headline grid)

**Commit:** `6a67201`

## What this is

ExactKV is a **compressor-agnostic crash-test and leaderboard** for LLM KV-cache compression under greedy decoding. It is a **research-grade evaluation framework**, not a production serving system.

Primary metrics: **divergence rate**, **acceptance rate**, **first-divergence index**.

`exactkv_failures = 0` is a **harness safety gate** on cited panels, not proof that compression is practically useful by itself.

## Start here

- [Evaluator guide](EVALUATOR_GUIDE.md)
- [Technical report](../paper/ExactKV_Technical_Report.md)
- [Claim boundaries](CLAIM_BOUNDARIES.md)
- [Headline leaderboard JSON](../reports/public_release/leaderboard_final.json) (6 ranked rows; mock/probe rows in `diagnostic_entries`)

## Cheap repro (CPU, no GPU)

```bash
pip install -e ".[dev]"
bash scripts/smoke_test.sh
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/check_site_claims.py
```

## GPU repro (headline panel)

```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda --dtype float16
```

Artifact: `reports/scale_7b/raw.json`

## License

MIT — see [LICENSE](../LICENSE).
