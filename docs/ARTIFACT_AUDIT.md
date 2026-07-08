# ExactKV public artifact audit (static review note)

**Audience:** external reviewers evaluating the GitHub research release without a full GPU rerun.

**Public cite:** ExactKV research release · git tag `v-release`

**Last updated:** 2026-07-08 (post wave-3 faithful appendix + site §11–15 layout)

---

## Scope of this note

This is a **repository self-audit**, not an independent third-party certification. It summarizes what a static reviewer can verify from the public tree, CI, release assets, and committed JSON artifacts.

## Verified public signals

| Signal | Status | Where |
|--------|--------|-------|
| Latest `main` CI green | ✅ | GitHub Actions workflow `CI` |
| Single public git tag | ✅ | `v-release` only ([`docs/VERSIONING.md`](VERSIONING.md)) |
| Formal GitHub Release | ✅ | Release notes + asset bundle + `SHA256SUMS` |
| Headline vs appendix cell accounting | ✅ | 8,132 headline + 1,568 faithful appendix (separate) |
| Diagnostic compressor separation | ✅ | `leaderboard_final.json` ranks `noop`/`int8`/`int4_sim` only |
| Claim-boundary audits in CI | ✅ | `check_site_claims.py`, `audit_public_claims.py` |
| CPU repro path (no GPU) | ✅ | [`REPRODUCE.md`](../REPRODUCE.md), `smoke_test.sh` |
| Wilson CI export | ✅ | [`reports/public_release/confidence_intervals.json`](../reports/public_release/confidence_intervals.json) |

## What the artifact demonstrates

1. **Task-type sensitivity** — `int4_sim` drift spans code → tool-call → reading on real 7B/8B panels.
2. **Mechanistic autopsy** — 1,103 divergent cells with top-k logit traces (near-tie / distribution shift / attention destruction).
3. **Verifier safety gate** — `exactkv_failures = 0` on all cited completed panels (harness invariant).
4. **Faithful adapter appendix** — TurboQuant task-conditional: near-clean on structured tasks, ~65% LongBench drift (wave-3, both models).

## What is intentionally not claimed

- Production serving readiness or end-to-end speedup
- Active GPU memory / VRAM savings at serving time
- Official LongBench / BFCL / MBPP leaderboard scores
- Full per-token verifier cost model on the 1,500-cell headline panel (see [`reports/systems/verifier_overhead.json`](../reports/systems/verifier_overhead.json))
- Isolated recompression overhead in headline panels (see [`reports/systems/recompression_overhead.json`](../reports/systems/recompression_overhead.json))

## Recommended reviewer path (~5 minutes, CPU)

```bash
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV && git checkout v-release
pip install -e ".[dev]"
bash scripts/smoke_test.sh
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/validate_final_release_package.py
```

## Open gaps (honest)

| Gap | Impact | Planned mitigation |
|-----|--------|-------------------|
| MBPP pass@1 / HumanEval validity downstream | Limits “task correctness beyond BFCL” story | Future panel extension |
| Headline-panel verifier micro-profile | Systems reviewers want cost vs acceptance | Evidence-plus timing proxy only today |
| DOI / Zenodo archival | Citation permanence | Optional post-release step |
| Scheduled tiny-model CI smoke | Stronger “runs on CI” story beyond unit tests | Optional workflow |

## Source-of-truth index

See technical report Appendix A/E and [`REPRODUCE.md`](../REPRODUCE.md) §4 for artifact paths.
