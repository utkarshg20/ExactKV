# ExactKV research release

**Git tag:** `v-release`

**Research release artifact:** 1,500-cell core panel + 8,132-cell external headline grid

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
pip install git+https://github.com/utkarshg20/ExactKV.git@v-release
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

## Release assets

After checkout, build or download the frozen bundle:

```bash
python3 scripts/build_release_artifact_bundle.py
shasum -a 256 -c SHA256SUMS
```

GitHub Release also attaches `exactkv-research-release-artifact-bundle.tar.gz` and `SHA256SUMS`.
