# Reproducing ExactKV research release

**Public cite:** ExactKV research release · **git tag:** `v-release`

This document is the one-page reproduction path for external reviewers. Full claim boundaries: [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md).

## 1. Pin the artifact

```bash
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV
git checkout v-release
```

Or install without cloning:

```bash
pip install git+https://github.com/utkarshg20/ExactKV.git@v-release
```

Optional frozen bundle (checksums in release assets):

```bash
# From GitHub Release assets after download:
shasum -a 256 -c SHA256SUMS
tar xzf exactkv-research-release-artifact-bundle.tar.gz
```

## 2. Environment

**Option A — pip (fastest, CPU repro):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

**Option B — conda:**

```bash
conda env create -f environment.yml
conda activate exactkv-research-release
pip install -e ".[dev]"
```

**Option C — Docker (CPU repro only):**

```bash
docker build -t exactkv-research-release .
docker run --rm -v "$PWD:/work" -w /work exactkv-research-release \
  bash -lc 'pip install -e ".[dev]" && bash scripts/smoke_test.sh'
```

Requires **Python 3.10+**. GPU is **not** required for artifact validation below.

## 3. Expected outputs (no GPU, ~2 minutes)

```bash
bash scripts/smoke_test.sh
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/check_site_claims.py
python3 scripts/validate_final_release_package.py
python3 scripts/audit_public_claims.py
python3 -m pytest tests/test_acceptance_logic.py tests/test_capture_divergence_topk.py tests/test_site_artifacts.py -q
```

**Pass criteria:**

| step | expected |
|------|----------|
| `smoke_test.sh` | `SMOKE TEST PASSED` |
| `exactkv_repro.py --reports-only` | exit 0, headline tables match committed JSON |
| `check_site_claims.py` | `PASSED` |
| `validate_final_release_package.py` | exit 0 |
| pytest | all green |

Regenerate public review artifacts (optional):

```bash
python3 scripts/build_public_review_artifacts.py
```

## 4. Headline artifacts (source of truth)

| artifact | path |
|----------|------|
| Core leaderboard panel | `reports/scale_7b/raw.json` (1,500 cells) |
| Public leaderboard JSON | `reports/public_release/leaderboard_final.json` |
| Wilson 95% confidence intervals | `reports/public_release/confidence_intervals.json` |
| External panel index | `reports/external_panels/summary_all.json` |
| Phase F kernel microbench | `reports/systems/latency_microbench.json` |
| Verifier timing proxy (evidence-plus) | `reports/systems/verifier_overhead.json` |
| Recompression overhead status | `reports/systems/recompression_overhead.json` |
| Stored-byte memory trace | `reports/systems/gpu_memory_trace.json` |
| Technical report | `paper/ExactKV_Technical_Report.md` |
| Artifact audit note | `docs/ARTIFACT_AUDIT.md` |

Regenerate summaries (optional):

```bash
python3 scripts/build_external_panel_summary.py --write
bash scripts/sync_site_data.sh
```

## 5. GPU reproduction (optional, requires HF access)

```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda --dtype float16
# writes reports/scale_7b/raw.json
```

External panels: `python3 scripts/run_external_panel.py --help`

## 6. What this repro does **not** claim

- End-to-end inference speedup or active GPU memory savings
- Official LongBench / BFCL / MBPP leaderboard scores
- Production serving readiness
- VeriCache reproduction

See [`docs/THREATS_TO_VALIDITY.md`](docs/THREATS_TO_VALIDITY.md) for limitations and reviewer checklist.
