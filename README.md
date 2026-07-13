# ExactKV

![CI](https://github.com/utkarshg20/ExactKV/actions/workflows/ci.yml/badge.svg)
[![Release](https://img.shields.io/github/v/release/utkarshg20/ExactKV?label=research%20release)](https://github.com/utkarshg20/ExactKV/releases/tag/v-release)

**When does compressed KV start lying?**

ExactKV is a compressor-agnostic **crash-test and leaderboard** for LLM KV-cache compression. It runs a draft/verify/commit loop, measures **first-divergence index**, **acceptance rate**, and **exactness failures**, and ships every headline number as committed JSON.

| Resource | Link |
|----------|------|
| Start here (reviewers) | [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) |
| Reproduce (CPU, ~2 min) | [`REPRODUCE.md`](REPRODUCE.md) |
| Artifact audit note | [`docs/ARTIFACT_AUDIT.md`](docs/ARTIFACT_AUDIT.md) |
| Technical report | [`paper/ExactKV_Technical_Report.md`](paper/ExactKV_Technical_Report.md) |
| Landing page | [`site/index.html`](site/index.html) |
| GitHub Release | [`v-release`](https://github.com/utkarshg20/ExactKV/releases/tag/v-release) |

**One public git tag: `v-release`.** Internal V-lineage docs are not semver. See [`docs/VERSIONING.md`](docs/VERSIONING.md).

ExactKV is a **research-grade evaluation framework**. It is **not** a production serving system and does **not** reproduce VeriCache.

---

## Results (headline panels)

Models: **Llama-3.1-8B** and **Mistral-7B-Instruct-v0.3**, greedy decoding.

**Primary metrics:** divergence rate, acceptance rate, first-divergence index.

**Safety gate:** `exactkv_failures = 0` on cited panels confirms verify/commit wiring — not that compression is practically useful by itself.

| Metric | Result |
|--------|--------|
| External GPU cells (headline) | **8,132** |
| Core leaderboard panel | **1,500** (`reports/scale_7b/raw.json`) |
| `int4_sim` drift, MBPP (code) | **6%** |
| `int4_sim` drift, HF LongBench (reading) | **~90%** |
| H2O-style eviction @ 75% kept, LongBench | **100%** |
| BFCL long-gen drift (mnt 16→256) | **9% → 62%** (7× within-task scaling) |

The 6%→90% code/reading span is **observational** (task, context, and `max_new` change together across families). The cleanest controlled within-task axis in the current release is BFCL generation length.
| Full-KV valid tool calls preserved | **106/106** (BFCL validity panel) |
| Wilson 95% CIs (headline + smoke) | [`confidence_intervals.json`](reports/public_release/confidence_intervals.json) |

### Faithful adapter appendix (separate from 8,132 headline)

| Metric | Result |
|--------|--------|
| Appendix cells total | **1,568** (wave-1 864 + wave-2 128 + wave-3 576) |
| `int8` combined drift (wave-3 grid) | **8.3%** |
| TurboQuant structured smoke (wave-2) | **3.1%** |
| TurboQuant LongBench (wave-3, both models) | **~63–67%** |
| SnapKV / KIVI r32 (wave-1) | **90–100%** (adapter diagnostics) |

### Compressor tiers

| Tier | Examples | Note |
|------|----------|------|
| Built-in real | `noop`, `int8` | Headline 1,500-cell panel |
| Built-in simulated | `int4_sim`, `int6_sim`, `h2o_sim` | Diagnostic, not upstream ports |
| Fallback / proxy | `spectralquant`, `shard` | Mock or probe-only leaderboard rows |
| Faithful adapter | `snapkv_experimental`, `turboquant_experimental`, `kivi_offline_r32` | Appendix smoke grid only |

External panel drift rates are **not** official LongBench/BFCL/MBPP scores. Compression ratios in the report are **stored tensor byte ratios**, not active GPU memory savings.

---

## How it works

| Step | What happens |
|------|----------------|
| Draft | Generate candidate tokens from compressed (lossy) KV |
| Verify | Compare each draft token to full-KV greedy argmax |
| Commit | Accept matching prefix; on mismatch, correct and advance full KV |
| Measure | Record first divergence, acceptance, agreement, failures |

ExactKV measures **whether compression is compatible with exact decoding**, not whether it is faster. Phase F kernel numbers are a **microbenchmark only**.

---

## Install

```bash
pip install git+https://github.com/utkarshg20/ExactKV.git@v-release
```

Development checkout:

```bash
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV
git checkout v-release
pip install -e ".[dev]"
```

Requires **Python 3.10+**, **PyTorch**, and **transformers**. GPU optional for terminal replay; required for benchmark panels.

---

## Usage

**Live terminal demo:**

```bash
python3 scripts/exactkv_live_demo.py --speed hero     # launch cut (~25s): dropoff→pickup + 6%→90%
python3 scripts/exactkv_live_demo.py --speed launch   # longer weather multi-drift
python3 scripts/exactkv_live_demo.py --mode cases --speed cinematic
```

Hero recording / Sora open: `launch/demo_hero_10.md`
**Verify artifacts (no inference):**

```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/check_site_claims.py
python3 scripts/validate_final_release_package.py
```

**Core leaderboard:**

```bash
python3 scripts/exactkv_leaderboard.py
```

**GPU headline panel (~1,500 cells):**

```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda --dtype float16
```

---

## Reproducing the benchmarks

Claim boundaries: [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md).

| Panel | Artifact |
|-------|----------|
| Core leaderboard (1,500 cells) | `reports/scale_7b/raw.json` |
| Public leaderboard JSON | `reports/public_release/leaderboard_final.json` |
| Wilson 95% CIs | `reports/public_release/confidence_intervals.json` |
| External panels (8,132 headline) | `reports/external_panels/` |
| Faithful appendix (1,568 cells) | `reports/external_panels/faithful/` |
| Extended int6 / int4-per-vec panel | `reports/external_panels/v30/` |
| Kernel microbench (Phase F) | `reports/systems/latency_microbench.json` |
| Verifier timing proxy | `reports/systems/verifier_overhead.json` |
| Recompression overhead status | `reports/systems/recompression_overhead.json` |

Full path: [`REPRODUCE.md`](REPRODUCE.md) · Limitations: [`docs/THREATS_TO_VALIDITY.md`](docs/THREATS_TO_VALIDITY.md)

Regenerate public review artifacts:

```bash
python3 scripts/build_public_review_artifacts.py
python3 scripts/build_external_panel_summary.py --write
bash scripts/sync_site_data.sh
```

---

## Layout

```
exactkv/          # runtime, compressors, benchmarks, platform
scripts/          # repro, panels, live demo, validators
paper/            # technical report (Markdown, LaTeX, PDF)
site/             # public landing page
reports/
  scale_7b/       # 1,500-cell headline panel
  external_panels/  # 8,132-cell grid + faithful appendix
  public_release/   # leaderboard + confidence intervals
docs/             # claim boundaries, evaluator guide, artifact audit
```

---

## What ExactKV is not

- Not a production serving system; no throughput or active VRAM savings claims
- Does not reproduce VeriCache (draft/verify semantics only)
- Not official benchmark scores — external panels are drift smoke tests
- `spectralquant` / `shard` are fallback/proxy diagnostics only
- `exactkv_failures = 0` is a harness safety gate, not “compression is always safe”

---

## License

MIT — see [`LICENSE`](LICENSE).
