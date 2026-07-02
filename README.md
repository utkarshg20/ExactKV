# ExactKV

![CI](https://github.com/utkarshg20/ExactKV/actions/workflows/ci.yml/badge.svg)
[![Release v0.11.0](https://img.shields.io/github/v/release/utkarshg20/ExactKV?label=release)](https://github.com/utkarshg20/ExactKV/releases/tag/v0.11.0)

**Version 0.11.0** (research release v3.0) · [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) · [Technical report](paper/ExactKV_Technical_Report.md) · [GitHub Release](https://github.com/utkarshg20/ExactKV/releases/tag/v0.11.0)

Package version, git tag, and evaluator guide all use **`v0.11.0`**. The **v3.0** label names the research artifact bundle (headline panels + external grid). Tag `v0.13.0-rc1` is a **future preview only**, not the cited public artifact (see [release notes](docs/RELEASE_v0.11.0.md)).

**When does compressed KV start lying?**

ExactKV is a compressor-agnostic **crash-test and leaderboard** for LLM KV-cache compression. It runs a draft/verify/commit loop: draft from compressed KV, verify each token against the full-KV reference, accept the matching prefix, correct on mismatch, and record **first-divergence index**, **acceptance rate**, **verifier agreement**, and exactness failures per cell.

Unlike Shard, TurboQuant, or KIVI, ExactKV does **not** ship a new compression kernel. It measures **where and how** compressors drift under greedy decoding, with every published number tracing to a saved artifact.

**Full write-up:** [`paper/ExactKV_Technical_Report.md`](paper/ExactKV_Technical_Report.md) · **Landing page:** [`site/index.html`](site/index.html)

ExactKV is a **research-grade evaluation framework**. It is **not** a production serving system and does **not** reproduce VeriCache.

---

## Results

All on **Llama-3.1-8B** and **Mistral-7B-Instruct-v0.3**, greedy decoding.

**Primary metrics:** divergence rate, acceptance rate, first-divergence index.
**Safety gate:** `exactkv_failures = 0` on cited panels confirms verify/commit wiring, not that compression is practically useful by itself.

| metric | result |
|--------|--------|
| External GPU cells (headline) | **8,132** |
| Core leaderboard panel | **1,500** (`reports/scale_7b/raw.json`) |
| `int4_sim` drift, MBPP (code) | **6%** |
| `int4_sim` drift, HF LongBench (reading) | **~90%** |
| H2O-style eviction @ 75% kept, LongBench | **100%** (worse than int4 at matched budget) |
| BFCL long-gen drift (mnt 16→256) | **9% → 62%** (7× within-task scaling) |
| Full-KV valid tool calls preserved | **106/106** (v2.7 BFCL validity panel) |
| Faithful adapter smoke (appendix) | **864** wave-1 + **128** wave-2, int8 ~8-9%, TurboQuant 3.1% (wave-2 smoke), SnapKV 90-97%, KIVI r32 100% |

### Compressor tiers

| tier | examples | note |
|------|----------|------|
| Built-in real | `noop`, `int8` | Headline 1,500-cell panel |
| Built-in simulated | `int4_sim`, `int6_sim`, `h2o_sim` | Diagnostic, not upstream ports |
| Fallback / proxy | `spectralquant`, `shard` | Mock or probe-only rows in leaderboard |
| Faithful adapter | `snapkv_experimental`, `turboquant_experimental`, `kivi_offline_r32` | Appendix smoke grid only |

External panel drift rates are **not** official LongBench/BFCL/MBPP scores. Compression ratios cited in the report are **stored tensor byte ratios**, not active GPU memory savings at serving time.

---

## How it works

| step | what happens |
|------|----------------|
| **Draft** | Generate candidate tokens from compressed (lossy) KV |
| **Verify** | Compare each draft token to full-KV greedy argmax |
| **Commit** | Accept matching prefix; on mismatch, correct and advance full KV |
| **Measure** | Record first divergence, acceptance, agreement, failures |

ExactKV measures **whether compression is compatible with exact decoding**, not whether it is faster. Phase F kernel numbers are a **microbenchmark only**, not end-to-end inference speedup.

---

## Install

```bash
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV
pip install -e ".[dev]"
```

Requires **Python 3.10+**, **PyTorch**, and **transformers**. GPU optional for the terminal replay demo; required for benchmark panels.

---

## Usage

**Terminal crash-test demo** (replay, no GPU weights):

```bash
python3 scripts/exactkv_terminal_crash_test.py --speed fast
```

**Verify artifacts + claim safety** (no inference):

```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/check_site_claims.py
python3 scripts/validate_final_release_package.py
```

**Core leaderboard** (from committed scale panel):

```bash
python3 scripts/exactkv_leaderboard.py
```

**GPU headline panel** (Llama + Mistral, ~1,500 cells):

```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda --dtype float16
```

**External panels** (LongBench, BFCL, MBPP, etc.):

```bash
python3 scripts/run_external_panel.py --help
bash scripts/setup_faithful_compressor_env.sh   # KIVI / kvpress adapters
```

---

## Reproducing the benchmarks

Every headline number traces to an on-disk JSON artifact. Claim boundaries: [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md).

| panel | artifact |
|-------|----------|
| Core leaderboard (1,500 cells) | `reports/scale_7b/raw.json` |
| Public leaderboard JSON | `reports/public_release/leaderboard_final.json` |
| External panels (8,132 headline total) | `reports/external_panels/` |
| Faithful adapter smoke (864, appendix) | `reports/external_panels/faithful/` |
| v3.0 int6 + int4_per_vec | `reports/external_panels/v30/` |

Regenerate summaries:

```bash
python3 scripts/build_external_panel_summary.py --write
python3 scripts/build_launch_pack.py
bash scripts/sync_site_data.sh   # refresh site/data/ from reports
```

---

## Layout

```
exactkv/
  runtime/           # ExactKVGenerator — draft / verify / commit loop
  compressors/       # Built-in + adapter registry (int8, int4_sim, h2o_sim, …)
  benchmarks/        # Scale panel + external panel runners
  platform/          # Leaderboard aggregation + public release packaging

scripts/
  exactkv_repro.py              # One-command artifact + claim validation
  run_phase_a_scale_benchmark.py
  run_external_panel.py
  exactkv_terminal_crash_test.py

paper/               # Technical report (Markdown, LaTeX, PDF)
site/                # Public landing page + data/
reports/
  scale_7b/          # 1,500-cell headline panel
  external_panels/   # 8,132-cell external grid + faithful appendix
  public_release/    # Leaderboard JSON, demo cards

docs/                # Claim boundaries, metrics, experiment corpus
```

---

## Key links

| | |
|---|---|
| Technical report | [`paper/ExactKV_Technical_Report.md`](paper/ExactKV_Technical_Report.md) |
| Landing page | [`site/index.html`](site/index.html) |
| Leaderboard JSON | [`reports/public_release/leaderboard_final.json`](reports/public_release/leaderboard_final.json) |
| Release notes | [`RELEASE.md`](RELEASE.md) |
| Claim boundaries | [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md) |
| Evaluator guide (start here) | [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) |
| Metrics | [`docs/METRIC_DEFINITIONS.md`](docs/METRIC_DEFINITIONS.md) |
| Novelty audit | [`docs/NOVELTY_AUDIT.md`](docs/NOVELTY_AUDIT.md) |

---

## What ExactKV is not

- **Not** a production serving system, no throughput or active VRAM savings claims
- **Does not** reproduce VeriCache, inspired by draft/verify semantics only
- **Not** official benchmark scores, external panels are drift smoke tests
- **SpectralQuant** runs in fallback/proxy mode when the real dependency is unavailable
- **Shard** is probe-first, not a full Shard / ShardCache integration
- **`exactkv_failures = 0`** is a harness safety gate on tested panels, not "compression is always safe"

---

## Historical development

ExactKV grew through a long verifier-first research arc (Experiments 001–113+, V1–V21) before Phases A–J formalized the public benchmark platform. For version lineage, experiment index, and historical artifacts, see [`docs/PROJECT_LINEAGE.md`](docs/PROJECT_LINEAGE.md), [`docs/VERSION_LINEAGE.md`](docs/VERSION_LINEAGE.md), and [`docs/HISTORICAL_ARTIFACT_INVENTORY.md`](docs/HISTORICAL_ARTIFACT_INVENTORY.md).

---

## License

MIT License. See [`LICENSE`](LICENSE).
