# ExactKV

**Lossy KV-cache compression. Exact full-KV outputs.**

ExactKV is an open-source inference runtime and benchmark suite that lets lossy KV-cache
compressors draft tokens quickly, then verifies those tokens against full-KV decoding so
the final output remains identical to normal full-KV inference under deterministic
(greedy) decoding.

Inspired by the [VeriCache paper](https://arxiv.org/abs/2605.17613).
ExactKV is not a reimplementation — it is a compressor-agnostic platform for verified
KV-cache generation and benchmark evaluation.

> **V1 is correctness-first.** It does not claim throughput gains or speedups over
> standard full-KV inference. The goal is to prove that the output is lossless, not that
> it is faster.

---

## Headline result (v0.4.0)

**Experiment 003 — asymmetric K/V sweep.** Across a 612-run core-suite sweep
(`Qwen/Qwen2.5-0.5B`, 34 prompts × 9 compressors × 2 draft lengths), ExactKV had
**0 failures** while revealing that **keys are far more fragile than values under
compression**. Keeping keys at full precision with INT8 values (`k_full_v8`)
accepted 98.8% of drafted tokens; compressing keys to simulated 4-bit
(`k4_v8_sim`) collapsed acceptance to 56.2%.

| Compressor       | K bits | V bits | Accept rate | Rejected | ExactKV failures |
| ---------------- | -----: | -----: | ----------: | -------: | ---------------: |
| k_full_v8        |   full |      8 |       0.988 |       22 |                0 |
| k8_v_full        |      8 |   full |       0.953 |       86 |                0 |
| int8             |      8 |      8 |       0.953 |       89 |                0 |
| k_full_v4_sim ⚠️ |   full |  4-sim |       0.890 |      174 |                0 |
| k8_v4_sim ⚠️     |      8 |  4-sim |       0.858 |      240 |                0 |
| k4_v8_sim ⚠️     |  4-sim |      8 |       0.562 |     1253 |                0 |
| int4_sim ⚠️      |  4-sim |  4-sim |       0.553 |     1272 |                0 |
| k8_v2_sim ⚠️     |      8 |  2-sim |       0.330 |     2302 |                0 |

**Takeaway:** On this setup, compressing keys aggressively was far more damaging
to ExactKV acceptance than compressing values.

> ⚠️ `_sim` = simulated sub-INT8 numeric quantization stored in `int8` containers
> (no real bit-packing). `k8_v_full` and `k_full_v8` carry no `_sim` suffix
> because they use only full precision and INT8. Average effective bit width is a
> comparison aid, not a real memory metric. ExactKV reports exactness,
> acceptance, divergence, rejection, and correction behaviour — **not**
> performance.

Full report: [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md)
· Project status: [`docs/PROJECT_STATUS_V0.4.0.md`](docs/PROJECT_STATUS_V0.4.0.md)

---

## Status

**V4 — asymmetric K/V compression experiments (v0.4.0).**  V1 proved correctness; V2 added the compressor registry, CLI, JSON/CSV reporting, sweep orchestration, and analysis layer; V3 added prompt suites, Markdown report generation, acceptance leaderboards, and the `report` CLI; V4 adds asymmetric K/V compressors, K/V metadata in reports and leaderboards, and Experiment 003.

**V1 gates (correctness prototype)**

| Gate | Status |
|---|---|
| `generate_full_greedy` matches `model.generate` | ✅ |
| NoOp ExactKV output == full greedy | ✅ |
| INT8 ExactKV output == full greedy | ✅ |
| DebugNoise forces rejection + ExactKV corrects | ✅ |
| Metrics reconcile (`drafted == accepted + rejected`) | ✅ |
| Benchmark runner emits valid JSON; `exactkv_failures == 0` | ✅ |

**V2 gates (experimental framework)**

| Gate | Status |
|---|---|
| INT4-sim ExactKV output == full greedy (≥2 prompts × ≥2 draft lengths) | ✅ |
| Registry resolves every compressor by name; all run end-to-end | ✅ |
| Sweep (`noop × int8 × int4_sim` × multiple draft lengths) with `exactkv_failures == 0` | ✅ |
| JSON round-trip lossless; CSV schema stable, one row per cell | ✅ |
| Acceptance counts reconcile in analysis; mismatch and failure reports correct | ✅ |
| CLI `bench --suite smoke` runs with locally cached weights, writes valid reports | ✅ |
| No tokens/sec, latency, or speedup language in any V2 output | ✅ |

**V3 gates (presentation and storytelling layer)**

| Gate | Status |
|---|---|
| Four named prompt suites (`core`, `structured`, `code`, `stress`) load and validate | ✅ |
| Histogram and example analysis functions reconcile counts on real sweep reports | ✅ |
| Markdown generator produces complete report from sweep JSON (leaderboard, examples, histograms) | ✅ |
| Every rendered artifact preserves `int4_sim` simulation labelling | ✅ |
| `python -m exactkv report` writes Markdown from existing JSON | ✅ |
| `docs/EXPERIMENT_002_CORE_SWEEP.md` written from real core-suite sweep with `exactkv_failures == 0` | ✅ |
| No-performance-field audit passes across all V3 code, reports, and docs | ✅ |

**V4 gates (asymmetric K/V compression experiments)**

| Gate | Status |
|---|---|
| `CompressorCapabilities` carries `key_bit_width`, `value_bit_width`, `asymmetric`; backfilled for V1–V3 | ✅ |
| `AsymmetricQuantSimCompressor(k_bits, v_bits)` with independent K/V quantisation and `full` passthrough | ✅ |
| All 7 named asymmetric compressors resolve via registry with `exactkv_failures == 0` | ✅ |
| JSON/CSV reports include `key_bit_width`, `value_bit_width`, `asymmetric`; backward compatible | ✅ |
| Leaderboard renders K bits, V bits, avg eff bits; `list-compressors` shows K/V metadata | ✅ |
| `docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md` written from 612-run sweep with `exactkv_failures == 0` | ✅ |
| No-performance-field audit passes across all V4 code, reports, and docs | ✅ |

---

## Install

```bash
# Clone
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV

# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Download model weights (first run only — ~1 GB)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')"
```

---

## Run tests

```bash
# All tests (requires cached model weights)
TRANSFORMERS_OFFLINE=1 pytest tests/ -v

# Acceptance logic only (no model needed)
pytest tests/test_acceptance_logic.py -v

# Specific gate
TRANSFORMERS_OFFLINE=1 pytest tests/test_int8_exactkv.py -v
```

Expected: **all 542 tests pass** in ~240–280 s on CPU with `Qwen/Qwen2.5-0.5B` in `float32`.

---

## Run the example script

```bash
# INT8 compressor (default)
TRANSFORMERS_OFFLINE=1 python examples/qwen_smoke.py

# NoOp compressor (should always match full, acceptance = 100%)
TRANSFORMERS_OFFLINE=1 python examples/qwen_smoke.py --compressor noop

# DebugNoise compressor (forces rejection; ExactKV still corrects)
TRANSFORMERS_OFFLINE=1 python examples/qwen_smoke.py --compressor debug_noise

# Custom prompt and length
TRANSFORMERS_OFFLINE=1 python examples/qwen_smoke.py \
    --prompt "Write a Python function that sorts a list:" \
    --max-new-tokens 40 \
    --draft-len 4
```

Example output (INT8, greedy, `Qwen/Qwen2.5-0.5B`):

```
  ExactKV matches full    : True
  Lossy  matches full     : True/False   ← no guarantee
  Acceptance rate         : 0.923
  Avg accepted / round    : 3.69
  Correction count        : 2
  Rejection count         : 5
  Compression ratio       : 0.258  (compressed/full; < 1 means smaller)
  Memory reduction factor : 3.87x  (full/compressed; > 1 means savings)
```

---

## Run the benchmark suite

```bash
# Run 2-prompt subset with INT8
TRANSFORMERS_OFFLINE=1 python - <<'EOF'
import json
from exactkv.benchmarks.prompts import load_smoke_prompts
from exactkv.benchmarks.runner import RunConfig, run_suite
from exactkv.runtime.model_runtime import ModelRuntime

rt = ModelRuntime("Qwen/Qwen2.5-0.5B", device="auto", dtype="float32")
cfg = RunConfig(compressor_name="int8", draft_len=4, max_new_tokens=32)
report = run_suite(rt, load_smoke_prompts()[:4], cfg)
print(json.dumps(report["aggregate"], indent=2))
EOF
```

---

## V2 reporting (JSON and CSV)

ExactKV V2 can write stable JSON and CSV reports that include compressor
metadata alongside correctness and acceptance metrics.

### Write a JSON report

```python
from exactkv.benchmarks.prompts import load_smoke_prompts
from exactkv.benchmarks.reports import build_run_manifest, write_json_report
from exactkv.benchmarks.runner import RunConfig, run_suite
from exactkv.runtime.model_runtime import ModelRuntime

rt  = ModelRuntime("Qwen/Qwen2.5-0.5B", device="auto", dtype="float32")
cfg = RunConfig(compressor_name="int8", draft_len=4, max_new_tokens=32)

report   = run_suite(rt, load_smoke_prompts()[:4], cfg)
manifest = build_run_manifest(
    model_name="Qwen/Qwen2.5-0.5B",
    prompt_suite="smoke",
    compressor_names=["int8"],
    draft_len=4,
    max_new_tokens=32,
)
write_json_report(report, "reports/run_int8.json", manifest=manifest)
```

### Write a CSV report

```python
from exactkv.benchmarks.reports import write_csv_report

write_csv_report(report, "reports/run_int8.csv")
```

### What V2 reports — and what it does not

**Reported:** exactness (token match), acceptance rate, accepted/rejected/corrected
counts, memory byte estimates, compressor metadata.

**Not reported:** tokens/second, throughput, latency, speedup, or any runtime
performance metric. V2 proves *correctness and acceptance behaviour*, not
production performance.

### int4\_sim memory disclaimer

`int4_sim` is a **simulated** INT4 compressor. It quantises values into the
signed-4-bit range `[-8, 7]` but stores them in `torch.int8` containers (1 byte
per element). It does **not** perform real 4-bit bit-packing.

Every JSON and CSV report row for `int4_sim` includes:

```
"is_simulated": true,
"supports_real_bytes_claim": false,
"memory_claim_note": "int4_sim uses int8 container storage in V2; do not
  interpret this as real packed INT4 memory savings."
```

Do not cite `int4_sim`'s `compressed_kv_bytes` as evidence of real INT4 memory
savings.

---

### V4 asymmetric compressors (experimental)

V4 adds seven asymmetric compressors that quantise K and V at **different**
bit-widths. All are simulated or use only real INT8 and full precision. None
perform real bit-packing below 8 bits.

| Name | K width | V width | `is_simulated` | Notes |
|---|---|---|---|---|
| `k8_v4_sim` ⚠️ | INT8 | INT4-sim | yes | V simulated in int8 container |
| `k8_v2_sim` ⚠️ | INT8 | INT2-sim | yes | V simulated in int8 container |
| `k4_v8_sim` ⚠️ | INT4-sim | INT8 | yes | K simulated in int8 container |
| `k_full_v4_sim` ⚠️ | full | INT4-sim | yes | V simulated in int8 container |
| `k4_v_full_sim` ⚠️ | INT4-sim | full | yes | K simulated in int8 container |
| `k8_v_full` | INT8 | full | **no** | Real INT8 K, full V |
| `k_full_v8` | full | INT8 | **no** | Full K, real INT8 V |

**Naming rule:** `_sim` suffix is present only when a compressor includes a
simulated sub-INT8 width (4-bit or 2-bit). `k8_v_full` and `k_full_v8` use
only real INT8 and full precision — no simulated storage — so they carry no
`_sim` suffix and report `is_simulated=False`.

These compressors are for **acceptance-rate experiments only**. Do not cite
their `compressed_kv_bytes` for simulated sides as real memory savings.

The **asymmetric leaderboard** (rendered via `exactkv report` and the
`render_compressor_leaderboard` API) compares acceptance behaviour across symmetric
and asymmetric configurations. The **K bits**, **V bits**, and **avg eff bits** columns
are metadata from compressor capabilities — they are not real memory measurements.
Average effective bits = (K bits + V bits) / 2, counting full precision as 32 bits.

---

## Project structure

```
exactkv/
├── __main__.py               # python -m exactkv entry point
├── cli.py                    # CLI subcommands: bench, sweep, analyze, list-compressors
├── config.py                 # ExactKVConfig, BenchmarkConfig
├── runtime/
│   ├── model_runtime.py      # ModelRuntime (HF model + tokenizer wrapper)
│   ├── prefill.py            # prefill_to_full_state (shared helper)
│   ├── generation.py         # generate_full_greedy, generate_lossy_greedy
│   └── exactkv_generator.py  # ExactKVGenerator (draft-verify-commit loop)
├── cache/
│   ├── full_state.py         # FullKVState (authoritative)
│   ├── compressed_state.py   # CompressedKVState (compressor-specific)
│   └── utils.py              # kv_seq_len, extract_kv_tensors, rebuild_cache
├── compressors/
│   ├── __init__.py           # registers all built-in compressors (V1–V3 + V4)
│   ├── base.py               # KVCompressor Protocol, CompressionStats, CompressorCapabilities
│   ├── registry.py           # register_compressor, get_compressor, list_compressors
│   ├── noop.py               # NoOpCompressor (identity, acceptance=100%)
│   ├── int8.py               # Int8Compressor (per-tensor symmetric INT8)
│   ├── int4_sim.py           # Int4SimCompressor (simulated INT4, int8 storage)
│   ├── debug_noise.py        # DebugNoiseCompressor (forces rejection, test only)
│   └── asymmetric_sim.py     # V4: AsymmetricQuantSimCompressor + 7 named subclasses
├── verification/
│   ├── acceptance.py         # compute_acceptance, AcceptanceResult, traces
│   └── engine.py             # VerificationEngine.verify_sequential
├── metrics/
│   ├── exactness.py          # token_exact_match, first_divergence_idx
│   ├── acceptance.py         # summarize_acceptance, AcceptanceSummary
│   └── memory.py             # estimate_kv_memory, MemorySummary
├── benchmarks/
│   ├── prompts.py            # load_prompts, load_smoke_prompts
│   ├── runner.py             # run_one, run_suite, RunConfig
│   ├── reports.py            # write_json_report, write_csv_report, build_run_manifest
│   └── sweeps.py             # run_sweep (multi-compressor × multi-draft-length)
└── analysis/
    ├── __init__.py           # public API re-exports
    ├── acceptance_tables.py  # build_acceptance_table, group_by_*, write_acceptance_table_csv
    ├── mismatch.py           # first_lossy_divergences, mismatch_position_summary
    └── failure_report.py     # build_failure_report, write_failure_report_json

benchmarks/
└── prompts/
    ├── smoke.jsonl           # 16 prompts — fast CI suite
    ├── core.jsonl            # 34 prompts — broad coverage
    ├── structured.jsonl      # 28 prompts — JSON/tables/schemas
    ├── code.jsonl            # 30 prompts — code generation/completion
    └── stress.jsonl          # 25 prompts — harder, lower-acceptance

examples/
└── qwen_smoke.py             # Demo script with side-by-side comparison

docs/
├── V1_SCOPE_STATEMENT.md
├── V2_SCOPE_STATEMENT.md
├── RELEASE_NOTES_V0.2.0.md
└── IMPLEMENTATION_PLAN.md

tests/
├── test_acceptance_logic.py        # acceptance logic unit tests (model-free)
├── test_full_generation.py         # full baseline gate
├── test_verification_engine.py     # verification engine
├── test_noop_exactkv.py            # NoOp ExactKV gate
├── test_int8_compressor.py         # INT8 compressor unit tests
├── test_int8_exactkv.py            # INT8 ExactKV gate
├── test_debug_noise_exactkv.py     # DebugNoise rejection gate
├── test_lossy_generation.py        # lossy generation
├── test_metrics.py                 # metrics unit tests
├── test_benchmark_runner.py        # benchmark runner gate
├── test_example_script.py          # example script smoke test
├── test_prefill_helper.py          # shared prefill helper
├── test_compressor_registry.py     # compressor registry gate
├── test_config_unified.py          # unified config
├── test_int4_sim_compressor.py     # INT4-sim unit tests
├── test_int4_sim_exactkv.py        # INT4-sim ExactKV gate
├── test_reports.py                 # reporting gate (JSON/CSV)
├── test_sweeps.py                  # sweep gate
├── test_analysis_acceptance_tables.py  # analysis: acceptance tables gate
├── test_analysis_mismatch.py           # analysis: mismatch gate
├── test_analysis_failure_report.py     # analysis: failure report gate
├── test_cli.py                     # CLI gate
└── test_prompt_suites.py           # V3: named-suite validation + CLI gate
```

---

## Sample v0.2.0 smoke sweep

See [`docs/EXPERIMENT_001_SMOKE_SWEEP.md`](docs/EXPERIMENT_001_SMOKE_SWEEP.md)
for a full write-up. Headline results across 6 prompts × 3 compressors × 2 draft
lengths (36 runs total):

| Compressor | Accept rate | Lossy divergences | ExactKV failures |
|---|---|---|---|
| `noop` | 1.000 | 0 / 12 | 0 / 12 |
| `int8` | **0.931** | 4 / 12 | **0 / 12** |
| `int4_sim` ⚠️ | 0.459 | 12 / 12 | **0 / 12** |

> ⚠️ `int4_sim` is **simulated** (quantised range `[-8, 7]`, stored in
> `torch.int8`). Its 3.95× memory reduction factor does **not** reflect real
> packed-4-bit savings. ExactKV corrected all 16 lossy divergences with zero
> failures.

No throughput, latency, or speedup is claimed.

---

## Experiment 002: Core suite sweep (v0.3.0)

See [`docs/EXPERIMENT_002_CORE_SWEEP.md`](docs/EXPERIMENT_002_CORE_SWEEP.md)
for the full Markdown report (generated via `python -m exactkv report`). 34
prompts × 3 compressors × 2 draft lengths = **204 runs total**:

| Compressor | Runs | Accept rate | Drafted | Accepted | Rejected | ExactKV failures |
|---|---|---|---|---|---|---|
| `noop` | 68 | **1.000** | 1428 | 1428 | 0 | **0** |
| `int8` | 68 | **0.951** | 1492 | 1400 | 92 | **0** |
| `int4_sim` ⚠️ | 68 | 0.553 | 2369 | 1097 | 1272 | **0** |

| Draft length | Accept rate | Drafted | Accepted |
|---|---|---|---|
| 4 | 0.865 | 2404 | 1960 |
| 8 | 0.805 | 2885 | 1965 |

> ⚠️ `int4_sim` is **simulated** (INT4 numeric range, stored in `torch.int8`).
> Memory figures reflect `int8` storage, not real packed 4-bit savings.
> ExactKV produced **0 failures** — every verified output matched
> `generate_full_greedy` exactly.

No throughput, latency, or speedup is claimed.

---

## Experiment 003: Asymmetric K/V sweep (v0.4.0)

See [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md)
for the full Markdown report. 34 prompts × 9 compressors × 2 draft lengths = **612 runs total**.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompt suite | `core.jsonl` (34 prompts) |
| Compressors | 9 (see table below) |
| Draft lengths | 4, 8 |
| Max new tokens | 24 |
| Total runs | **612** |
| **ExactKV failures** | **0** |
| Lossy divergences | 386 (expected) |
| Mean accept rate | 0.739 |

**Acceptance by compressor:**

| Compressor | K bits | V bits | Simulated | Accept rate | Rejected | ExactKV fail |
|---|---|---|---|---|---|---|
| `k_full_v8` | full | 8 | no | **0.988** | 22 | **0** |
| `k8_v_full` | 8 | full | no | **0.953** | 86 | **0** |
| `int8` | 8 | 8 | no | **0.953** | 89 | **0** |
| `k_full_v4_sim` ⚠️ | full | 4 | yes | 0.890 | 174 | **0** |
| `k8_v4_sim` ⚠️ | 8 | 4 | yes | 0.858 | 240 | **0** |
| `k4_v8_sim` ⚠️ | 4 | 8 | yes | 0.562 | 1253 | **0** |
| `k4_v_full_sim` ⚠️ | 4 | full | yes | 0.561 | 1255 | **0** |
| `int4_sim` ⚠️ | 4 | 4 | yes | 0.553 | 1272 | **0** |
| `k8_v2_sim` ⚠️ | 8 | 2 | yes | 0.330 | 2302 | **0** |

> ⚠️ Simulated compressors store sub-INT8 values in `torch.int8` containers.
> Memory figures reflect `int8` storage, not real packed savings.
> **All 612 runs: ExactKV failures = 0.** Every verified output matched
> `generate_full_greedy` exactly.
>
> `k_full_v8` (full-precision K, INT8 V) is the highest-acceptance asymmetric
> compressor tested. `k8_v2_sim` (INT8 K, INT2-sim V) has the lowest acceptance
> at 0.330 — aggressive V compression severely damages acceptance.
>
> Average effective bit width is a metadata comparison aid, not a real memory
> measurement. No throughput, latency, or speedup is claimed.

---



## V2 sweeps

Use `run_sweep` to compare multiple compressors and draft lengths across a
prompt suite in a single call. The model is loaded once and reused across all
cells.

```python
from exactkv.benchmarks.prompts import load_smoke_prompts
from exactkv.benchmarks.reports import write_csv_report, write_json_report
from exactkv.benchmarks.sweeps import run_sweep
from exactkv.runtime.model_runtime import ModelRuntime

rt = ModelRuntime("Qwen/Qwen2.5-0.5B", device="auto", dtype="float32")

sweep = run_sweep(
    runtime=rt,
    prompts=load_smoke_prompts()[:4],
    compressor_names=["noop", "int8", "int4_sim"],
    draft_lengths=[4, 8],
    max_new_tokens=32,
    prompt_suite="smoke",
)

# Results: 4 prompts × 3 compressors × 2 draft_lengths = 24 rows
write_json_report(sweep, "reports/sweep.json")
write_csv_report(sweep, "reports/sweep.csv")
print(f"ExactKV failures: {sweep['aggregate']['exactkv_failures']}")
print(f"Mean acceptance rate: {sweep['aggregate']['mean_acceptance_rate']:.3f}")
```

Sweeps report **acceptance rate**, **exactness**, and **memory byte estimates**
across the full compressor × draft-length grid. They do **not** report
tokens/second, latency, throughput, or speedup.

`int4_sim` rows carry `is_simulated=True` and `supports_real_bytes_claim=False`
in every output format; the `memory_claim_note` field explains the int8 storage
limitation.

---

## V2 analysis

The `exactkv.analysis` package analyses existing benchmark and sweep reports
without re-running the model.

### Acceptance tables

Summarise acceptance rate, accepted/rejected/corrected counts grouped by
compressor, draft length, or prompt category:

```python
from exactkv.analysis import (
    build_acceptance_table,
    group_acceptance_by_compressor,
    group_acceptance_by_draft_len,
    group_acceptance_by_category,
    write_acceptance_table_csv,
)

table   = build_acceptance_table(sweep_report)          # per (compressor × draft_len)
by_comp = group_acceptance_by_compressor(sweep_report)  # draft_len collapsed
by_dl   = group_acceptance_by_draft_len(sweep_report)   # compressor collapsed
by_cat  = group_acceptance_by_category(sweep_report)    # grouped by prompt type

write_acceptance_table_csv(table, "reports/acceptance.csv")
```

### Mismatch analysis

Identify where lossy divergences and ExactKV rejections occur:

```python
from exactkv.analysis import (
    first_lossy_divergences,
    mismatch_position_summary,
    rejection_position_summary,
)

divergences = first_lossy_divergences(sweep_report)    # first_divergence_idx per result
summary     = mismatch_position_summary(sweep_report)  # aggregate stats
rejections  = rejection_position_summary(sweep_report) # per-result rejection counts

print(f"Lossy diverged: {summary['lossy_divergence_count']} / {summary['total_runs']}")
print(f"Mean first divergence at token: {summary['mean_first_divergence_idx']}")
```

### Failure reports

Classify ExactKV failures vs. expected lossy divergences:

```python
from exactkv.analysis import build_failure_report, write_failure_report_json

fr = build_failure_report(sweep_report)
print(f"Status: {fr['status']}")                         # "pass" or "fail"
print(f"ExactKV failures: {fr['exactkv_failure_count']}")    # must be 0
print(f"Lossy divergences: {fr['lossy_divergence_count']}")  # expected for real compressors

write_failure_report_json(fr, "reports/failures.json")
```

**Key distinction:**
- **Lossy divergence** (`lossy.token_exact_match == False`) is *expected*. It proves that the compressor changes the output and demonstrates why verification is necessary.
- **ExactKV failure** (`exactkv_failure == True`) means the verified output did *not* match `generate_full_greedy`. This is a correctness bug and must always be zero.

No timing, throughput, latency, or speedup metrics are produced by any analysis function.

---

## V3 prompt suites

ExactKV ships five named prompt suites. All report **acceptance and exactness
behaviour** — none of them make throughput, latency, or speedup claims.

| Suite | Prompts | Purpose |
|---|---|---|
| `smoke` | 16 | Fast CI and quick sanity checks. **Default for tests.** |
| `core` | 34 | Broad category coverage. Default for documented experiments. |
| `structured` | 28 | JSON, tables, schemas, function-calls, YAML. Tests acceptance on highly-templated output. |
| `code` | 30 | Code generation, completion, debugging, SQL, bash. Tests acceptance on syntax-sensitive continuations. |
| `stress` | 25 | Longer, harder, higher-entropy prompts. Designed to surface lower acceptance rates and more lossy divergence. |

Use a named suite from Python:

```python
from exactkv.benchmarks.prompts import load_suite, list_suites

print(list_suites())            # ['code', 'core', 'smoke', 'stress', 'structured']
prompts = load_suite("core")    # 34 prompts
```

Or from the CLI:

```bash
python -m exactkv bench --suite core  --compressor int8 --max-new-tokens 32 ...
python -m exactkv sweep --suite stress --compressors noop,int8 --draft-lengths 4,8 ...
```

Use `--suite-file path/to/custom.jsonl` to supply your own prompts; this overrides
`--suite`.

> All suites produce reports that measure acceptance rate, exactness, and memory
> byte estimates. They do **not** measure tokens/second, latency, throughput, or
> speedup.

---

## V3 analysis compute (histograms and examples)

The `exactkv.analysis` package now includes compute utilities that summarise
acceptance and divergence behaviour in text/table form — **no images, no model
re-run, no performance metrics**.

### Accepted-length histogram

How many tokens are accepted per verification round, on average:

```python
from exactkv.analysis import accepted_length_histogram

h = accepted_length_histogram(sweep_report)
# h["buckets"] → {"0": 0, "1": 2, "2-3": 5, "4-7": 8, "8-15": 3, "16+": 0}
# h["total"]   → 18

# Group by compressor
by_comp = accepted_length_histogram(sweep_report, group_by="compressor_name")
# by_comp["int8"]["buckets"] → {...}
```

### First-divergence bucket table

Where in the generated sequence does lossy divergence first appear:

```python
from exactkv.analysis import first_divergence_histogram

h = first_divergence_histogram(sweep_report)
# h["buckets"] → {"no_divergence": 12, "0": 0, "1-4": 3, "5-16": 1, ...}
```

### Lossy divergence examples

Side-by-side text comparison of full / lossy / ExactKV output for results where
the unverified lossy output diverged:

```python
from exactkv.analysis import extract_lossy_divergence_examples

examples = extract_lossy_divergence_examples(sweep_report, limit=3)
for ex in examples:
    print(f"Prompt  : {ex['prompt']}")
    print(f"Full    : {ex['full_text']}")
    print(f"Lossy   : {ex['lossy_text']}")   # diverges at token ex['first_divergence_idx']
    print(f"ExactKV : {ex['exactkv_text']}")  # always matches Full
    print(f"Note    : {ex['explanation']}")
```

Every example includes an `explanation` field that states:
*"Lossy divergence is expected … ExactKV corrects this … A non-zero
`exactkv_matches_full=False` would be a correctness bug, not a lossy
divergence."*

> These functions produce text/table artifacts for acceptance and exactness
> analysis. They do **not** produce tokens/second, latency, throughput, or
> speedup metrics.

---

## V3 Markdown reports

Existing JSON/CSV reports (from `bench`, `sweep`, or `run_suite`) can be
rendered into docs-ready Markdown using the `exactkv.reporting` package.
Reports document **exactness, acceptance behaviour, and divergence examples**
— no performance claims, no timing, no speedup.

```python
from exactkv.reporting import render_markdown_report, write_markdown_report

# Render to string
md = render_markdown_report(sweep_report, title="My Sweep Report")

# Write to file (creates parent dirs automatically)
write_markdown_report(sweep_report, "reports/my_sweep.md")
```

The rendered Markdown includes:

* **Correctness summary** — ExactKV failure count (must be 0) and lossy
  divergence count (expected for lossy compressors).
* **Acceptance leaderboard** — by compressor, by draft length, and
  compressor × draft-length grid for sweep reports.
* **Histogram tables** — accepted-length distribution, first-divergence
  position, and rejection-count distribution (text tables, no images).
* **Lossy divergence examples** — side-by-side Full / Lossy / ExactKV
  output excerpts with an explanation that lossy divergence is expected.
* **ExactKV failure examples** — should always be empty; included to surface
  bugs immediately.
* **Required disclaimers** — lossy divergence vs. ExactKV failure, `int4_sim`
  simulation honesty, and explicit "no speedup/throughput/latency claim".

> These reports are for exactness, acceptance, and divergence storytelling.
> They do **not** claim speedup, throughput, latency, or production readiness.
> `int4_sim` memory figures reflect `int8` container storage, not real packed
> INT4 savings.

---

## V2 CLI

`python -m exactkv` exposes the most-used ExactKV operations as CLI subcommands.
The CLI reports **exactness and acceptance behaviour only** — it does **not**
report tokens/second, latency, throughput, or speedup.

### list-compressors

Print all registered compressors with their capabilities, including K/V bit-width
metadata added in V4:

```bash
python -m exactkv list-compressors
```

Each compressor entry now shows `key_bit_width`, `value_bit_width`, and `asymmetric`.
Full-precision sides display as `full`. This metadata is a comparison aid — it is not
a real memory measurement.

### bench

Run a single-compressor benchmark over a named (or custom) prompt suite:

```bash
python -m exactkv bench \
  --model Qwen/Qwen2.5-0.5B \
  --suite smoke \
  --compressor int8 \
  --draft-len 8 \
  --max-new-tokens 32 \
  --json-out reports/bench_int8.json \
  --csv-out  reports/bench_int8.csv
```

**Defaults:** `--compressor int8`, `--draft-len 8`, `--max-new-tokens 16`.

Use `--suite-file path/to/custom.jsonl` to supply your own prompt suite.

### sweep

Run a multi-compressor × multi-draft-length grid sweep:

```bash
python -m exactkv sweep \
  --model Qwen/Qwen2.5-0.5B \
  --suite smoke \
  --compressors noop,int8,int4_sim \
  --draft-lengths 2,4,8 \
  --max-new-tokens 32 \
  --json-out reports/sweep.json \
  --csv-out  reports/sweep.csv
```

The summary printed to stdout includes `total_runs`, `exactkv_failures`,
`lossy_divergence_count`, and `mean_acceptance_rate`. No timing fields.

### analyze

Analyse an existing JSON report without re-running the model:

```bash
python -m exactkv analyze \
  --report reports/sweep.json \
  --acceptance-csv reports/acceptance.csv \
  --failure-json  reports/failures.json
```

Returns exit code 0 when `exactkv_failure_count == 0` ("pass"), 1 otherwise.

### report (V3)

Render an existing JSON report to a docs-ready Markdown document — no model
re-run, no timing output:

```bash
python -m exactkv report \
  --report reports/experiment_002_core_sweep.json \
  --markdown-out docs/EXPERIMENT_002_CORE_SWEEP.md \
  --title "Experiment 002: Core Suite Sweep" \
  --max-examples 5
```

Options: `--no-examples` to skip example blocks, `--max-examples INT` to
control how many per section (default 3).

### int4_sim disclaimer

`int4_sim` is a **simulated** INT4 compressor. All CLI outputs set
`is_simulated=True` and `supports_real_bytes_claim=False` for `int4_sim` rows.
Do **not** interpret `int4_sim` memory numbers as real packed-4-bit savings.

---

## V1 design principles

- **Correctness first.** Every output token ID must match `generate_full_greedy`.
- **Compressor-agnostic.** Any class satisfying the `KVCompressor` protocol can be plugged in.
- **No speedup claims.** ExactKV proves losslessness and measures acceptance, not throughput.
- **Greedy only.** Sampling, beam search, and speculative bonus tokens are deferred to a future version.
- **Sequential verification.** Parallel (single-pass) verification is deferred to a future version.
- **DynamicCache note.** The cache utilities target transformers 5.8.1 internal structure;
  see `docs/V1_SCOPE_STATEMENT.md` for brittleness notes.

---

## Future research directions

* **V4 (complete) — asymmetric K/V acceptance experiments.** V4 implemented
  simulated asymmetric compressors (`k8_v4_sim`, `k8_v2_sim`, `k4_v8_sim`,
  `k_full_v4_sim`, `k4_v_full_sim`, `k8_v_full`, `k_full_v8`), K/V metadata
  reporting, and Experiment 003 (612-run core-suite sweep; `exactkv_failures == 0`).
  No real backends, no performance claims. See
  [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md).
* **V5 — workspace-aware memory accounting (Phases A–C complete).** Distinguish
  `stored_kv_bytes`, `materialized_working_kv_bytes`, `metadata_bytes`, and
  `temporary_workspace_bytes` in JSON/CSV reports and Markdown renders. Stored
  bytes omit the dequantisation working set; `total_kv_footprint_bytes` is a
  conservative accounting sum, not a measured peak GPU value. Markdown reports
  now include a "Workspace-Aware Memory Accounting" section with a per-compressor
  table. No backend, no performance claims.
* **V6 — real backend adapter interface and first backend candidate.** Design a
  `BackendAdapter` so a real quantisation format (e.g. a KIVI- or
  TurboQuant-style quantizer) could plug into the `KVCompressor` protocol and be
  evaluated by acceptance behaviour. Implementation only behind separate approval.
* **V7 — attention-aware and V-specific backend ideas.** Sparse V dequantization,
  layer-aware V compression, and real asymmetric compressor comparisons —
  evaluated, not just reconstructed.
* **V8 — serving-stack integration.** Only after correctness/acceptance are well
  understood; without adopting any serving-stack performance claims as ExactKV's.

See [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md)
for the asymmetric-K/V writeup,
[`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](docs/RELATED_WORK_KV_CACHE_COMPRESSION.md)
for the full related-work survey, and
[`docs/RELEASE_NOTES_V0.4.0.md`](docs/RELEASE_NOTES_V0.4.0.md) for V4 release notes.

---

## Key documents

| Document | Purpose |
|---|---|
| [`docs/PROJECT_STATUS_V0.4.0.md`](docs/PROJECT_STATUS_V0.4.0.md) | Internal project status, version timeline, what ExactKV does/doesn't prove |
| [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md) | Full Experiment 003 asymmetric K/V sweep report |
| [`docs/RELEASE_NOTES_V0.4.0.md`](docs/RELEASE_NOTES_V0.4.0.md) | v0.4.0 release notes (V1–V4 history, results, limitations) |
| [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md) | Research note on asymmetric K/V and workspace-aware memory |
| [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](docs/RELATED_WORK_KV_CACHE_COMPRESSION.md) | Survey of KV-cache compression/quantization/eviction/serving + TurboQuant+ section |
| [`docs/RESEARCH_BACKLOG.md`](docs/RESEARCH_BACKLOG.md) | Concrete future-experiment backlog (real backends, eviction, serving) |
| [`docs/V5_SCOPE_DRAFT.md`](docs/V5_SCOPE_DRAFT.md) | Draft V5 plan (workspace memory + real backend planning) — not implemented |
| `docs/PRIVATE_FUTURE_POST_NOTES_EXPERIMENT_003.md` | 🔒 Private draft announcement notes for later — not for posting |

---

## Related work

ExactKV is a verification/evaluation framework, not a compression backend. It
does **not** implement KIVI, KVQuant, KV-AdaQuant, TurboQuant/TurboQuant+, KVTC,
Palu, SnapKV, H2O, StreamingLLM, PyramidKV, LMCache, vLLM, or PagedAttention, and
makes no speedup/throughput/latency claims. Its sub-INT8 `_sim` compressors store
values in `int8` containers and are not real packed-bit backends.

* [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](docs/RELATED_WORK_KV_CACHE_COMPRESSION.md)
  — full survey (quantization, asymmetric K/V, eviction, transform coding,
  serving) and a dedicated TurboQuant+ section.
* [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md)
  — asymmetric K/V deep dive and matched-budget caveat.
* VeriCache (below) — the draft-with-compressed-KV, verify-with-full-KV algorithm
  ExactKV is built on.

---

## Citation

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is a
compressor-agnostic, Hugging Face-first implementation, a structured benchmark harness,
and a framework for evaluating compressors by acceptance behaviour under full-KV
verification.
