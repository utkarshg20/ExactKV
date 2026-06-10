# ExactKV

**Compress the KV cache. Keep every output token.**

Lossy KV-cache compression makes drafting cheaper — but one bad draft token can
derail the whole sequence. ExactKV runs a **draft → verify → commit** loop: compressors
draft on a lossy cache, verification always uses **full-precision KV**, and the final
greedy output **matches uncompressed inference exactly** (`exactkv_output_ids ==
full_output_ids`).

Inspired by [VeriCache](https://arxiv.org/abs/2605.17613). ExactKV is not a
reimplementation — it is a **compressor-agnostic research platform** for evaluating
KV-cache compression by exactness, acceptance, divergence, and honest memory accounting.

---

## How it works

1. **Draft** — generate candidate tokens using a compressed (lossy) KV cache.
2. **Verify** — check each draft token against full-KV greedy predictions.
3. **Commit** — accept the matching prefix; on mismatch, correct and advance authoritative full KV.
4. **Repeat** — recompress from the updated full state; alignment invariant holds every round.

ExactKV measures **whether compression is compatible with exact decoding**, not whether it
is faster. It does **not** claim throughput, latency, speedup, or production readiness.

---

## At a glance

| Item | Value |
|---|---|
| **Latest release** | [`v0.9.0`](docs/RELEASE_NOTES_V0.9.0.md) — real-backend gauntlet (Experiments 008–011) |
| **Next** | [**V10**](docs/V10_SCOPE_DRAFT.md) — harden evaluation suite and divergence forensics before public launch |
| **Status** | Research milestone; [not public-launch final](docs/PROJECT_STATUS_V0.9.0.md) |
| **Hard gate** | `exactkv_failures == 0` on every published experiment |
| **Default model** | `Qwen/Qwen2.5-0.5B` (greedy, single-request, CPU-first) |
| **Compressors** | 15 built-in (`noop`, `int8`, asymmetric `_sim`, layer-aware boundary, `backend_passthrough`, …) |

---

## Latest results (v0.9.0)

**V9 real-backend gauntlet** — Experiments 008–011; `exactkv_failures == 0` on all cells.
Restricted adapters are **factory-only** (not default registry).

| Track | Headline (0.5B core, accept) |
|---|---:|
| `int8` / `k8_v4_boundary4_v8_sim` | ~0.97 / ~0.95 |
| KVQuant simquant (Exp 010) | **0.792** |
| TurboQuant Python (Exp 008) | **0.435** |
| KIVI offline (Exp 009) | **0.012** |
| Qwen2.5-1.5B (Exp 011) | exactness preserved; boundary4 **0.954** > k8_v4_sim **0.945** |

> ⚠️ Not production backends (no llama.cpp, KIVI CUDA, KVQuant deployment CUDA).
> `_sim` = int8 containers. No throughput, latency, or production-serving claims.
> See [v0.9.0 release notes](docs/RELEASE_NOTES_V0.9.0.md).

---

## Version timeline

| Version | Tag | Focus | Status |
|---|---|---|---|
| V1–V3 | — | Correctness prototype, registry, CLI, sweeps, Markdown reports | ✅ |
| V4 | `v0.4.0` | Asymmetric K/V compressors; Experiment 003 (612 runs) | ✅ |
| V5 | `v0.5.0` | Workspace-aware memory accounting; Experiment 004 | ✅ |
| V6 | `v0.6.0` | `BackendAdapter`; restricted kvpress KnormPress; Experiment 005 | ✅ |
| V7 | `v0.7.0` | Layer-aware V policies; Experiments 006 / 006C | ✅ |
| V8 | `v0.8.0` | Serving harness; Experiment 007 | ✅ |
| V9 | `v0.9.0` | Real backend gauntlet; Exp 008–011; 1.5B validation | ✅ |
| V10 | — | Evaluation suite hardening; divergence forensics | Next |
| V11–V12 | — | Scale, serving probes → v1.0.0 | Planned |

All published sweeps report **`exactkv_failures == 0`**. ExactKV reports exactness and
acceptance behaviour — **not** tokens/sec, throughput, or latency.

**Contents:** [Install](#install) · [Tests](#run-tests) · [Example](#run-the-example-script) · [Benchmarks](#run-the-benchmark-suite) · [CLI](#v2-cli) · [Compressors](#v4-asymmetric-compressors-experimental) · [Roadmap](#roadmap-and-research) · [Docs](#key-documents)

---

## Install

```bash
# Clone
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV

# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Download model weights (first run only — ~1 GB)
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
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

Expected: **all ~1,400 tests pass** in a few minutes on CPU with `Qwen/Qwen2.5-0.5B` in `float32`.

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

### V7 layer-aware boundary compressors (experimental)

V7 adds three compressors that keep **full-precision V** on the first *N* layers
and **INT4-sim V** elsewhere, with uniform INT8 K. All are simulated (`is_simulated=True`).

| Name | Boundary depth (N) | Notes |
|---|---|---|
| `k8_v4_boundary_v8_sim` ⚠️ | 1 | First layer V protected |
| `k8_v4_boundary2_v8_sim` ⚠️ | 2 | First two layers V protected |
| `k8_v4_boundary4_v8_sim` ⚠️ | 4 | First four layers V protected |

See [Experiment 006C](docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) for acceptance
results. Leaderboard columns show `mixed 8/4-sim` for V where applicable — metadata
only, not real packed-bit storage.

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
│   ├── __init__.py           # registers 15 built-in compressors
│   ├── base.py               # KVCompressor Protocol, CompressorCapabilities
│   ├── registry.py           # register_compressor, get_compressor, list_compressors
│   ├── backend_adapter.py    # V6: BackendAdapter + PassThroughBackendAdapter
│   ├── noop.py               # NoOpCompressor (identity, acceptance=100%)
│   ├── int8.py               # Int8Compressor (per-tensor symmetric INT8)
│   ├── int4_sim.py           # Int4SimCompressor (simulated INT4, int8 storage)
│   ├── asymmetric_sim.py     # V4: asymmetric K/V quant compressors
│   ├── layer_aware_sim.py    # V7: boundary-layer V policies (N=1/2/4)
│   ├── kvpress_knorm.py      # V6: restricted KVPress KnormPress adapter
│   └── debug_noise.py        # DebugNoiseCompressor (forces rejection, test only)
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
├── analysis/
│   ├── __init__.py           # public API re-exports
│   ├── acceptance_tables.py  # build_acceptance_table, group_by_*, write_acceptance_table_csv
│   ├── mismatch.py           # first_lossy_divergences, mismatch_position_summary
│   └── failure_report.py     # build_failure_report, write_failure_report_json
└── reporting/
    ├── markdown.py           # render_markdown_report, write_markdown_report
    ├── leaderboard.py        # asymmetric + layer-aware leaderboard rendering
    ├── histograms.py         # histogram tables for Markdown reports
    └── examples.py           # divergence / failure example blocks

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

## Earlier experiments

Full Markdown reports for every published sweep. All report **`exactkv_failures == 0`**.

| # | Release | Suite | Runs | Headline | Report |
|---|---|---|---:|---|---|
| 001 | v0.2.0 | smoke (6) | 36 | `int8` accept 0.931 | [`EXPERIMENT_001`](docs/EXPERIMENT_001_SMOKE_SWEEP.md) |
| 002 | v0.3.0 | core (34) | 204 | `int8` accept 0.951 | [`EXPERIMENT_002`](docs/EXPERIMENT_002_CORE_SWEEP.md) |
| 003 | v0.4.0 | core (34) | 612 | Keys more fragile than values | [`EXPERIMENT_003`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md) |
| 004 | v0.5.0 | core (34) | 340 | Workspace memory accounting | [`EXPERIMENT_004`](docs/EXPERIMENT_004_WORKSPACE_MEMORY.md) |
| 005 | v0.6.0 | core (34) | 272 | Restricted kvpress KnormPress | [`EXPERIMENT_005`](docs/EXPERIMENT_005_KVPRESS_KNORM.md) |
| 006 | v0.7.0 | core (34) | 374 | Layer-aware V sweep | [`EXPERIMENT_006`](docs/EXPERIMENT_006_LAYER_AWARE_V.md) |
| 006C | v0.7.0 | core (34) | 170 | Boundary depth ablation | [`EXPERIMENT_006C`](docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) |
| 007 | V8 | core (34) | 238 | Serving harness compatibility | [`EXPERIMENT_007`](docs/EXPERIMENT_007_SERVING_CONTEXT.md) |

No throughput, latency, or speedup is claimed in any experiment report.

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

## Roadmap and research

| Topic | Document |
|---|---|
| v0.8.0 docs | [`RELEASE_NOTES_V0.8.0.md`](docs/RELEASE_NOTES_V0.8.0.md) · [`EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) · [`PROJECT_STATUS_V0.8.0.md`](docs/PROJECT_STATUS_V0.8.0.md) · [`DEFERRED_WORK_REGISTER.md`](docs/DEFERRED_WORK_REGISTER.md) |
| V9 scope (current) | [`V9_SCOPE_STATEMENT.md`](docs/V9_SCOPE_STATEMENT.md) |
| V9+ planning | [`ROADMAP.md`](docs/ROADMAP.md) · [`DEFERRED_WORK_REGISTER.md`](docs/DEFERRED_WORK_REGISTER.md) |
| V6–V8 planning detail | [`docs/FUTURE_ROADMAP_V6_V8.md`](docs/FUTURE_ROADMAP_V6_V8.md) |
| Asymmetric K/V research | [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md) |
| Related work survey | [`docs/RELATED_WORK_KV_CACHE_COMPRESSION.md`](docs/RELATED_WORK_KV_CACHE_COMPRESSION.md) |
| Experiment backlog | [`docs/RESEARCH_BACKLOG.md`](docs/RESEARCH_BACKLOG.md) |

---

## Key documents

**Start here**

| Document | Purpose |
|---|---|
| [`docs/RELEASE_NOTES_V0.8.0.md`](docs/RELEASE_NOTES_V0.8.0.md) | Latest release (`v0.8.0`) |
| [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) | All experiments 001–007 |
| [`docs/PROJECT_STATUS_V0.8.0.md`](docs/PROJECT_STATUS_V0.8.0.md) | Project status (not launch-final) |
| [`docs/DEFERRED_WORK_REGISTER.md`](docs/DEFERRED_WORK_REGISTER.md) | Deferred work V9–v1.0.0 |
| [`docs/EXPERIMENT_007_SERVING_CONTEXT.md`](docs/EXPERIMENT_007_SERVING_CONTEXT.md) | Latest experiment — serving harness (238 runs) |
| [`docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) | Experiment 006C — boundary-depth ablation |
| [`docs/V9_SCOPE_STATEMENT.md`](docs/V9_SCOPE_STATEMENT.md) | V9 — real backend integration gauntlet |
| [`docs/SERVING_CACHE_LIFECYCLE_HARNESS.md`](docs/SERVING_CACHE_LIFECYCLE_HARNESS.md) | V8 Phase B — cache-lifecycle harness |
| [`docs/SERVING_CONTEXT_FEASIBILITY.md`](docs/SERVING_CONTEXT_FEASIBILITY.md) | V8 Phase A — serving feasibility |
| [`docs/V8_SCOPE_STATEMENT.md`](docs/V8_SCOPE_STATEMENT.md) | V8 scope (in progress) |

**Experiments** (all with `exactkv_failures == 0`)

| # | Report |
|---|---|
| 003 | [`EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md) |
| 004 | [`EXPERIMENT_004_WORKSPACE_MEMORY.md`](docs/EXPERIMENT_004_WORKSPACE_MEMORY.md) |
| 005 | [`EXPERIMENT_005_KVPRESS_KNORM.md`](docs/EXPERIMENT_005_KVPRESS_KNORM.md) |
| 006 | [`EXPERIMENT_006_LAYER_AWARE_V.md`](docs/EXPERIMENT_006_LAYER_AWARE_V.md) |
| 006A | [`EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md`](docs/EXPERIMENT_006A_DIVERGENCE_ANALYSIS.md) |
| 006C | [`EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md`](docs/EXPERIMENT_006C_BOUNDARY_DEPTH_ABLATION.md) |
| 007 | [`EXPERIMENT_007_SERVING_CONTEXT.md`](docs/EXPERIMENT_007_SERVING_CONTEXT.md) |

**Release notes:** [`v0.4.0`](docs/RELEASE_NOTES_V0.4.0.md) · [`v0.5.0`](docs/RELEASE_NOTES_V0.5.0.md) · [`v0.6.0`](docs/RELEASE_NOTES_V0.6.0.md) · [`v0.7.0`](docs/RELEASE_NOTES_V0.7.0.md) · [`v0.8.0`](docs/RELEASE_NOTES_V0.8.0.md)

**Scope and design:** [`V6_SCOPE_STATEMENT.md`](docs/V6_SCOPE_STATEMENT.md) · [`V7_SCOPE_STATEMENT.md`](docs/V7_SCOPE_STATEMENT.md) · [`BACKEND_ADAPTER_INTERFACE.md`](docs/BACKEND_ADAPTER_INTERFACE.md) · [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](docs/RELATED_WORK_KV_CACHE_COMPRESSION.md)

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
