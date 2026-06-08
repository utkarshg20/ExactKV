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

## Status

**V1 — correctness prototype.**

| Gate | Status |
|---|---|
| `generate_full_greedy` matches `model.generate` | ✅ |
| NoOp ExactKV output == full greedy | ✅ |
| INT8 ExactKV output == full greedy | ✅ |
| DebugNoise forces rejection + ExactKV corrects | ✅ |
| Metrics reconcile (drafted == accepted + rejected) | ✅ |
| Benchmark runner emits valid JSON; `exactkv_failures == 0` | ✅ |

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

Expected: **all tests pass** in ~90–120 s on CPU with `Qwen/Qwen2.5-0.5B` in `float32`.

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

## Project structure

```
exactkv/
├── config.py             # ExactKVConfig, BenchmarkConfig
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
│   ├── base.py               # KVCompressor Protocol, CompressionStats, CompressorCapabilities
│   ├── registry.py           # register_compressor, get_compressor, list_compressors
│   ├── __init__.py           # registers all built-in compressors
│   ├── noop.py               # NoOpCompressor (identity, acceptance=100%)
│   ├── int8.py               # Int8Compressor (per-tensor symmetric INT8)
│   ├── int4_sim.py           # Int4SimCompressor (simulated INT4, int8 storage)
│   └── debug_noise.py        # DebugNoiseCompressor (forces rejection, test only)
├── verification/
│   ├── acceptance.py         # compute_acceptance, AcceptanceResult, traces
│   └── engine.py             # VerificationEngine.verify_sequential
├── metrics/
│   ├── exactness.py          # token_exact_match, first_divergence_idx
│   ├── acceptance.py         # summarize_acceptance, AcceptanceSummary
│   └── memory.py             # estimate_kv_memory, MemorySummary
└── benchmarks/
    ├── prompts.py            # load_prompts, load_smoke_prompts
    ├── runner.py             # run_one, run_suite, RunConfig
    ├── reports.py            # write_json_report, write_csv_report, build_run_manifest
    └── sweeps.py             # run_sweep (multi-compressor × multi-draft-length)
├── analysis/
│   ├── acceptance_tables.py  # build_acceptance_table, group_by_*, write_acceptance_table_csv
│   ├── mismatch.py           # first_lossy_divergences, mismatch_position_summary
│   └── failure_report.py     # build_failure_report, list_exactkv_failures, write_failure_report_json

benchmarks/
└── prompts/
    └── smoke.jsonl           # 16 prompts across 6 categories

examples/
└── qwen_smoke.py             # Demo script with side-by-side comparison

docs/
├── V1_SCOPE_STATEMENT.md
└── IMPLEMENTATION_PLAN.md

tests/
├── test_acceptance_logic.py
├── test_full_generation.py
├── test_verification_engine.py
├── test_noop_exactkv.py
├── test_int8_compressor.py
├── test_int8_exactkv.py
├── test_debug_noise_exactkv.py
├── test_lossy_generation.py
├── test_metrics.py
├── test_benchmark_runner.py
└── test_example_script.py
```

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

table = build_acceptance_table(sweep_report)        # per (compressor × draft_len)
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

divergences = first_lossy_divergences(sweep_report)  # first_divergence_idx per result
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
print(f"Status: {fr['status']}")            # "pass" or "fail"
print(f"ExactKV failures: {fr['exactkv_failure_count']}")    # must be 0
print(f"Lossy divergences: {fr['lossy_divergence_count']}")  # expected for real compressors

write_failure_report_json(fr, "reports/failures.json")
```

**Key distinction:**
- **Lossy divergence** (`lossy.token_exact_match == False`) is *expected*. It proves that the compressor changes the output and demonstrates why verification is necessary.
- **ExactKV failure** (`exactkv_failure == True`) means the verified output did *not* match `generate_full_greedy`. This is a correctness bug and must always be zero.

No timing, throughput, latency, or speedup metrics are produced by any analysis function.

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

## V2 CLI

`python -m exactkv` exposes the most-used ExactKV operations as CLI subcommands.
The CLI reports **exactness and acceptance behaviour only** — it does **not**
report tokens/second, latency, throughput, or speedup.

### list-compressors

Print all registered compressors with their capabilities:

```bash
python -m exactkv list-compressors
```

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

### int4_sim disclaimer

`int4_sim` is a **simulated** INT4 compressor. All CLI outputs set
`is_simulated=True` and `supports_real_bytes_claim=False` for `int4_sim` rows.
Do **not** interpret `int4_sim` memory numbers as real packed-4-bit savings.

---

## V1 design principles

- **Correctness first.** Every output token ID must match `generate_full_greedy`.
- **Compressor-agnostic.** Any class satisfying the `KVCompressor` protocol can be plugged in.
- **No speedup claims.** V1 proves losslessness, not throughput.
- **Greedy only.** Sampling, beam search, and speculative bonus tokens are V2+.
- **Sequential verification.** Parallel verification is V2+.
- **DynamicCache note.** The cache utilities target transformers 5.8.1 internal structure;
  see `docs/V1_SCOPE_STATEMENT.md` for brittleness notes.

---

## Citation

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is a
compressor-agnostic, Hugging Face-first implementation, a structured benchmark harness,
and a framework for evaluating compressors by acceptance behaviour under full-KV
verification.
