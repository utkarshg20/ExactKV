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

## Project structure

```
exactkv/
├── config.py             # ExactKVConfig dataclass
├── runtime/
│   ├── model_runtime.py      # ModelRuntime (HF model + tokenizer wrapper)
│   ├── generation.py         # generate_full_greedy, generate_lossy_greedy
│   └── exactkv_generator.py  # ExactKVGenerator (draft-verify-commit loop)
├── cache/
│   ├── full_state.py         # FullKVState (authoritative)
│   ├── compressed_state.py   # CompressedKVState (compressor-specific)
│   └── utils.py              # kv_seq_len, extract_kv_tensors, rebuild_cache
├── compressors/
│   ├── base.py               # KVCompressor Protocol, CompressionStats
│   ├── noop.py               # NoOpCompressor (identity, acceptance=100%)
│   ├── int8.py               # Int8Compressor (per-tensor symmetric INT8)
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
    └── runner.py             # run_one, run_suite, RunConfig

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
