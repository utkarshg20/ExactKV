# ExactKV

**Lossy KV-cache compression. Exact full-KV outputs.**

ExactKV is an open-source inference runtime and benchmark suite that lets lossy KV-cache
compressors draft tokens quickly, then verifies those tokens against full-KV decoding so
the final output remains identical to normal full-KV inference under deterministic
(greedy) decoding.

Inspired by the [VeriCache paper](https://arxiv.org/abs/2605.17613).
ExactKV is not a reimplementation — it is a compressor-agnostic platform for verified
KV-cache generation and benchmark evaluation.

## Status

**V1 — correctness prototype.**

V1 proves the core ExactKV loop:
- Full-KV generation matches `model.generate` exactly.
- Compressed-KV drafts are verified against full KV.
- Output token IDs match full-KV decoding exactly under greedy decoding.
- Accept/reject/correction bookkeeping is correct.

V1 does **not** claim production-ready throughput or VeriCache-level performance.
See `docs/V1_SCOPE_STATEMENT.md`.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/
python examples/qwen_smoke.py
```

## Phase 1 scope

- Model: `Qwen/Qwen2.5-0.5B` (or TinyLlama)
- Decoding: greedy only (`do_sample=False`)
- Single request, single device
- Compressors: NoOp, INT8 (per-tensor symmetric), DebugNoise
- No vLLM, LMCache, Triton, CUDA kernels, CPU offload, batching, sampling, INT4

## Project structure

```
exactkv/
├── runtime/          # ModelRuntime, greedy generation, ExactKVGenerator
├── cache/            # FullKVState, CompressedKVState, cache utils
├── compressors/      # KVCompressor interface, NoOp, INT8, debug_noise
├── verification/     # VerificationEngine, AcceptanceResult
├── metrics/          # exactness, acceptance, memory
└── benchmarks/       # BenchmarkRunner, prompt suites
```
