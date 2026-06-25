# ExactKV Launch Blog (Phase K — Final)

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures when compressed KV paths first diverge token-by-token, how much of the draft remains accepted, and whether verifier-backed execution preserves exactness.

## A long research arc before the release benchmark

ExactKV **did not start at Phase A**. Before the formal release pipeline, the project spent many months on verifier-first prototypes across the **V1–V21 version arc** (V1–V13 scope statements plus V14–V21 safety/runtime ladder work), more than 120 experiment documents, trace-level correctness work, structured-output demos, a safety ladder with L3/L4 no-commit boundaries, shadow observer runtime probes, and explicit no-go investigations for vLLM, LMCache, and unqualified memory/timing claims. Release Gate R2 catalogued **1,176** historical artifacts in [`docs/HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md). See [`docs/VERSION_LINEAGE.md`](VERSION_LINEAGE.md) for per-version evidence. Phases A–J then formalized, scaled, packaged, validated, and released that earlier system — they did not invent the verifier-mediated exactness question.

## What we are releasing

The public headline is a **1500-cell** real-GPU benchmark on:

- **meta-llama/Llama-3.1-8B**
- **mistralai/Mistral-7B-Instruct-v0.3**

Source of truth: `reports/scale_7b/raw.json` — float16, `deterministic_mode=false`, **`exactkv_failures = 0`**.

Compressors in the panel: `noop`, `int8`, `int4_sim`, `spectralquant`, `shard`.

### What the leaderboard measures

- **First divergence** — the token index where lossy compressed-KV greedy output first disagrees with full-KV reference
- **Acceptance rate** — how much of each draft prefix the full-KV verifier accepts
- **Verifier agreement** and **exactness failures** — whether ExactKV's final output matches full-KV greedy on tested cells

Top Llama rows: `noop` and `int8` score **1.0** with full acceptance. `int4_sim` shows lower acceptance (~0.85) while keeping **zero** ExactKV failures on the panel. Mistral rows are fully numeric after Release Gate R1 (e.g. `int8` score **0.983**, acceptance **1.0**).

Full tables: [`reports/public_release/leaderboard_final.json`](../reports/public_release/leaderboard_final.json).

## How it works (verifier-first)

1. Draft tokens from compressed KV.
2. Verify each token against full-KV greedy predictions.
3. Accept matching prefixes; correct on mismatch.
4. Record drift metrics and failures.

This is evaluation infrastructure — **not a production serving system**.

## Reproduce in one command

```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check
```

Technical report: [`docs/EXACTKV_TECHNICAL_REPORT.md`](EXACTKV_TECHNICAL_REPORT.md)  
Project lineage: [`docs/PROJECT_LINEAGE.md`](PROJECT_LINEAGE.md)

## Required caveats

| Topic | Caveat |
|-------|--------|
| **VeriCache** | Closest conceptual prior art for draft/verify semantics. **ExactKV does not reproduce VeriCache** serving throughput. |
| **Serving** | ExactKV is **not a production serving system**. |
| **Phase F speedups** | INT8 **1.63×** and INT4 **1.54×** are **kernel microbenchmark** results on a fixed KV shape — **not end-to-end** inference speedups. `block_sparse` is not Triton-accelerated in current evidence. |
| **Compression ratio** | **Stored tensor byte ratios** only; VRAM not measured on this panel |
| **SpectralQuant** | **Fallback/proxy** in the current environment (`spectralquant_available=False`) |
| **Shard** | **Probe-first** heuristic analysis (`probe_only=True`); not full Shard / ShardCache integration |
| **Execution** | Scale run used **sequential** model execution due to RunPod volume constraints. |

## Read next

- [`docs/CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md)
- [`docs/NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md)
- [`reports/public_release/demo_cards.md`](../reports/public_release/demo_cards.md)

ExactKV asks a simple question with hard evidence: **when does compressed KV start lying?** The release package answers it with token-level metrics, a public leaderboard, and explicit boundaries on what we do not claim.
