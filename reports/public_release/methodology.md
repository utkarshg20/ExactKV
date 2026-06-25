# ExactKV Evaluation Methodology

## Public release evidence

The authoritative public release benchmark is **Phase H+ scale_7b** (`reports/scale_7b/raw.json`, 1500 cells, real GPU, `deterministic_mode=false`). Phase A (336 cells) remains internal/historical supporting evidence.

## Divergence (canonical)

All divergence metrics use Phase G `FirstDivergenceAuthority`:
- `canonical_first_divergence_index`
- Types: `token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`

## Acceptance

Token-level ExactKV speculative decoding acceptance rate from benchmark cells.

## Leaderboard scoring (locked)

```
score = 0.35 * acceptance_rate
      + 0.25 * verifier_agreement
      + 0.20 * (1 - normalized_first_divergence)
      + 0.10 * (1 - failure_rate)
      + 0.10 * stability_score
```

## Compressors

Built-in: noop, int8, int4_sim, k8_v4_sim
Phase H+ scale panel: noop, int8, int4_sim, spectralquant (fallback/proxy), shard (probe-first)
External adapters: spectralquant_real (fallback mode when dependency missing), shard_real (probe-first heuristic)

## Claim boundaries (Phase I/J)

- ExactKV is **not a production serving system**.
- Phase F speedups (when cited) are **kernel microbenchmark** results only — **not end-to-end** inference speedups.
- Compression ratios are **stored tensor byte ratios** unless active GPU memory is explicitly measured.
- **SpectralQuant** uses **fallback/proxy** mode when the real dependency is unavailable.
- **Shard** (`shard_real`) is **probe-first** heuristic analysis, not a full Shard integration.
- ExactKV is inspired by verifier-mediated compressed-KV ideas; it does **not reproduce VeriCache** serving throughput.

## Reproducibility

Regenerate from existing reports without re-inference:

```bash
python3 scripts/exactkv_repro.py --reports-only
```
