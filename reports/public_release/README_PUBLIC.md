# ExactKV Public Benchmark Release

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM
KV-cache compression. It measures token-level drift, first divergence, acceptance
rate, verifier agreement, and exactness failures across compressors and models.

ExactKV grew through a long verifier-first research arc (**V1–V21**) before the formal release
phases. See [`docs/PROJECT_LINEAGE.md`](../../docs/PROJECT_LINEAGE.md),
[`docs/VERSION_LINEAGE.md`](../../docs/VERSION_LINEAGE.md), and
[`docs/HISTORICAL_ARTIFACT_INVENTORY.md`](../../docs/HISTORICAL_ARTIFACT_INVENTORY.md).

**Not a production serving system.** ExactKV does not reproduce VeriCache serving throughput.

## Quick start

```bash
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check
python3 scripts/build_launch_pack.py
python3 scripts/exactkv.py run publish
```

## Public release benchmark (authoritative)

- **Evidence track:** Phase H+ scale_7b (authoritative public release)
- **Headline external cells:** 8,132 (`exactkv_failures = 0`)
- **Core leaderboard panel:** 1,500 cells (`reports/scale_7b/raw.json`)
- **Models evaluated:** meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-Instruct-v0.3
- **ExactKV failures:** 0
- **Deterministic mode:** False
- **Divergence authority:** Phase G `FirstDivergenceAuthority` (canonical)

- **Historical Phase A panel (internal):** 336 cells — supporting cross-model evidence, not the final public release benchmark.

## Artifacts

| File | Description |
|------|-------------|
| `leaderboard_final.json` | Ranked compressor × model scores (from scale_7b when available) |
| `benchmark_summary.md` | Aggregate metrics |
| `methodology.md` | Evaluation methodology + claim boundaries |
| `demo_cards.json` / `demo_cards.md` | Release + historical demo cards |
| `launch_manifest.json` | Phase K launch pack manifest |
| `repro_command.sh` | Reports-only reproduction |
| `release_manifest.json` | Source artifact pointers |

## Documentation links

- Technical report: [`docs/EXACTKV_TECHNICAL_REPORT.md`](../../docs/EXACTKV_TECHNICAL_REPORT.md)
- Project lineage: [`docs/PROJECT_LINEAGE.md`](../../docs/PROJECT_LINEAGE.md)
- Version lineage (V1–V21): [`docs/VERSION_LINEAGE.md`](../../docs/VERSION_LINEAGE.md)
- Novelty audit: [`docs/NOVELTY_AUDIT.md`](../../docs/NOVELTY_AUDIT.md)
- Claim boundaries: [`docs/CLAIM_BOUNDARIES.md`](../../docs/CLAIM_BOUNDARIES.md)
- Metric definitions: [`docs/METRIC_DEFINITIONS.md`](../../docs/METRIC_DEFINITIONS.md)
- Reproducibility: [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md)
- Demo cards: [`demo_cards.md`](demo_cards.md)

## Claims policy

No end-to-end speedup, latency, or active GPU memory savings claims. Phase F results (when cited) are kernel microbenchmark only. Compression ratios are stored tensor byte ratios unless active GPU memory is explicitly measured. Scale run used sequential model execution (volume constraint).

Generated: 2026-06-26T14:48:46.545545+00:00
