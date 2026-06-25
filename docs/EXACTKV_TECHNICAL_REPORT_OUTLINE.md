# ExactKV Technical Report — Outline (Phase K preparation)

> Full technical report not yet finalized. This outline incorporates Release Gate R2 project lineage requirements.

## Required future section: Project Lineage — From Verifier Prototypes to Release Benchmark

### Bullet notes (from discovered inventory)

- ExactKV **did not start at Phase A**; pre-A arc spans **V1–V21**, Experiments 001–113+, Phases 11–21, and extensive safety/shadow/runtime probe work.
- **Early foundation:** VISION + V1–V3 scope; smoke/core sweeps (Exp 001–002); asymmetric KV (Exp 003).
- **Verifier-first core:** draft/verify/commit loop; `exactkv_failures == 0` hard gate; span verification (Exp 028–029).
- **Trace correctness:** acceptance rate, first divergence, verifier agreement — V10 suites, Phase G `FirstDivergenceAuthority`.
- **Compression studies:** int8/int4 sim, layer-aware V, TurboQuant/KIVI/KVQuant/SpectralQuant/Shard probes — adapter honesty caveats apply.
- **Demos:** Exp 034/034b pharmacy correction, terminal crash-test, LongBench-style drift — illustrative, not throughput.
- **Safety ladder:** L3 guarded draft-shadow no-commit; L4 verifier-mediated dry-run specs; integration safety (Phase 18A).
- **Shadow/runtime probes:** Exp 076–085 generation/decode shadow observers — diagnostic only.
- **No-go investigations:** vLLM (Exp 059–065), LMCache prototype path, Exp 027 truth boundary — inform forbidden claims.
- **Formal A–J:** benchmark formalization, kernels (F), truth engine (G), platform/scale (H+), novelty (I), release freeze (J).
- **Release gates:** R0 evidence integrity, R1 Mistral leaderboard repair, R2 lineage archaeology.
- **Authoritative now:** `reports/scale_7b/raw.json` (1500 cells), `reports/public_release/*`, claim boundaries in Phase I/J docs.

### Suggested report sections (full document)

1. Abstract
2. Introduction & problem framing
3. **Project lineage (pre-A through release)** ← required
4. Verifier-mediated exactness methodology
5. Benchmark design (Phase A–J)
6. Scale 7B/8B results (authoritative)
7. Kernel microbenchmark evidence (qualified)
8. Prior art & claim boundaries
9. Limitations & future work
10. Reproducibility appendix

### Regenerate lineage data

```bash
python3 scripts/build_project_lineage.py
python3 scripts/check_project_lineage.py
```

See: [`PROJECT_LINEAGE.md`](PROJECT_LINEAGE.md) · [`HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md)
