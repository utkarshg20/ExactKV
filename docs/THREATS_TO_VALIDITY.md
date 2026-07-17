# Threats to validity (ExactKV research release)

**Audience:** external reviewers · **git tag:** `v-release`

ExactKV is a **research-grade evaluation framework**, not a production serving system. This document lists the main threats to validity and how the release mitigates or bounds them.

## 1. Task and benchmark coverage

| threat | mitigation / boundary |
|--------|------------------------|
| Panels are smoke/drift tests, not official benchmark scores | Claim boundaries in [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md); external panels labeled drift diagnostics |
| Prompt suites are stress panels, not full benchmark splits | 50-variant core panel documented in technical report §5 |
| Faithful adapters are appendix smoke grids, not headline evidence | Separate artifact tree; wave-2 is Mistral-only smoke |

## 2. Compressor honesty tiers

| tier | threat | boundary |
|------|--------|----------|
| `_sim` compressors | Not upstream product ports | Diagnostic only; labeled in README and leaderboard |
| SpectralQuant | Real dependency often unavailable | Fallback/proxy mode; not headline evidence |
| Shard | Probe-first heuristic | Not full Shard / ShardCache integration |
| KIVI / TurboQuant adapters | Integration diagnostics | Restricted env; not production CUDA paths |

## 3. Metrics and semantics

| threat | mitigation |
|--------|------------|
| `exactkv_failures = 0` misread as “compression is safe” | Documented as harness wiring gate on cited panels only |
| Divergence rate vs stability score confusion | Technical report tables label divergence rate explicitly |
| Stored-byte ratios vs active GPU memory | Repeated in README, site, and report; no VRAM savings claims |

## 4. Systems measurements

| measurement | scope | artifact |
|-------------|-------|----------|
| Phase F kernel latency | **Microbenchmark only** — fixed KV shape, not end-to-end decode | `reports/systems/latency_microbench.json` |
| Stored tensor memory before/after compress | Same microbench shape; not serving-time peak VRAM | `reports/systems/gpu_memory_trace.json` |
| Systems diagnostic (96 cells, 7B/8B) | Peak CUDA allocation + per-path wall-clock (full/lossy/ExactKV); **not** serving RPS/TTFT/unqualified VRAM savings | `reports/systems/systems_diagnostic.json` |
| Per-token verifier overhead on 7B panels | **Not measured** in headline 1,500-cell panel | Do not infer from Phase F alone |
| Wall-clock panel runtime | Hardware-dependent; systems_diagnostic reports harness path timing only | Re-run on your GPU for local timing |

## 5. Reproducibility threats

| threat | mitigation |
|--------|------------|
| Floating nondeterminism on GPU | Headline panel uses `deterministic_mode=false`; document when comparing exact token match |
| HF model revision drift | Pin model IDs in report; record transformers/torch versions in §5 stack table |
| Large raw JSON not in release zip | Full headline JSON committed in repo; bundle ships index + checksums |

## 6. External validation gaps (known)

- **TurboQuant experimental** (wave-2 smoke): strongest faithful structured-task drift so far (3.1% combined on MBPP+BFCL smoke), but **128 cells, Mistral only** — not merged into 8,132 headline.
- **int8** remains the only non-catastrophic **real** built-in compressor on faithful wave-1 (~8–9% drift).
- No third-party independent rerun is bundled; reviewers should run [`REPRODUCE.md`](../REPRODUCE.md) CPU path minimum.

## 7. Reviewer checklist

1. Confirm **only** git tag [`v-release`](https://github.com/utkarshg20/ExactKV/releases/tag/v-release) on [Tags](https://github.com/utkarshg20/ExactKV/tags).
2. Confirm [CI](https://github.com/utkarshg20/ExactKV/actions/workflows/ci.yml) green on latest `main`.
3. Run CPU repro in [`REPRODUCE.md`](../REPRODUCE.md) (~2 min).
4. Read [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md) before interpreting drift tables.
5. Treat Phase F speedups as **kernel microbenchmark only**.
