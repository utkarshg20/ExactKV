# ExactKV Claim Boundaries (Phase J)

Authoritative allowed / qualified / forbidden public claims. See also [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) and [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md).

**Positioning:** ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures token-level drift, first divergence, acceptance rate, verifier agreement, and exactness failures across compressors and models.

---

## Allowed

- Compressor-agnostic token-level drift benchmark
- First-divergence measurement across compressors
- Acceptance-rate and verifier-agreement leaderboard
- Real 7B/8B benchmark evidence (Phase H+ scale_7b, 1500 cells, current artifact)
- Research-grade evaluation framework
- `exactkv_failures == 0` on cited panels
- SpectralQuant **fallback/proxy** support (disclosed)
- Shard **probe-first** analysis (disclosed)

---

## Qualified

| Claim | Qualification |
|-------|----------------|
| Public benchmark platform | Repo-local artifacts; not hosted SaaS |
| SpectralQuant support | Fallback/proxy when `spectralquant_available=False` |
| Shard support | Probe-first heuristic; not full Shard integration |
| Speedups | Phase F **kernel microbenchmark** only — not end-to-end |
| Compression ratios | **Stored tensor byte ratios** unless active GPU memory measured |
| Systems diagnostic peaks / wall-clock | Observed peak CUDA allocation and per-path wall-clock on the **96-cell `systems_diagnostic` panel** (7B/8B). Peak includes model weights + KV + temporaries. **Not** serving throughput, TTFT, RPS, or unqualified production VRAM savings. |
| Triton KV kernel path | Tested shape/hardware; block_sparse uses torch backend |
| VeriCache relationship | Inspired by draft/verify semantics; **does not reproduce** VeriCache |

---

## Forbidden

- production ready / production serving system
- first ever / first and only / nothing like this exists
- active GPU memory savings / VRAM savings (unqualified)
- end-to-end speedup / faster inference (unqualified)
- beats VeriCache / reproduces VeriCache
- invented compressed-KV verification
- real SpectralQuant (without real dependency evidence)
- real Shard (without full adapter evidence)
- fastest / SOTA (unless explicitly sourced and qualified)

---

## Required caveats in public copy

When mentioning the topic, include:

| Topic | Required caveat language |
|-------|--------------------------|
| Phase F / speedup | kernel microbenchmark; not end-to-end |
| compression ratio | stored tensor byte ratio |
| systems diagnostic | diagnostic peak CUDA / harness wall-clock; not serving RPS or production VRAM savings |
| SpectralQuant | fallback / proxy |
| Shard | probe-first |
| VeriCache | does not reproduce |
| production | not a production serving system |

Enforced by `scripts/audit_public_claims.py` and `scripts/check_public_release.py`.
