# ExactKV Launch — X Thread (Phase K Final)

**1/** ExactKV is a compressor-agnostic crash-test + leaderboard for LLM KV-cache compression.

It measures when compressed KV first diverges token-by-token, draft acceptance rate, and whether verifier-backed execution stays exact.

**2/** Important context: ExactKV did NOT start at Phase A.

Years of verifier-first prototypes (**V1–V21** version arc, 120+ experiment docs, safety ladder, shadow probes, no-go serving investigations) came first. A–J formalized & released that research.

Lineage: docs/PROJECT_LINEAGE.md

**3/** Public headline benchmark:

• 1500 cells, real GPU (float16)
• meta-llama/Llama-3.1-8B + mistralai/Mistral-7B-Instruct-v0.3
• exactkv_failures = 0
• Source: reports/scale_7b/raw.json

**4/** Metrics that matter:

• first_divergence_index — when lossy KV starts lying
• acceptance_rate — how much draft the full-KV verifier accepts
• verifier agreement + exactness failures

Leaderboard: reports/public_release/leaderboard_final.json

**5/** Llama-3.1-8B top rows (release panel):

noop 1.0 | int8 1.0 | int4_sim ~0.86 acceptance | shard probe-first lower

Mistral rows fully numeric after R1 repair (int8 score 0.983, acceptance 1.0).

**6/** Verifier loop:

draft from compressed KV → verify vs full KV → accept prefix / correct → repeat.

Research evaluation framework. NOT a production serving system.

**7/** Reproduce:

python3 scripts/exactkv_repro.py --reports-only
python3 scripts/exactkv_repro.py --release-check

Full report: docs/EXACTKV_TECHNICAL_REPORT.md

**8/** Caveats (read before citing):

• Does NOT reproduce VeriCache serving throughput
• Phase F INT8 1.63× / INT4 1.54× = kernel microbenchmark ONLY (not end-to-end speedup)
• Compression ratios = stored tensor byte ratios (VRAM not measured; no memory-savings claim)
• SpectralQuant = fallback/proxy in this environment
• Shard = probe-first heuristic (not full Shard / ShardCache integration)
• Sequential model execution on scale run (volume constraint)

**9/** Demo cards + historical inventory:

reports/public_release/demo_cards.md
docs/HISTORICAL_ARTIFACT_INVENTORY.md (1176 artifacts)

**10/** ExactKV: when does compressed KV start lying?

Token-level drift evidence. Public leaderboard. Hard claim boundaries.

docs/launch_blog_final.md | docs/CLAIM_BOUNDARIES.md
