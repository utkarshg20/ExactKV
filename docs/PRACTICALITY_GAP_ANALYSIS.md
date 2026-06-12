# Practicality Gap Analysis

**Status:** V12 Phase 7 / Experiment 027 — claim-boundary review only.
**Purpose:** Concise map of what ExactKV still lacks before speed or active GPU memory claims are defensible.

> This document does **not** authorize positive speed, throughput, latency, runtime,
> tokens/sec, active GPU memory savings, production-serving, or model accuracy
> improvement claims. It lists gaps and required next builds only.

See also: [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md).

---

| Missing piece | Current status | Why it matters | Required next build | Priority |
| ------------- | -------------- | -------------- | ------------------- | -------- |
| **Performance proof** | **Not measured.** No tokens/sec, throughput, latency, speedup, or `runtime_seconds` in standard reports. Sequential verify adds overhead (up to `draft_len−1` full forwards per round). | External feedback and practical-systems reviewers expect a speed story before claiming KV compression is useful in production. | Parallel/span verification; accepted-tokens-per-verifier-call metric; warmup-controlled benchmark harness; baselines: full greedy, lossy-only draft, ExactKV sequential, ExactKV span-verify (future). | **P0** |
| **Active GPU memory proof** | **Not proven.** V5 `total_kv_footprint_bytes` is shape accounting, not device peak. Exp 018 pilot shows GPU peak dominated by model weights (~2 GiB) at tested scale; compressor ordering on V5 does not match peak ordering. `active_gpu_kv_bytes` not in standard schema. | Reviewers conflate accounting footprint with VRAM savings; without isolated methodology, memory claims would be misleading. | Avoid full materialization where possible; true packed attention path; backend keeping compressed KV active on device; isolated methodology separating weights, KV, temporaries, allocator noise (extend Exp 018 protocol). | **P0** |
| **Parallel verification** | **Not implemented.** V1 hard constraint: sequential one-token verify (`VerificationEngine.verify_sequential`). Documented in `exactkv_generator.py` and `engine.py`. D21 deferred. | One-token-at-a-time verify cannot amortize verifier cost across draft spans; blocks any honest speed headline. | Draft *k* tokens → one full-KV forward over span → accept prefix until mismatch → correct → commit span. Touch: `exactkv_generator.py`, `verification/engine.py`, acceptance bookkeeping tests. | **P0** |
| **Hot adapter** | **Partial.** Factory-only restricted adapters (TurboQuant Python, KIVI offline, KVQuant simquant, kvpress KnormPress). No SnapKV, Shard/ShardKV, or other widely cited production-legible methods integrated. | Public audiences recognize SnapKV/Shard-style names; Qwen-only `_sim` compressors are technically valid but not legible as “state of the art.” | Restricted factory adapter for one hot method (SnapKV or ShardKV) with `exactkv_failures == 0` on a small panel; clear `supports_real_bytes_claim` labeling. | **P1** |
| **Llama-3.1-8B** | **Not run.** All published V10 suite cells use Qwen2.5 (0.5B–3B). Results transfer within Qwen family; Llama is the public benchmark lingua franca. | Blog posts, demos, and reviewer trust often anchor on Llama-3.x; Qwen-only story reads as niche. | Future small suite (10–20 prompts): full greedy, int8, best repair policy; optional span-verify once built; HF Llama-3.1-8B-Instruct or base. | **P1** |
| **Visual plots** | **Not produced for launch.** Markdown tables exist in experiment reports; no curated figure bundle. | Communicating acceptance, divergence, and exactness at a glance requires charts; tables alone do not satisfy “killer demo” expectations. | Plot specs (see Exp 027 §12): acceptance by compressor/policy, first-divergence histogram, category leaderboard, rejection/correction flow, K vs V error, exactness=0 summary. Generate from gitignored JSON/CSV in V13. | **P1** |
| **Killer demo** | **Not scripted.** Harness exists (Exp 007/017 sidecar); no single narrated JSON/tool prompt walkthrough showing drift → catch → correct → exact match. | One memorable trace convinces more than aggregate acceptance tables. | Scripted demo: structured JSON/tool prompt, lossy draft drifts, ExactKV catches mismatch, verifier corrects, final output matches full greedy; optional sidecar metadata overlay. | **P1** |
| **Serving-ish demo** | **Sidecar probe only (Exp 017).** Metadata-only observer; direct vLLM/LMCache **no-go**. Not multi-request, not batched, not paged. | “Serving-ish” signals production path without claiming vLLM integration. | Extend sidecar harness with recorded lifecycle timeline on 1–2 prompts; still observational, not production serving. | **P2** |
| **One headline number** | **Exactness and acceptance only.** Allowed: `exactkv_failures == 0`, divergence cells caught, Exp 020 pilot rejection reduction, Exp 025 full-suite repair-policy acceptance (carefully worded). Forbidden: speedup, throughput, latency, active GPU memory savings, production readiness, accuracy improvement. | Launch narratives need one memorable stat; must not overclaim. | Pick from allowed set after V13 practicality work; e.g. “128-prompt suite, 896 cells, zero exactness failures” plus best policy acceptance on hard categories — **not** tokens/sec. | **P1** |

---

## Recommended sequencing (V13 Practicality Proof)

1. **Parallel/span verification** (unblocks performance methodology)
2. **Warmup-controlled benchmark harness** (diagnostic timing only; no launch headline until baselines complete)
3. **Isolated GPU memory methodology** (extend Exp 018; still no savings claim until compressed-active path exists)
4. **One hot adapter + Llama small suite** (public legibility)
5. **Plots + killer demo** (communication layer)
6. **Phase 8 release decision** — defer public v1.0.0 if P0 items remain open

---

## Related

- [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — V13 Practicality Proof (Phase 0 active)
- [`V12_SCOPE_STATEMENT.md`](V12_SCOPE_STATEMENT.md) — Phase 7 complete after Exp 027
- [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md) — Exp 018 pilot protocol
- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) — prior launch gate (v1.0.0 not yet)
