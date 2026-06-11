# Launch Narrative Draft (Deferred — Not for Public Posting)

**Status:** V11 Phase 6 draft. **Do not publish externally** until v1.0.0 gate review.
**Tone:** Technical, precise, honest. **Not marketing copy.**

> This draft explains ExactKV's research story. It is **not** approved for public launch.
> It must not be read as claiming speed, active GPU memory savings, production serving,
> model accuracy improvement, or universal benchmark status.

---

## Draft narrative

KV-cache compression promises cheaper autoregressive inference — but lossy caches change
what the model would have predicted. A single wrong draft token can force expensive
corrections or silently change the generation trajectory. **ExactKV** asks a narrower,
verifiable question: *can we use compressed KV as a draft state while guaranteeing that
the final greedy output matches full-precision inference exactly?*

ExactKV implements a **draft → verify → commit** loop inspired by the VeriCache idea
([arxiv:2605.17613](https://arxiv.org/abs/2605.17613)). Compressors propose tokens using
a lossy KV cache. A **full-KV verifier** checks each draft token against authoritative
full-precision predictions. The loop accepts matching prefixes, corrects on mismatch,
and recompresses from the updated full state. The hard gate is simple:
`exactkv_output_ids == full_output_ids`. Across twenty published experiments and thousands
of cells, **exactkv_failures == 0**.

**Why acceptance matters.** Exactness alone does not tell you whether compression is
*useful*. ExactKV reports **acceptance rate** — how often draft tokens survive verification
without correction — plus rejection counts, correction positions, and lossy-divergence
forensics. High acceptance means the compressed cache is a good draft source; low acceptance
means verification is doing more work, even though the final output remains exact.

**Why famous KV backends did not automatically win.** ExactKV evaluated restricted
adapters for TurboQuant, KIVI, and KVQuant-style simquant alongside built-in baselines.
All preserved exactness when wrapped. None automatically beat a simple **`int8` simulated
baseline** on acceptance. Among restricted backends, **KVQuant simquant** performed best;
**KIVI offline** was weakest. External paper results are **not** ExactKV results — only
cells run inside ExactKV's verifier count.

**How V10 hardened the claims.** Version 0.10.0 expanded from a 34-prompt legacy suite
to **128 versioned prompts** across seven categories, with per-category leaderboards,
draft/generation sensitivity (Experiment 013), and harder-category real-backend spot-checks
(Experiment 014). Findings such as "`long_context` is hardest" and "boundary-layer V policies
can beat uniform k8_v4_sim" became **category-qualified**, not single global numbers.

**How V11 hardened them further.** Experiments 015–016 showed the exactness gate and
compressor rankings largely **transfer to 1.5B and 3B** on the same V10 suites — with
shrinking margin for layer-aware policies at 3B. Experiment 017 showed a **metadata-only
serving sidecar** can observe cache lifecycle without owning the authoritative verifier,
while **direct vLLM/LMCache integration remains no-go**. Experiment 018 documented how
to pilot PyTorch CUDA allocations **without** conflating them with V5 workspace accounting
or claiming active GPU memory savings. Experiments 019–020 went deeper: mechanistic
**divergence autopsy** (logit margins, token types, layer-wise KV error) and an
**autopsy-guided repair-policy pilot** where category-adaptive and fallback-int8 selectors
improved acceptance on a shared prompt panel — still exact, still experiment-layer only.

---

## Required caveats (must accompany any future public post)

- **Not a speed or throughput story.** ExactKV does not report tokens/sec, latency, speedup, or runtime improvements.
- **Not active GPU memory savings.** V5 `total_kv_footprint_bytes` is conservative accounting; Exp 018 pilot is separate and not in the standard schema.
- **Not production serving.** No vLLM, LMCache, or PagedAttention integration; sidecar is observational.
- **Not final model accuracy improvement.** Exactness preserves greedy output; compression is evaluated as draft quality, not downstream task accuracy.
- **Not a universal benchmark.** V10/V11 suites are deliberate research panels.
- **Simulated compressors are not real packed-bit backends.** `_sim` policies use int8 containers.
- **Repair policies are pilot-only** — not enabled in core ExactKV by default.

---

## Suggested one-liner (if ever approved)

> ExactKV: compress the KV cache for drafting, verify every token against full precision, keep exact greedy output — and measure whether compression is actually useful.

---

## Related

- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md)
- [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md)
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) — D18
