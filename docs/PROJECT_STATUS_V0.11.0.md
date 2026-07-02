# ExactKV Project Status (v0.11.0)

**As of:** v0.11.0 (V-release complete; legacy: V11).

ExactKV is a correctness-first, compressor-agnostic research platform for evaluating
lossy KV-cache compression under ExactKV's draft-verify-commit loop. Through V11 it
has published twenty experiment reports (001–020), seven versioned V10 prompt suites
(128 prompts), multi-model validation through **Qwen2.5-3B**, serving sidecar probe
artifacts, GPU memory methodology, divergence autopsy, and a repair-policy pilot —
all with `exactkv_failures == 0` on every published sweep cell. The project is **ready
to tag v0.11.0** but **not ready for public v1.0.0 launch** without final narrative
review and v1.0.0 documentation.

---

## 1. One-paragraph status

V11 closed the remaining scale, serving-context, profiling, forensics, and
launch-documentation gaps before a defensible public release. Experiments 015–016
validated 1.5B and 3B on full V10 suites; Experiment 017 reaffirmed serving sidecar
feasibility and direct vLLM/LMCache no-go; Experiment 018 documented GPU memory pilot
methodology without changing the standard schema; Experiments 019–020 added mechanistic
divergence autopsy and an autopsy-guided repair-policy pilot. ExactKV's claims are
**more defensible** than at v0.10.0, but V10/V11 suites are **not universal benchmarks**
and public launch remains deferred until v1.0.0 gates are explicitly cleared.

---

## 2. Version timeline (v0.1.0 → v0.11.0)

| Version | Tag | Theme |
|---|---|---|
| V1 | — | Draft-verify-commit prototype; exactness gate |
| v0.2.0 | V2 | Compressor registry; JSON/CSV reports |
| v0.3.0 | V3 | Named prompt suites; Markdown reports |
| v0.4.0 | V4 | Asymmetric K/V compressors; Experiment 003 |
| v0.5.0 | V5 | Workspace-aware memory accounting; Experiment 004 |
| v0.6.0 | V6 | `BackendAdapter`; restricted kvpress; Experiment 005 |
| v0.7.0 | V7 | Layer-aware V policies; Experiments 006 / 006C |
| v0.8.0 | V8 | Serving-context harness; Experiment 007 |
| v0.9.0 | V9 | Real backend gauntlet (Exp 008–010); 1.5B validation (Exp 011) |
| v0.10.0 | V10 | Evaluation-suite hardening; Exp 012–014 |
| **v0.11.0** | **V11** | Launch hardening; Exp 015–020; readiness package |

---

## 3. What ExactKV is

- A **verification and evaluation framework** for lossy KV-cache compression.
- Compressors draft on **lossy KV**; verification uses **full-precision KV**; output matches `generate_full_greedy`.
- Measures **exactness, acceptance, rejection, correction, divergence**, and honest workspace-memory accounting.
- Hugging Face–centric runtime; validated on **Qwen2.5-0.5B, 1.5B, and 3B** on V10 suites.

---

## 4. What ExactKV is not

- **Not** a production serving system (no vLLM/LMCache integration).
- **Not** a throughput or latency benchmark.
- **Not** a packed-bit quantization library for all compressors (`_sim` = int8 containers).
- **Not** TurboQuant production, KIVI CUDA/Triton, or KVQuant deployment CUDA.
- **Not** a universal public benchmark.
- **Not** public-launch final at v0.11.0.

---

## 5. Current strongest findings

| Finding | Source |
|---|---|
| Exactness gate on all published cells | Exp 001–020 |
| `int8` strongest simple baseline through 3B | Exp 012–016, 019–020 |
| boundary4 > k8_v4_sim in many panels; margin scale-sensitive | Exp 012–016 |
| `long_context` hardest category | Exp 012–014, 019–020 |
| KVQuant > TurboQuant > KIVI among restricted backends | Exp 008–010, 014 |
| 3B exactness on full V10 suites | Exp 016 |
| Sidecar probe pass; vLLM/LMCache no-go | Exp 017 |
| Autopsy-guided policies improve acceptance (pilot) | Exp 020 |

---

## 6. Scale validation story

| Model | V10 full suites | Cells | Failures |
|---|---|---:|---:|
| Qwen2.5-0.5B | Exp 012 | 896 | 0 |
| Qwen2.5-1.5B | Exp 015 | 896 | 0 |
| Qwen2.5-3B | Exp 016 | 896 | 0 |

Transfer findings from 0.5B largely hold at 1.5B; boundary4 margin shrinks at 3B.

---

## 7. Suite-hardening story

V10 (Exp 012–014): 128 prompts, per-category leaderboards, sensitivity grid, hard-category real-backend spot-checks. Carried forward unchanged in V11 validation runs.

---

## 8. Real-backend story

Restricted adapters (KVQuant, TurboQuant, KIVI, kvpress) preserve exactness when wrapped; KVQuant simquant leads on acceptance among tested restricted backends. All remain **factory-only**, not default registry.

---

## 9. Serving sidecar story

Experiment 017: metadata-only sidecar probe passes; authoritative full-KV verifier remains separate. Direct vLLM/LMCache integration **no-go** reaffirmed.

---

## 10. GPU memory methodology story

Experiment 018: PyTorch CUDA allocation pilot documented; `pilot_success`. V5 `total_kv_footprint_bytes` remains the launch memory metric. No active GPU memory savings claims.

---

## 11. Divergence autopsy story

Experiment 019: logit margins, top-k overlap, token-type and layer-wise KV error on 400 cells. Attention deferred; repair hypotheses proposed.

---

## 12. Repair-policy pilot story

Experiment 020: six policy variants on shared 25-prompt panel; `fallback_int8` and `category_adaptive` beat baselines on acceptance while preserving exactness. Policies **not** in core ExactKV.

---

## 13. Remaining limitations

- No production serving, batching, or sampling.
- Attention / per-head forensics deferred.
- Repair policies pilot-scale only.
- Public launch narrative draft — not final post.
- Physical report bundle optional until release attach.

---

## 14. v1.0.0 readiness decision

**Ready for `v0.11.0` tag.** **Not ready for public v1.0.0 launch** without v1.0.0 status/release notes, narrative approval, and optional full pytest + artifact bundle.

See [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) §20.

---

## Related

- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md)
- [`RELEASE_NOTES_V0.11.0.md`](RELEASE_NOTES_V0.11.0.md)
- [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md)
- [`PROJECT_STATUS_V0.10.0.md`](PROJECT_STATUS_V0.10.0.md) — superseded for current status
