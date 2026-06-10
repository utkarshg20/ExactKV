# ExactKV Project Status (v0.10.0)

**As of:** v0.10.0 (V10 complete). **Not public-launch final.**

ExactKV is a correctness-first, compressor-agnostic research platform for evaluating
lossy KV-cache compression under ExactKV's draft-verify-commit loop. Through V10 it
has published fourteen experiment reports (001–014, including 006A and 006C), seven
versioned V10 prompt suites (128 prompts), three restricted real-backend adapter
families (factory-only), and Experiments 012–014 validating suite expansion,
sensitivity/forensics, and harder-category real-backend spot-checks — all with
`exactkv_failures == 0` on every published sweep cell. The project is **ready to tag
v0.10.0** but **not ready for public v1.0.0 launch**; V11 must complete serving,
profiling, and multi-model hardening first.

---

## 1. One-paragraph status

V10 hardened ExactKV's evaluation story: broader prompt suites, per-category
leaderboards, draft/generation sensitivity, divergence forensics beyond proxy analysis,
and factory-only real-backend spot-checks on harder categories — without changing the
exactness gate, generation logic, verification logic, or report schemas. ExactKV's
claims about compressor behaviour are **more defensible** than at v0.9.0, but V10
suites are **not universal benchmarks** and public launch remains deferred until V11
and the v1.0.0 documentation package are complete.

---

## 2. Version timeline (v0.1.0 → v0.10.0)

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
| **v0.10.0** | **V10** | Evaluation-suite hardening; Exp 012–014; readiness assessment |

---

## 3. What ExactKV is

- A **verification and evaluation framework** for lossy KV-cache compression.
- Built on the VeriCache draft-then-verify idea: compressors draft on lossy KV;
  verification uses **full-precision KV**; output matches `generate_full_greedy`.
- Measures **exactness, acceptance, rejection, correction, divergence**, and
  **honest workspace-memory accounting**.
- Hugging Face–centric runtime; primary sweeps on `Qwen/Qwen2.5-0.5B`; 1.5B validated
  on legacy `core` (Exp 011), not yet on full V10 suites.

---

## 4. What ExactKV is not

- **Not** a production serving system (no vLLM/LMCache integration).
- **Not** a throughput or latency benchmark (no tokens/sec, speedup, or runtime claims).
- **Not** a packed-bit quantization library for all compressors (`_sim` = int8 containers).
- **Not** TurboQuant production, KIVI CUDA/Triton, or KVQuant deployment CUDA.
- **Not** a universal public benchmark (V10 suites are deliberate, bounded panels).
- **Not** public-launch final at v0.10.0.

---

## 5. Current best findings

| Finding | Source |
|---|---|
| Exactness gate on all published cells | Experiments 001–014 |
| `int8` strongest simple lossy baseline on V10 panels | Exp 012–014 |
| boundary4 > k8_v4_sim survives suite expansion; margin shrinks on harder prompts | Exp 012, 014 |
| `long_context` consistently hardest category | Exp 012, 014 |
| draft_len 8 lowers accept vs draft_len 2 | Exp 013 |
| KVQuant > TurboQuant > KIVI among restricted real backends; exact-safe | Exp 008–010, 014 |
| Divergence forensics: token-type + structured-output heuristics (no attention weights) | Exp 013 |
| Three external backends preserve exactness when wrapped | Exp 008–010, 014 |
| 1.5B exactness on legacy `core` | Exp 011 |

---

## 6. V10 suite expansion story

- **Phase 1:** Seven JSONL suites — `core_v2` + six category suites — **128 prompts** total.
- **Phase 2 (Exp 012):** 128 × 7 = **896 cells**; per-suite and per-category leaderboards;
  prompt-level win/loss; `exactkv_failures == 0`.
- Suites documented in [`V10_PROMPT_SUITES.md`](V10_PROMPT_SUITES.md); validated by
  `scripts/validate_v10_prompt_suites.py`.

---

## 7. Sensitivity/forensics story

- **Phase 3 (Exp 013):** 60 prompts × 4 compressors × 3×3 grid = **2160 cells**.
- `draft_len` {2,4,8} × `max_new_tokens` {16,32,64}; boundary4 margin widens at longer drafts.
- Token-type divergence counts and structured-output bracket/quote heuristics on code/tool cells.
- **No fabricated attention weights.**

---

## 8. Real-backend spot-check story

- **Phase 4 (Exp 014):** 40 harder-category prompts × 7 compressors = **280 merged cells**.
- KVQuant **0.634**, TurboQuant **0.309**, KIVI **0.019** on spot-check (vs higher on V9 `core`).
- Cross-panel merge from isolated environments; factory-only adapters.

---

## 9. Remaining weaknesses

- **0.5B-primary** on V10 Experiments 012–014; 1.5B not re-run on expanded suites.
- **Low-n** in some categories; spot-check is 40 prompts only.
- **No active GPU memory** measurements (D14).
- **No serving sidecar** evaluation (D13).
- **Heuristic forensics** only — no attention maps.
- **Production external backends** not integrated.

---

## 10. Whether v1.0.0 is ready

**No.** Per [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md) and
[`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) §23, v1.0.0 requires **V11**
(substance) plus final launch package (D17–D20).

---

## 11. V11 plan

| Focus | Deferred IDs |
|---|---|
| 1.5B+ on V10 suites | Multi-model matrix |
| Serving sidecar probe | D13 |
| Active GPU memory methodology | D14 |
| Optional attention logging (small subset) | D7 |
| Raw report bundle + launch narrative | D17–D18 |
| `PROJECT_STATUS` / `RELEASE_NOTES` v1.0.0 | D19–D20 |

See [`ROADMAP.md`](ROADMAP.md) §V11 and [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md).

---

## Related

- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md)
- [`V10_READINESS_ASSESSMENT.md`](V10_READINESS_ASSESSMENT.md)
- [`PROJECT_STATUS_V0.9.0.md`](PROJECT_STATUS_V0.9.0.md) — superseded for current status
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
