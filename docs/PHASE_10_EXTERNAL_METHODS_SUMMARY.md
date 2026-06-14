# Phase 10 External Methods Summary (V13)

**Status:** Consolidation checkpoint — **research-demo-ready, not production/serving-ready.**  
**Date:** 2026-06-09 · Phase **10H**

> Shard and SpectralQuant external-method probes are **complete for this phase**.  
> No new experiments, adapters, or evaluations were added in 10H — documentation only.

Companion: [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md) · [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) · [`leaderboard.md`](leaderboard.md)

---

## 1. Core ExactKV results (integrated compressors & verification)

These are **not** external-method probes. They use built-in or full-panel integrated compressors on documented V10/V13 panels.

| Result | Experiment / artifact | Scope | exactkv_failures |
|---|---|---|---|
| Full-panel Qwen suites | Exp 012, 015, 016 | 128-prompt V10; built-in INT8 / K8V4 sim | **0** |
| Span verification grid | Exp 029 | 600 cells; span ≡ sequential exactness | **0** |
| Llama-3.1-8B small suite | Exp 033 | 12-prompt × 4 compressors (48 cells) | **0** |
| Pharmacy terminal demo (primary) | Exp 034b | `pharm_001` — lossy `drop` rejected, `pickup` committed | **0** on trace |
| LongBench-style drift demo (secondary) | Exp 037 | `lb_md_001` — outcome heuristic green; token path drifted | **0** on trace |
| Repair policies | Exp 025 | Full-suite adaptive selectors (separate tier) | **0** |

**Core claim (scoped):** ExactKV detects and repairs draft-token drift relative to full-KV greedy decoding **on tested panels** — final output matches full greedy when `exactkv_failures == 0`.

---

## 2. External restricted backends

### Shard — external drafter (Llama-only)

| Phase | Experiment | Panel | Key result | exactkv_failures |
|---|---|---|---|---|
| Feasibility | Exp 038 | 4 prompts | `restricted_go` | **0** |
| Stress | Exp 039 | 32 × 64tok | 6/32 draft divergences (18.75%) | **0** |
| Ablation | Exp 040 | 32 × 5 settings | max single-knob 31.25% (length_128tok) | **0** all settings |
| Combined stress | Exp 041 | 32 × stream_bits=4 + 128tok | **18/32 divergences (56.25%)** | **0** |

- **Leaderboard:** RESTRICTED BACKEND · external-drafter probe · Exp 039–041  
- **Not** default registry · **Not** full-panel compressor ranking  
- Shard README throughput/memory = **external results**, not ExactKV results  
- **Recommendation:** `stop_shard_bounded_probe_complete`

### SpectralQuant — factory-only materializing adapter

| Phase | Experiment | Scope | Key result | exactkv_failures |
|---|---|---|---|---|
| API / tensor probe | Exp 042 | Synthetic K/V | `tensor_smoke_only` | N/A |
| Real KV tensor smoke | Exp 043 | Qwen 0.5B prefill K/V | round-trip pass; layer-0 key err ~39 | N/A (not generation) |
| Adapter smoke | Exp 044 | 4 prompts | mean accept 0.629 | **0** |
| Restricted panel | Exp 045 | **12 prompts** | mean accept **0.481**; 11/12 draft divergences | **0** |

- **Leaderboard:** RESTRICTED BACKEND · SpectralQuant experimental adapter · Exp 045  
- **Materializing adapter:** compresses K/V, then materialises dequant tensors for draft — **no active memory savings**  
- **Calibration:** 6-prompt eigenspectral (panel run); not paper-scale  
- **Not** default registry · small-panel acceptance only  
- External SpectralQuant paper/README = **not** ExactKV results

### Other restricted / smoke backends (unchanged)

| Method | Tier | Experiment | Notes |
|---|---|---|---|
| TurboQuant Python | RESTRICTED | Exp 008, 014 | Factory-only; not production llama.cpp |
| KIVI offline | RESTRICTED | Exp 009, 014 | Offline simulate path |
| KVQuant sim | RESTRICTED | Exp 010, 014, 023 | simquant adapter |
| SnapKV experimental | SMOKE ONLY | Exp 032b | 8-cell smoke; not full-suite |

---

## 3. What the project can claim

- ExactKV detects and repairs draft-token drift relative to full-KV greedy decoding **on tested panels**.
- **Shard** and **SpectralQuant** now have caveated **restricted external-method results** under ExactKV verification.
- **SpectralQuant adapter** is **materializing** and **factory-only** (`spectralquant_experimental`).
- **Shard** remains **external-drafter only** (Mode B; Llama-3.1-8B probe panels).
- **`exactkv_failures == 0`** on all listed Shard runs (038–041) and SpectralQuant adapter runs (044–045).
- Tiered leaderboard separates full-panel compressors from restricted backends and smoke-only adapters.
- Primary public demo: pharmacy terminal crash test. Secondary: LongBench-style drift demo.

---

## 4. What the project cannot claim

- No **speedup**, throughput, latency, or tokens/sec improvement.
- No **active GPU memory savings** or VRAM reduction (Exp 031; materializing adapters).
- No **production serving**, vLLM integration, or LMCache integration (Exp 017 no-go).
- No **full compressor leaderboard comparability** for restricted rows (apples-to-oranges).
- No **model accuracy improvement** from compression.
- No **default registry** entry for Shard or SpectralQuant.
- No **full benchmark coverage** for Shard (32-prompt Llama probe) or SpectralQuant (12-prompt Qwen panel).
- No **external paper/README** numbers as ExactKV results.

---

## 5. Launch readiness status

| Label | Applies? |
|---|---|
| Launch-ready | **No** |
| Prelaunch-ready | **Partial** — infra + audits exist; 9C validation incomplete |
| **Research-demo-ready** | **Yes** |
| Still blocked (v1.0 / public launch) | **Yes** |

**Recommended wording:** **Research-demo-ready, not production/serving-ready.**

Public launch and **v1.0.0 remain not approved** ([`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md)).

---

## 6. Leaderboard tiering (current)

| Tier | Examples |
|---|---|
| **FULL PANEL** | INT8, K8/V4 sim on Qwen V10 / Llama small suite |
| **REPAIR POLICY** | Exp 025 adaptive selectors |
| **RESTRICTED BACKEND** | TurboQuant, KIVI, KVQuant, **Shard external-drafter**, **SpectralQuant experimental adapter** |
| **SMOKE ONLY** | SnapKV experimental (8 cells) |
| **FUTURE CANDIDATE** | (empty) |

Regenerate: `python3 scripts/exactkv_leaderboard.py --md --html`

---

## 7. Strongest public-facing results

1. **Pharmacy terminal demo** — clearest “compressed KV lied; ExactKV corrected” story.  
2. **Tiered crash-test leaderboard** — honest separation of full-panel vs restricted vs smoke.  
3. **Exactness wall / public visual package** (Exp 036) — `exactkv_failures == 0` on cited panels.  
4. **Shard + SpectralQuant restricted rows** — caveated external-method coverage without overclaiming integration.

---

## 8. Strongest technical results

1. **Exp 029** — 600-cell span verification grid; exactness preserved.  
2. **Exp 041** — Shard combined stress exposes **56.25% draft divergence** with **0 ExactKV failures** (verifier holds).  
3. **Exp 045** — SpectralQuant 12-prompt panel: **0 failures**, mean acceptance 0.481, frequent draft divergence corrected.  
4. **Exp 043** — Real model K/V tensor round-trip validates SpectralQuant engine on HF cache tensors.

---

## 9. Recommended next research direction

**Public demo polish + clean release prep** (Phase 9C validation path):

- Clean-clone repro ([`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md))  
- Record pharmacy terminal demo  
- Optional `v0.13.0-rc` research preview tag — **not v1.0.0**

**Not recommended next (without explicit approval):** broader full-panel SpectralQuant ranking, production serving claims, or default-registry integration.

Alternatives deferred: broader model/prompt validation panel, runtime architecture design, calibration quality improvement.

---

## 10. Experiment index (external methods)

| Exp | Doc |
|---|---|
| 038–041 | Shard probe series — [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) |
| 042–045 | SpectralQuant probe → restricted panel — [`EXPERIMENT_042_SPECTRALQUANT_PROBE.md`](EXPERIMENT_042_SPECTRALQUANT_PROBE.md) through [`EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md`](EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md) |

---

## 11. Phase map (10A–10H)

| Phase | Deliverable | Status |
|---|---|---|
| 10A | LongBench-style secondary demo | ✅ |
| 10B | Shard probe 038–041 | ✅ complete |
| 10C | Shard leaderboard integration | ✅ |
| 10D | SpectralQuant probe (042) | ✅ |
| 10E | SpectralQuant smoke leaderboard (superseded by 10G) | ✅ |
| 10F | Real KV + adapter smoke (043–044) | ✅ |
| 10G | Restricted adapter panel (045) | ✅ |
| **10H** | **This consolidation doc** | ✅ |
