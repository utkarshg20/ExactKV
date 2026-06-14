# Experiment 045: SpectralQuant Restricted Adapter Panel

_Phase 10G — 12-prompt factory-only adapter panel on Qwen2.5-0.5B._

> **SpectralQuant remains a factory-only experimental adapter.**
> **It is not a default ExactKV compressor.**
> **The adapter materializes decompressed K/V for generation.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**
> Results are scoped to this small restricted panel only.

---

## 1. Purpose

Determine whether the Exp 044 `spectralquant_experimental` adapter deserves a **caveated RESTRICTED BACKEND** leaderboard row beyond smoke-only / tensor-probe status.

---

## 2. Setup

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | CPU, float32 |
| External repo | `SPECTRALQUANT_REPO_PATH=~/spectralquant` |
| Adapter | `create_spectralquant_experimental_adapter()` |
| `draft_len` | 4 |
| `max_new_tokens` | 32 |
| Verification | Default sequential full-KV (unchanged) |

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp045_spectralquant_restricted_panel.py
```

Report: `reports/experiment_045_spectralquant_restricted_panel.json` (gitignored).

---

## 3. Calibration details

| Field | Value |
|---|---|
| Calibration prompts | **6** |
| `max_tokens_per_layer` | 256 |
| `avg_bits` | 4.0 |
| `qjl_projections` | 32 |
| API | `EigenspectralCalibrator.calibrate` + eigenspectrum `fit_quantizers` |

Not paper-scale calibration. Limitation disclosed in all public wording.

---

## 4. Adapter design summary

- **Class:** `SpectralQuantExperimentalAdapter` (`exactkv/external/spectralquant_adapter.py`)
- **Path:** `BackendAdapter` tensor compress → store compressed payloads → materialize dequant K/V for draft
- **Materializing:** Yes — compresses then **immediately materialises** full dequant tensors for draft forward
- **Registry:** Factory-only — **not** in `get_compressor()` default registry
- **`supports_real_bytes_claim`:** false

---

## 5. Panel composition

**12 prompts** (all planned prompts ran; no runtime reduction):

| Category | Prompt IDs |
|---|---|
| natural_language | cv2_nat_001, cv2_nat_002 |
| retrieval_copy | rc_001, rc_002 |
| long_context | lc_001, lc_002 |
| tool_schema (JSON) | tj_001, tj_002 |
| code_structured | cs_py_001, cs_py_002 |
| reasoning_math | rm_001, rm_002 |

---

## 6. Results

| Metric | Value |
|---|---|
| `prompt_count` | **12** |
| `status` | **pass** |
| `token_exact_match_count` | **12 / 12** |
| Draft divergence prompts | **11 / 12** (verifier corrected) |
| Output divergence | **0** |

---

## 7. exactkv_failures

**0** — all 12 prompts preserved full-greedy final output under ExactKV verification.

---

## 8. Acceptance summary

| Stat | Value |
|---|---|
| Mean acceptance | **0.481** |
| Median acceptance | **0.421** |
| Min acceptance | **0.109** |

Small-panel adapter acceptance only — **not** full-panel compressor acceptance.

Per-round accepted-prefix lengths vary widely (many 0-length rounds when drafts reject early).

---

## 9. Divergence examples

Draft diverged from verifier on **11/12** prompts; ExactKV corrected and final output matched full KV.

Example (cv2_nat_001):
- Acceptance rate: **0.266**
- Accepted-prefix lengths included 0, 1, 4, 2, 3 across rounds
- `exactkv_failures`: 0

No final output divergence (`first_divergence_idx`: null on exact-match prompts).

---

## 10. Reconstruction / error caveats

Tensor round-trip on panel prompt (layers 0, 12, 23):

| Layer | Key max abs error | Value max abs error |
|---:|---:|---:|
| 0 | **~39.0** | ~0.75 |
| 12 | ~1.7 | ~0.75 |
| 23 | (see report) | (see report) |

**Large key reconstruction error on layer 0** — lossy quant, not lossless. All adapter layers use compress/decompress; error varies by layer.

---

## 11. What this proves

- Factory-only SpectralQuant adapter runs a **12-prompt restricted panel** with **`exactkv_failures=0`**.
- Lossy draft divergence occurs frequently but ExactKV verification preserves exact greedy output.
- Adapter is **materializing** — not an active-memory-savings path.

---

## 12. What this does not prove

- Full V10 / 128-prompt benchmark coverage.
- Paper-scale SpectralQuant calibration or CUDA kernel path.
- Speed, active GPU memory savings, or production serving.
- Model accuracy improvement.
- Ranking parity with full-panel INT8/K8V4 compressors.

---

## 13. Leaderboard classification decision

**Promoted to RESTRICTED BACKEND** (Exp 045):

- `exactkv_failures == 0` ✓
- `prompt_count >= 8` ✓ (12 prompts)

Row: **SpectralQuant experimental adapter** · mean acceptance **0.481** · Exp 045 · badges: RESTRICTED, FACTORY-ONLY ADAPTER, SMALL PANEL, MATERIALIZING, NOT DEFAULT, NO SPEED/MEMORY CLAIM.

Removed prior SMOKE ONLY tensor-probe row (Exp 042) — superseded by adapter panel.

---

## 14. Limitations

- 6-prompt calibration; 12-prompt panel; CPU float32 only in this run.
- Materializing adapter — no active memory savings.
- High per-layer key reconstruction error possible.
- External SpectralQuant README/paper metrics are **not** ExactKV results.

---

## 15. Recommendation

**`promote_restricted_backend`** — keep as caveated RESTRICTED BACKEND row; optional future: expand panel or improve calibration before any full-panel claim. Do **not** add to default registry.

---

## Run commands

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp045_spectralquant_restricted_panel.py
pytest tests/test_spectralquant_restricted_panel.py -q
python3 scripts/exactkv_leaderboard.py --md --html
```
