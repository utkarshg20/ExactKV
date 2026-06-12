# Experiment 030b: Batched Span Verification GPU/fp16 Parity Investigation

_V13 Phase 3b — span parity investigation only._

> This is a **span parity investigation**, not a production benchmark.
> This does **not** claim speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, or production serving.
> ExactKV does **not** claim model accuracy improvement.

---

## 1. Purpose

Diagnose why batched span verification on fp16 GPU disagrees with sequential single-step verification (Exp 030 fallback), and test safe fixes.

## 2. Why this follows Exp 030

Exp 030 passed exactness via fp16 sequential fallback but span wall-clock matched sequential (~21.2 tok/s). Batched span verify must be restored only after token-level parity is proven.

## 3. Environment

| Item | Value |
|---|---|
| GPU | `NVIDIA RTX A5000` |
| torch | `2.8.0+cu128` |
| transformers | `4.43.4` |
| CUDA | `12.8` |

## 4. Reproduced blocker

| Item | Value |
|---|---|
| Parity pass (variant A fp16 default) | **False** |
| First mismatch index | `7` |
| Sequential verifier (tail) | `[6896, 13, 576]` |
| Batched verifier (tail) | `[6896, 13, 3555]` |

## 5. Prompt/cell details

| prompt_id | `lc_003` |
| compressor | `k8_v4_sim` |
| draft_len | 8 |
| failing round | 2 |

## 6. Variants tested

| Variant | parity | first mismatch | forward mode |
|---|---|---|---|
| `A_cuda_fp16_default` | False | `7` | `default` |
| `A2_cuda_fp16_all_teacher` | False | `7` | `default` |
| `A3_cuda_fp16_cache_position` | False | `7` | `cache_position` |
| `A4_cuda_fp16_cache_position_mask` | False | `7` | `cache_position_and_mask` |
| `A5_cuda_fp16_position_ids` | False | `7` | `position_ids` |
| `A6_engine_batched_after_fix` | True | `None` | `engine` |
| `B_cuda_fp32_default` | True | `None` | `default` |
| `C_cuda_fp16_eager` | True | `None` | `default` |
| `C2_cuda_fp16_eager_cache_position` | True | `None` | `cache_position_and_mask` |
| `D_cuda_fp32_eager` | True | `None` | `default` |
| `E_cpu_fp32_default` | True | `None` | `default` |

## 7. Batched vs sequential token comparison

Draft at round 2: `[2550, 1969, 2432, 92382, 44378, 6896, 13, 576]`

## 8. Logit difference / argmax flip findings

| Position | seq argmax | batched argmax | argmax match | max |logit diff| |
|---:|---:|---:|---|---:|
| 7 | 576 | 3555 | False | 0.0273 |

## 9. Root cause analysis

fp16 SDPA batched forwards tie-break argmax differently than sequential single-step forwards when top logits are nearly equal (lc_003 position 7: tokens 576 vs 3555, max |logit diff| ≈ 0.03). **Fix:** math-only SDPA context + explicit `cache_position` / `attention_mask` in `VerificationEngine._verify_span_batched` (Exp 030b).

## 10. Fix attempted, if any

Applied in `VerificationEngine._verify_span_batched` (Exp 030b):

- Math-only SDPA context on fp16 CUDA (`sdp_kernel` flash/mem-efficient disabled).
- Explicit `cache_position` and `attention_mask` for DynamicCache multi-token decode.
- Parity guard: if batched ≠ sequential, `verify_span` still returns sequential.

## 11. Fix result, if any

**Pass** — variant `A6_engine_batched_after_fix` restores batched verifier token parity with sequential on the lc_003 round-2 cell. fp16 eager attention also passes (variants C/C2), confirming SDPA tie-breaking as root cause.

## 12. Remaining fallback behavior

Blanket fp16 sequential fallback **removed**. `verify_span` uses batched path when parity holds; sequential parity guard remains if batched ≠ sequential.

## 13. Exactness result

End-to-end span ≡ full greedy on fp16 GPU lc_003: **True** (pytest `test_span_gpu_parity.py` on RunPod A5000).

## 14. Whether full Exp 030 timing should be rerun

**Yes** — only after batched parity is restored and enabled in `verify_span`.

## 15. What this proves

- Token-level batched vs sequential verifier extraction can be compared per HF forward variant.
- Whether argmax flips are due to numeric logit divergence vs indexing.

## 16. What this does not prove

- General GPU speedup from span verification.
- Production serving readiness.
- Active GPU memory savings.

## 17. Next steps

- Apply winning forward kwargs in `VerificationEngine._verify_span_batched`.
- Remove blanket fp16 fallback; keep sequential parity guard.
- Rerun full Exp 030 timing on GPU.
- Phase 4 (Exp 031) may proceed in parallel.

Reproduce:

```bash
TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_030b_span_parity_investigation.py --device cuda
```

