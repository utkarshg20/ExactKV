# Experiment 073: Qwen-Family Divergence Panel (Phase 16H)

**Status:** offline Qwen-family divergence panel — run `scripts/research/run_exp073_qwen_family_divergence_panel.py` for report.

> This is an **offline Qwen-family divergence panel**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Top-k agreement is supplementary and is **not** a proof of exact generation preservation.  
> Unsupported or blocked models are **reported explicitly**.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md`](EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md) · `exactkv/attention/hf_full_replay_probe.py`

---

## 1. Purpose

Phase 16H reuses the Phase 16G full-depth divergence trace across a small Qwen-family model panel to test whether `free_running_accumulation` is model-specific or general.

---

## 2. Relation to Phase 16G

Phase 16G traced one checkpoint (`Qwen/Qwen2.5-0.5B`) and classified root cause as cumulative free-running hidden-state amplification with tiny teacher-forced local errors. Phase 16H asks whether the same pattern holds on nearby Qwen2.5 base/instruct (and optional larger) checkpoints.

---

## 3. Model panel

| Tier | Model IDs |
|---|---|
| Default | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct` |
| Optional (`--include-optional-models`) | `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-1.5B-Instruct` |

Models that fail load, memory, download, or QKV extraction are recorded as blocked entries — the panel does not require all models to succeed.

---

## 4. Teacher-forced vs free-running modes

Same as Phase 16G:

- **`teacher_forced_layer_inputs`** — both paths share identical layer inputs; isolates local attention mismatch.
- **`free_running`** — each path feeds its own hidden state; measures accumulated divergence.

Default sweep: token lengths `32,64`; chunk sizes `16,64`; 2 prompts; CPU `float32`; accumulator `float32`.

---

## 5. Model-level classification

| Classification | Criteria |
|---|---|
| `free_running_accumulation_confirmed` | Tiny teacher-forced local errors + growing free-running errors |
| `local_attention_mismatch_detected` | Teacher-forced attention/context errors spike |
| `parity_failure` | Full manual replay vs HF parity fails |
| `unsupported_architecture` | QKV extraction blocked |
| `model_load_blocked` | Model load failed |
| `unknown` | Does not match above heuristics |

---

## 6. Results by model

Run locally:

```bash
python3 scripts/research/run_exp073_qwen_family_divergence_panel.py
```

Report: `reports/experiment_073_qwen_family_divergence_panel.json` (gitignored).

**Local CPU run (default panel):**

| Model | Classification | TF max attn | FR max post-MLP | Top-1 agree | Parity |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | `free_running_accumulation_confirmed` | ~4.3×10⁻⁶ | **0.501** | **4/4** | pass |
| `Qwen/Qwen2.5-0.5B-Instruct` | `free_running_accumulation_confirmed` | ~4.3×10⁻⁶ | **0.553** | **4/4** | pass |

Both models reproduce the Phase 16G pattern: tiny teacher-forced local errors, growing free-running accumulation, perfect supplementary top-k agreement.

---

## 7. Top-k agreement summary

Per-model free-running top-1/top-5/top-10 agreement is recorded as **supplementary signal only**. Perfect top-k overlap does not imply numeric equivalence or exact generation preservation.

---

## 8. Memory accounting summary

Per-cell theoretical streaming working-set reduction vs materialized compressed path (`best_theoretical_streaming_reduction`). **Theoretical only** — not measured active GPU memory or production savings.

---

## 9. Blockers

Blocked models and cells preserve `blockers` lists (load failure, unsupported extraction, parity failure, trace exception). Blockers are never hidden.

---

## 10. What this proves

- Whether Phase 16G `free_running_accumulation` pattern generalizes across nearby Qwen-family checkpoints.
- Whether teacher-forced local errors stay tiny across the panel.
- Whether final top-k agreement remains high despite numeric divergence (supplementary).

---

## 11. What this does not prove

- Exact generation preservation or model-output preservation.
- Production correctness of streaming attention in ExactKV runtime.
- Speed, throughput, latency, or measured GPU memory savings.
- vLLM/LMCache/serving integration.
- VeriCache throughput or serving reproduction.

---

## 12. Relation to ExactKV restored verification

Independent offline research probe. Does not change ExactKV default generation or `ExactKVGenerator` wiring.

---

## 13. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16I (complete):** [`EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md`](EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md) — formalized offline tolerance/interpretation policy.

**Phase 16J (proposed):** generation-shadow wiring review — only with explicit approval; still no default runtime integration.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Cross-model divergence panel | Generation preservation claim |
| Model-level classification heuristics | Production correctness guarantee |
| Supplementary top-k agreement | Speedup / throughput / latency |
| Theoretical memory accounting | Measured active GPU memory savings |
