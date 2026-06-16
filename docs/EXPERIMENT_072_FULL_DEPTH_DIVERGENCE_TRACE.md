# Experiment 072: Full-Depth Divergence Trace (Phase 16G)

**Status:** offline divergence trace — run `scripts/research/run_exp072_full_depth_divergence_trace.py` for report.

> This is an **offline divergence trace**, not model generation integration.  
> Streaming attention is **not wired into ExactKV generation**.  
> Top-k agreement is supplementary and is **not** a proof of exact generation preservation.  
> No CUDA kernel is implemented.  
> No Triton kernel is implemented.  
> No vLLM integration is implemented.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md`](EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md) · `exactkv/attention/hf_full_replay_probe.py`

---

## 1. Purpose

Phase 16G traces where streaming-vs-materialized divergence grows across the full 24-layer Qwen2.5 decoder stack, using per-layer checkpoints and two diagnostic replay modes.

---

## 2. Relation to Phase 16F

Phase 16F showed perfect top-1/top-k agreement but failed strict numeric tolerance at full depth. Phase 16G isolates whether divergence is local (per-layer attention) or cumulative (free-running hidden-state amplification).

---

## 3. Why full-depth divergence needed tracing

A single end-to-end error number cannot distinguish attention mismatch from MLP/residual amplification or tolerance policy mismatch. Layer-by-layer traces localize growth.

---

## 4. Free-running vs teacher-forced trace modes

| Mode | Behavior |
|---|---|
| `free_running` | Each path feeds its own hidden state to the next layer |
| `teacher_forced_layer_inputs` | Both paths receive the same materialized-chain layer input |

---

## 5. Per-layer trace points

After each layer, streaming vs materialized metrics at:

- `layer_input`
- `attn_context` (before `o_proj`)
- `attn_output` (after `o_proj`)
- `post_attention_hidden`
- `post_mlp_hidden`

---

## 6. Threshold crossing analysis

First layer where free-running `post_mlp_hidden` max abs error exceeds `1e-4`, `1e-3`, `1e-2`, `1e-1`.

---

## 7. Final logit/top-k agreement analysis

At final layer: norm + `lm_head`, compare logits. Top-1/top-5/top-10 overlap recorded as supplementary signal only.

---

## 8. Root cause classification

| Classification | Condition (heuristic) |
|---|---|
| `local_attention_mismatch` | Teacher-forced attn_context error large |
| `post_attention_amplification` | Attn small, post-attention residual large |
| `mlp_residual_amplification` | Attn small, post-MLP large |
| `free_running_accumulation` | Teacher-forced small, free-running grows |
| `tolerance_policy_issue` | Top-1 agrees, numeric tolerance fails |
| `unknown` | No clear pattern |

---

## 9. Results

```bash
python3 scripts/research/run_exp072_full_depth_divergence_trace.py
```

Report (gitignored): `reports/experiment_072_full_depth_divergence_trace.json`

Default: 2 prompts × 3 chunk sizes × 2 trace modes = **12 cells**.

**Local CPU run (`Qwen/Qwen2.5-0.5B`, float32):**

| Metric | Value |
|---|---|
| Phase 16F failure reproduced | **Yes** |
| Teacher-forced max attn_context error | **4.77×10⁻⁶** |
| Teacher-forced max post_mlp error | **2.67×10⁻⁵** |
| Free-running max post_mlp error | **0.501** |
| Root cause (all 6 free-running cells) | **`free_running_accumulation`** |
| First layer > 1e-4 (free-running) | layer **3–4** |
| First layer > 1e-1 (free-running) | layer **16–23** |
| Final logit max error | **0.134** |
| Free-running top-1 agreement | **6/6** |

---

## 10. What this proves

- Where divergence first exceeds numeric thresholds across the stack
- Whether local per-layer streaming attention matches materialized under teacher-forced inputs
- Whether free-running accumulation explains full-depth numeric divergence

---

## 11. What this does not prove

- Model output preservation or generation equivalence
- Production correctness of streaming attention
- Runtime integration into ExactKV generation
- Speed, throughput, latency, or measured GPU memory savings

---

## 12. Relation to ExactKV restored verification

Orthogonal: restored verification tracks greedy continuation from full KV; this trace diagnoses compressed-path divergence mechanics.

---

## 13. Relation to VeriCache parity

Does not reproduce VeriCache throughput, serving, or memory panels.

---

## 14. Next step

**Phase 16H (complete):** [`EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md`](EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md) — broader Qwen-family offline divergence panel.

**Phase 16I (proposed):** depth-aware tolerance policy documentation — still no default runtime integration.

---

## Claims boundary

| Allowed | Forbidden |
|---|---|
| Layer-by-layer divergence trace | Generation preservation claim |
| Root cause heuristics | Production correctness guarantee |
| Supplementary top-k agreement | Speedup / throughput / latency |
