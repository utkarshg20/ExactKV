# Experiment 039: Shard External-Drafter Stress Panel

_Restricted stress panel — not Shard integration, not a default ExactKV compressor._

> **Shard remains an external-drafter probe, not an integrated ExactKV compressor.**
> **External Shard README claims are not ExactKV results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**
> **Results are scoped to the tested panel only.**

Artifacts (gitignored): `reports/experiment_039_shard_external_stress_panel.json`

---

## 1. Purpose

Characterize whether **Shard draft drift** appears under longer/harder prompts and longer greedy generation after Exp 038 established `restricted_go` on a 4-prompt × 16-token feasibility panel.

Divergence between Shard draft tokens and full-KV greedy verifier tokens is **expected and useful** — it is not a probe failure. `exactkv_failures` counts only cases where ExactKV-verified committed output does not match full-KV greedy.

## 2. Setup

| Item | Value |
| --- | --- |
| Platform | RunPod proxy SSH (`5vh87lgjxtyeou-64411443@ssh.runpod.io`) |
| GPU | NVIDIA L40S (46 GB) |
| ExactKV path | `/workspace/ExactKV` |
| Shard path | `/root/shard` (external clone; not vendored) |
| `SHARD_REPO_PATH` | `/root/shard` |
| `HF_TOKEN` | set on pod (environment only) |
| Script | `scripts/research/run_exp039_shard_stress_panel.py` |
| Run date | 2026-06-13 |

```bash
export SHARD_REPO_PATH=/root/shard
export HF_TOKEN=...   # from environment only
python3 scripts/research/run_exp039_shard_stress_panel.py --try-run \
  --device cuda --dtype float16 --max-new-tokens 64 --per-category 4
```

## 3. Model

`meta-llama/Llama-3.1-8B-Instruct` — gated; loaded with HF token + Meta license.

## 4. Prompt categories

Eight categories × 4 prompts = **32 prompts** (deterministic panel):

| Category | Example IDs |
| --- | --- |
| structured JSON | `struct_json_001` … `struct_json_004` |
| retrieval-copy | `rc_001` … `rc_004` |
| long-context summary | `lc_001` … `lc_004` |
| code completion | `code_py_001` … `code_py_004` |
| LongBench-style QA | `lb_sp_001` … `lb_sp_004` |
| tool-call JSON | `tj_001` … `tj_004` |
| instruction constraints | `ic_001` … `ic_004` |
| multilingual | `ml_de_001` … (sorted multilingual suite) |

## 5. Shard settings used

Documented knobs from `shard.Cache` / `Cache.from_model` (not invented):

| Setting | Value | Notes |
| --- | --- | --- |
| `streaming` | `true` | decode streaming quantizer enabled |
| `stream_bits` | `8` | Shard documents 8-bit streaming as lossless |
| `stream_qjl` | `false` | default |
| `k_target_cr` | `16.0` | `Cache.from_model` prefill compression target |

Prefill PCA+VQ uses Shard defaults (`sink_tokens=4`, `residual_window=64`, etc.).

## 6. Tokenizer alignment

| Check | Result |
| --- | --- |
| BOS harness | `prompt_ids_comparable()` (probe harness only) |
| `tokenizer_alignment_pass` | **true** (32/32 prompts) |
| `blocked_prompt_count` | **0** |

## 7. Results summary

| Field | Value |
| --- | --- |
| `panel_status` | **`pass`** |
| `prompt_count` | **32** |
| `max_new_tokens` | **64** |
| `draft_len` | **4** |
| `divergence_count` | **6** |
| `exactkv_failures` | **0** |
| `no_divergence_observed` | **false** |
| **Recommendation** | **`restricted_go_with_divergence`** |

**26/32 prompts** showed full 64-token greedy agreement between Shard draft path and HF full-KV verifier. **6 prompts** diverged; ExactKV verification kept committed output aligned with full-KV greedy on all aligned prompts.

## 8. Accepted-prefix distribution

| Stat | Value |
| --- | --- |
| min | 14 |
| max | 64 |
| mean | 58.22 |
| histogram | `{14: 2, 24: 1, 42: 1, 45: 1, 60: 1, 64: 26}` |

Most prompts (26) accepted the full 64-token prefix before any divergence.

## 9. First-divergence examples

| prompt_id | category | first_div | kind | draft tok | verifier tok |
| --- | --- | --- | --- | --- | --- |
| `struct_json_004` | structured_json | 60 | semantic_or_token_mismatch | `sku` | `qty` |
| `lc_001` | long_context_summary | 45 | formatting_or_punctuation | ` output` | ` exactly` |
| `lc_002` | long_context_summary | 14 | formatting_or_punctuation | ` any` | ` loss` |
| `lc_003` | long_context_summary | 24 | formatting_or_punctuation | ` drafting` | ` draft` |
| `lc_004` | long_context_summary | 14 | formatting_or_punctuation | ` the` | ` any` |
| `lb_sp_004` | longbench_style_qa | 42 | formatting_or_punctuation | ` not` | ` the` |

Divergence clustered in **long-context summary** (4/6) and one each in structured JSON continuation and LongBench-style QA.

## 10. exactkv_failures

**0** — verified committed output matched full-KV greedy on every aligned prompt, including the six with draft divergence.

## 11. What this proves

- A bounded 32-prompt × 64-token stress panel **does** expose Shard draft divergence vs full-KV greedy on Llama-3.1-8B.
- ExactKV verification can act as a **crash-test harness** for external Shard drafts: divergence observed, yet `exactkv_failures=0`.
- Long-context and structured-continuation prompts are more drift-prone in this panel than retrieval-copy / tool-json / code subsets.

## 12. What this does not prove

- Shard production integration or default-registry readiness.
- Speedup, VRAM savings, serving throughput, or model accuracy improvement.
- Behavior under stronger compression (e.g. lower `stream_bits`) or `max_new_tokens=128`.
- Generalization beyond this panel, model, or Shard settings.

## 13. Limitations

- Single model (Llama-3.1-8B-Instruct).
- Default `stream_bits=8` (Shard-documented lossless streaming) may understate drift vs more aggressive compression.
- Divergence classification is coarse heuristic (formatting vs semantic), not human evaluation.
- Panel runtime ~5.5 minutes on L40S for 32×64 tokens.

## 14. Next steps

1. Optional follow-up: documented lower `stream_bits` (e.g. 4 or 6) or `max_new_tokens=128`.
2. Expand panel or add harder long-context needles if more drift signal needed.
3. Still **no** default registry entry — Mode B external drafter only.

## Related

- [`EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md`](EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md)
- [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md)

Regenerate:

```bash
export SHARD_REPO_PATH=/root/shard
python3 scripts/research/run_exp039_shard_stress_panel.py              # blocked (planned count)
python3 scripts/research/run_exp039_shard_stress_panel.py --try-run    # GPU + HF_TOKEN
```
