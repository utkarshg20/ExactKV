# Experiment 041: Shard Combined Stress (stream_bits=4 × 128 tokens)

_Restricted combined stress — not Shard integration, not a default ExactKV compressor._

> **Shard remains an external-drafter probe, not an integrated ExactKV compressor.**
> **External Shard README claims are not ExactKV results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**
> **Results are scoped to this bounded panel only.**

Artifacts (gitignored): `reports/experiment_041_shard_combined_stress.json`

---

## 1. Purpose

Test whether **combining** lossy decode streaming (`stream_bits=4`) with longer generation (`max_new_tokens=128`) increases Shard external-drafter drift while ExactKV verification still preserves full-KV greedy output.

## 2. Setup

| Item | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Prompt panel | Exp 039/040 32-prompt panel (reused) |
| Platform | RunPod L40S (proxy SSH `5vh87lgjxtyeou-64411443@ssh.runpod.io`) |
| `SHARD_REPO_PATH` | `/root/shard` |
| `HF_HOME` | `/workspace/.cache/huggingface` (offline load; no `HF_TOKEN` in shell) |
| Script | `scripts/research/run_exp041_shard_combined_stress.py` |
| Run date | 2026-06-13 |

Shard settings:

| Knob | Value |
| --- | --- |
| `streaming` | `true` |
| `stream_bits` | **4** (lossy; 8 is documented lossless) |
| `stream_qjl` | `false` |
| `k_target_cr` | `16.0` |
| `max_new_tokens` | **128** |
| `draft_len` | 4 |

```bash
export SHARD_REPO_PATH=/root/shard
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
bash scripts/research/exp041_shard_combined_stress_runpod.sh
```

## 3. Why this combination was tested

Exp 040 showed independent drift increases from:

- **length_128tok** (31.25% divergence rate)
- **stream_bits_4** (25.00% divergence rate)

This run isolates the **combined** effect on the same 32-prompt panel.

## 4. Results

| Metric | Value |
| --- | --- |
| `combined_status` | **pass** |
| `prompt_count` | 32 |
| `blocked_prompt_count` | 0 |
| `tokenizer_alignment_pass` | **true** |
| `exactkv_failures` | **0** |
| `divergence_count` | **18 / 32** |
| `divergence_rate` | **56.25%** |
| `semantic_divergence_count` | 2 |
| `formatting_divergence_count` | 16 |
| `accepted_prefix_mean` | 91.44 / 128 |
| `accepted_prefix_median` | 96.5 |
| `accepted_prefix_min` | 14 |
| `recommendation` | `stop_shard_bounded_probe_complete` |

## 5. Comparison to Exp 040

| Exp 040 / 041 setting | divergence rate | divergence count | mean accepted prefix |
| --- | --- | --- | --- |
| `baseline_64tok` | 18.75% | 6/32 | 58.22/64 |
| `length_128tok` | 31.25% | 10/32 | 104.62/128 |
| `stream_bits_4` (64 tok) | 25.00% | 8/32 | 56.59/64 |
| **Exp 041 combined** | **56.25%** | **18/32** | **91.44/128** |

| vs Exp 040 single | rate delta | count delta |
| --- | --- | --- |
| vs `baseline_64tok` | +37.50 pp | +12 |
| vs `length_128tok` | +25.00 pp | +8 |
| vs `stream_bits_4` | +31.25 pp | +10 |

**Combined setting increased drift beyond every Exp 040 single-knob run** (`increased_vs_all_exp040_singles=true`). The combined rate (56.25%) exceeds the maximum single-setting rate (31.25%) by a wide margin — lossy streaming and longer generation interact to increase draft drift on this panel.

## 6. Accepted-prefix distribution

Histogram (`count=32`):

`{14:3, 21:1, 40:1, 45:1, 59:1, 62:1, 67:2, 70:1, 79:1, 86:1, 94:2, 95:1, 98:1, 115:1, 128:14}`

14 prompts reached full 128-token accepted prefixes; 18 diverged earlier. Mean accepted prefix fell to 91.44 vs 104.62 for `length_128tok` alone (lossy streaming shortened accepted runs).

## 7. Divergence examples

| prompt_id | category | first_div | draft | verifier | kind |
| --- | --- | --- | --- | --- | --- |
| `struct_json_002` | structured_json | 67 | ` defined` | ` duplicated` | formatting |
| `struct_json_003` | structured_json | 115 | ` of` | ` use` | formatting |
| `rc_001` | retrieval_copy | 70 | `.` | `,` | formatting |
| `rc_004` | retrieval_copy | 95 | `,` | ` and` | formatting |
| `lc_001` | long_context_summary | 45 | ` output` | ` exactly` | formatting |

Two additional divergences were classified **semantic** (token substitutions); top-five cited examples are formatting/punctuation class.

## 8. exactkv_failures

**0** — ExactKV full-KV verifier output matched greedy full-KV on every prompt despite 18 draft divergences.

## 9. What this proves

- **Combining** `stream_bits=4` with `max_new_tokens=128` **materially increases** Shard external-drafter drift (56.25% vs 31.25% max single-knob in Exp 040).
- **ExactKV verification remains exact** (`exactkv_failures=0`) under the harshest tested Shard configuration.
- The bounded Shard probe series (Exp 038–041) has characterized feasibility, stress, ablation, and combined stress without integrating Shard into ExactKV.

## 10. What this does not prove

- Shard production integration or default-registry readiness.
- Speedup, VRAM savings, serving throughput, or model accuracy improvement.
- Generalization beyond this single configuration and 32-prompt Llama panel.
- That `stream_bits=4` at 64 tokens alone predicts combined behavior — interaction effects matter.

## 11. Limitations

- One configuration only — not a benchmark suite.
- 32 prompts × 128 tokens on L40S (~12.5 min GPU).
- `stream_bits=4` is lossy; `stream_bits=8` is documented lossless in Shard.
- Divergence classification is a coarse heuristic.
- Offline cached Llama weights; no fresh `HF_TOKEN` in run shell (weights already on pod).

## 12. Final Shard recommendation

**`stop_shard_bounded_probe_complete`** — bounded external-drafter characterization is complete. Do **not** add Shard to the default compressor registry. **Move next research effort to SpectralQuant** (or archive Shard external-drafter work). ExactKV verification held at `exactkv_failures=0` throughout Exp 038–041.

## Related

- [`EXPERIMENT_040_SHARD_EXTERNAL_ABLATION.md`](EXPERIMENT_040_SHARD_EXTERNAL_ABLATION.md)
- [`EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md`](EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md)

```bash
export SHARD_REPO_PATH=/root/shard
bash scripts/research/exp041_shard_combined_stress_runpod.sh
```
