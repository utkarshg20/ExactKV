# Experiment 040: Shard External-Drafter Lossiness and Length Ablation

_Restricted ablation — not Shard integration, not a default ExactKV compressor._

> **Shard remains an external-drafter probe, not an integrated ExactKV compressor.**
> **External Shard README claims are not ExactKV results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**
> **Results are scoped to the tested panel only.**
> **`stream_bits=8` is documented-lossless streaming — do not call it lossy compression.**

Artifacts (gitignored): `reports/experiment_040_shard_external_ablation.json`

---

## 1. Purpose

Measure how Shard external-drafter draft divergence changes under **longer generation** and **documented compression/streaming knobs** after Exp 039 found 6/32 divergences at baseline (`stream_bits=8`, `max_new_tokens=64`).

## 2. Setup

| Item | Value |
| --- | --- |
| Platform | RunPod proxy SSH (`5vh87lgjxtyeou-64411443@ssh.runpod.io`) |
| GPU | NVIDIA L40S (46 GB) |
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Prompt panel | Exp 039 32-prompt panel (reused) |
| `SHARD_REPO_PATH` | `/root/shard` |
| Script | `scripts/research/run_exp040_shard_ablation.py` |
| Run date | 2026-06-13 |

```bash
export SHARD_REPO_PATH=/root/shard
export HF_TOKEN=...   # environment only
bash scripts/research/exp040_shard_ablation_runpod.sh
```

## 3. Settings tested

| setting_name | max_new_tokens | stream_bits | stream_qjl | k_target_cr | Notes |
| --- | --- | --- | --- | --- | --- |
| `baseline_64tok` | 64 | 8 | false | 16.0 | Exp039 baseline — **documented-lossless** streaming |
| `length_128tok` | 128 | 8 | false | 16.0 | Longer generation; same streaming |
| `stream_bits_4` | 64 | **4** | false | 16.0 | **Lossy** decode streaming (8 is documented lossless) |
| `stream_qjl_on` | 64 | 8 | **true** | 16.0 | Optional QJL quantizer (`streaming.py`) |
| `k_target_cr_32` | 64 | 8 | false | **32.0** | Stronger prefill compression via `Cache.from_model` |

## 4. Unsupported settings skipped

| Setting | Reason |
| --- | --- |
| `stream_bits_2` | Not documented in Shard README (only `stream_bits=8` shown as supported lossless streaming) |
| `custom_sink_window` | `sink_tokens` / `residual_window` not exposed via `Cache.from_model` in probe path |

## 5. Results table

| setting_name | div count | div rate | mean prefix | median | min | exactkv_failures | semantic | formatting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_64tok` | 6 | 0.1875 | 58.22 | 64 | 14 | **0** | 1 | 5 |
| `length_128tok` | **10** | **0.3125** | 104.62 | 128 | 14 | **0** | 2 | 8 |
| `stream_bits_4` | 8 | 0.2500 | 56.59 | 64 | 8 | **0** | 0 | 8 |
| `stream_qjl_on` | 6 | 0.1875 | 58.22 | 64 | 14 | **0** | 1 | 5 |
| `k_target_cr_32` | 5 | 0.1562 | 58.50 | 64 | 14 | **0** | 1 | 4 |

All settings: `tokenizer_alignment_pass=true`, `blocked_prompt_count=0`.

## 6. Accepted-prefix distribution by setting

| setting | histogram (selected) |
| --- | --- |
| `baseline_64tok` | `{14:2, 24:1, 42:1, 45:1, 60:1, 64:26}` |
| `length_128tok` | `{14:2, 24:1, 42:1, 45:1, 60:1, 64:4, 128:22}` |
| `stream_bits_4` | `{8:1, 14:2, 24:1, 42:1, 45:1, 60:1, 64:25}` |
| `stream_qjl_on` | same as baseline |
| `k_target_cr_32` | `{14:1, 24:1, 42:1, 45:1, 60:1, 64:27}` |

## 7. Divergence rate by setting

- **Highest:** `length_128tok` (31.25%) — longer generation exposes more drift.
- **Lossy streaming:** `stream_bits_4` (25.00%) — modest increase vs baseline (18.75%).
- **Lowest:** `k_target_cr_32` (15.62%) — stronger prefill CR did not increase drift on this panel.
- **No effect:** `stream_qjl_on` identical to baseline.

## 8. Semantic divergence examples

| setting | prompt_id | first_div | draft | verifier |
| --- | --- | --- | --- | --- |
| baseline / qjl / k32 | `struct_json_004` | 60 | `sku` | `qty` |
| length_128tok | `struct_json_004` | 60 | `sku` | `qty` |
| length_128tok | (additional semantic on longer panel) | — | 2 semantic total | — |

`stream_bits_4` produced **zero semantic** divergences — all 8 were formatting/punctuation class.

## 9. exactkv_failures

**0 for every setting** — ExactKV verification kept committed output aligned with full-KV greedy despite draft drift.

## 10. What this proves

- **Length ablation works:** `max_new_tokens=128` increased divergence rate (18.75% → 31.25%).
- **Lossy streaming (`stream_bits=4`)** increases drift modestly vs documented-lossless `stream_bits=8`.
- **ExactKV crash-test harness** holds across all ablation cells (`exactkv_failures=0`).
- `stream_qjl` and higher `k_target_cr` did not materially change drift on this bounded panel.

## 11. What this does not prove

- Shard production integration or default-registry readiness.
- Speedup, VRAM savings, serving throughput, or model accuracy improvement.
- Optimal Shard settings for production — panel is small and Llama-only.
- That `stream_bits=8` is lossy — Shard documents it as lossless streaming.

## 12. Limitations

- 32-prompt panel reused from Exp 039; not a full benchmark.
- Five settings × 32 prompts (~32 min on L40S).
- `stream_bits=4` is the lowest tested lossy value; README only documents `stream_bits=8`.
- Divergence classification is coarse heuristic.

## 13. Recommendation

**`expand_shard_lossy_ablation`** — continue bounded Shard external-drafter work (e.g. `stream_bits=4` at `max_new_tokens=128`, or additional documented knobs). ExactKV verification remained exact across settings. **Do not stop Shard yet**; also consider **SpectralQuant feasibility** as parallel Mode B probe — not a substitute for these Shard results.

Still **no** default registry entry.

## Related

- [`EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md`](EXPERIMENT_039_SHARD_EXTERNAL_STRESS_PANEL.md)
- [`EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md`](EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md)
