# Experiment 038: Shard External-Drafter Probe

_Restricted feasibility probe — not Shard integration, not a default ExactKV compressor._

> **Shard is not integrated as a default ExactKV compressor.**
> This is an **external-drafter feasibility probe** (Mode B).
> **External Shard claims are not ExactKV results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**

Artifacts (gitignored): `reports/experiment_038_shard_external_drafter_probe.json`

---

## 1. Purpose

Determine whether [Shard](https://github.com/krish1905/shard) can be used as an **external lossy drafter** under ExactKV-style full-KV verification on **Llama-3.1-8B**.

Core question: can ExactKV compare Shard-produced draft token IDs against a full-KV HF greedy verifier and report acceptance, first divergence, and correction need?

Expected outcome class: **restricted feasibility**, not production integration.

## 2. What Shard is being tested as

| Role | Shard in this probe |
| --- | --- |
| External draft source | `shard.Cache` + `enable_llama_fused_attention()` on an isolated Llama model |
| Authoritative verifier | ExactKV/HF full-KV greedy path (`ModelRuntime` + `VerificationEngine`) |
| Default compressor | **No** — not in registry |
| KVCompressor backend | **No** — Llama-only, attention monkey-patch |

Shard is probed only for **token-level draft vs verifier** comparison. This does **not** claim cache-level ExactKV integration unless the compressed Shard KV path is actually used (the probe uses Shard's `past_key_values=cache` generate path and labels it external).

## 3. What is not being claimed

- Shard is production-ready inside ExactKV.
- Shard README throughput, memory, or LongBench numbers are ExactKV results.
- Speedup, active GPU memory savings, or serving integration.
- Qwen `KVCompressor` backend via Shard.
- ExactKV panel leaderboard numbers for Shard (future-candidate tier only until a real probe passes).

## 4. Setup used (RunPod follow-up)

| Item | Value |
| --- | --- |
| Platform | RunPod proxy SSH (`5vh87lgjxtyeou-64411443@ssh.runpod.io`) |
| GPU | NVIDIA L40S (46 GB) |
| ExactKV path | `/workspace/ExactKV` (git clone + probe file overlay) |
| Shard path | `/root/shard` (external clone; **not** vendored into ExactKV) |
| `SHARD_REPO_PATH` | `/root/shard` |
| Dependencies | `pip install -e .` (ExactKV); `pip install -e /root/shard` (Shard) |
| `HF_TOKEN` on pod | **set** (environment only; not logged or committed) |
| Run script | `scripts/research/exp038_shard_probe_runpod.sh` |

```bash
# On RunPod (after Meta license + HF_TOKEN in environment only — never commit token)
export SHARD_REPO_PATH=/root/shard
export HF_TOKEN=...   # from environment; do not log or commit
bash scripts/research/exp038_shard_probe_runpod.sh
```

## 5. Shard repo path availability

| Check | Result |
| --- | --- |
| Clone | ✅ `git clone https://github.com/krish1905/shard.git /root/shard` |
| `SHARD_REPO_PATH` set | ✅ `/root/shard` |
| Vendored into ExactKV | ❌ (by design) |

## 6. Import result

| Field | Value |
| --- | --- |
| `shard_import_success` | **true** |
| `Cache` + `enable_llama_fused_attention` | present |
| Import-only run (no `--try-run`) | `probe_status: blocked` — expected; model not loaded |

## 7. Dependency issues

| Dependency | Status |
| --- | --- |
| `torch` | ✅ preinstalled on RunPod image |
| `transformers` | ✅ via ExactKV `pip install -e .` |
| Shard (`pip install -e /root/shard`) | ✅ |
| `triton` | optional (Shard has CPU/PyTorch fallbacks) |
| Meta Llama weights | ❌ **blocked** — gated model; `HF_TOKEN` required |

## 8. Model / tokenizer alignment

| Check | Result |
| --- | --- |
| Model targeted | `meta-llama/Llama-3.1-8B-Instruct` |
| Verifier load | ✅ with `HF_TOKEN` + Meta license |
| Shard draft path | ✅ `shard.Cache` + fused attention on isolated Llama |
| Initial alignment failure | BOS prefix mismatch (Shard `tokenizer(...)` prepends BOS; verifier used `add_special_tokens=False`) |
| Harness fix | `prompt_ids_comparable()` — BOS-prefix normalization in probe only (no generation/verification logic change) |
| `tokenizer_alignment_pass` | **true** (after BOS fix) |

## 9. `--try-run` executed?

**Yes** on RunPod L40S.

| Run | When | `HF_TOKEN` | Outcome |
| --- | --- | --- | --- |
| 1 | 2026-06-13 | no | `restricted_no_go` — gated Llama 401 |
| 2 | 2026-06-13 | yes | `restricted_no_go` — alignment false negative (BOS) |
| 3 | 2026-06-13 | yes | **`pass`** — alignment + panel complete |

Command:

```bash
python3 scripts/probe_shard_external_drafter.py --try-run --device cuda --dtype float16 --max-new-tokens 16 --draft-len 4
```

## 10. Probe result (RunPod, final)

| Field | Value |
| --- | --- |
| `probe_status` | **`pass`** |
| `blocked_reason` | _(empty)_ |
| `shard_import_success` | true |
| `tokenizer_alignment_pass` | **true** |
| `model_used` | `meta-llama/Llama-3.1-8B-Instruct` |
| `prompt_count` | 4 |
| `exactkv_failures` | **0** |
| `accepted_prefix_lengths` | `[16, 16, 16, 16]` |
| `first_divergence_indices` | `[null, null, null, null]` |
| **Recommendation** | **`restricted_go`** — Mode B external-drafter feasibility only; **not** default registry |

All four minimal prompts (JSON, retrieval-copy, long-context summary, code) produced **full 16-token greedy agreement** between Shard draft path and full-KV HF verifier on this panel. Divergence on harder/longer panels may still appear later; absence here is a real probe outcome, not a fabricated metric.

### Prior blocked runs

| Environment | `probe_status` | Notes |
| --- | --- | --- |
| No `SHARD_REPO_PATH` | `blocked` | CI/default |
| Local `~/shard`, import only | `blocked` | import OK; no `--try-run` |
| RunPod, no `HF_TOKEN` | `restricted_no_go` | gated-model 401 |
| RunPod, `HF_TOKEN`, pre-BOS fix | `restricted_no_go` | tokenizer alignment false negative |

## 11. Exact blockers (resolved for this probe)

1. ~~**Missing `HF_TOKEN`**~~ — resolved on RunPod (environment only).
2. ~~**Meta Llama license**~~ — accepted for token used.
3. ~~**BOS alignment harness**~~ — fixed via `prompt_ids_comparable()`.

Remaining constraints (by design):

- Shard remains **external** — not in ExactKV default registry.
- No speedup, VRAM, serving, or accuracy claims from this probe.

## 12. Claims allowed

- Shard import succeeded on RunPod with external clone.
- Probe harness completed `--try-run` with real acceptance/divergence fields on a 4-prompt panel.
- Full-KV HF verifier remained authoritative; `exactkv_failures=0` on this panel.
- Mode B external-drafter **restricted_go** feasibility classification.

## 13. Claims forbidden

- Shard production integration or default-registry entry.
- External Shard README memory/throughput numbers as ExactKV results.
- Speedup, VRAM savings, serving, or accuracy improvement.
- Generalizing this 4-prompt, 16-token panel to LongBench or production workloads.

## 14. Recommendation

**`restricted_go`** — external Shard drafter is **feasible** under ExactKV-style full-KV verification on Llama-3.1-8B for this probe panel.

### Next Shard step

1. Optional: expand prompt panel / `max_new_tokens` on RunPod (still Mode B only).
2. Document any divergence cases when they appear (useful signal, not harness failure).
3. Still **no** default registry entry — external drafter path only.

## Minimal prompt panel (when unblocked)

1. Structured JSON prompt
2. Retrieval-copy prompt
3. Long-context summary prompt
4. Code prompt

## Related docs

- [`EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md`](EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md)
- [`EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md`](EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md)
- [`PARALLEL_WORK_INTEGRATION_REPORT.md`](PARALLEL_WORK_INTEGRATION_REPORT.md)

Regenerate JSON:

```bash
export SHARD_REPO_PATH=/root/shard
python3 scripts/probe_shard_external_drafter.py              # import check
python3 scripts/probe_shard_external_drafter.py --try-run    # requires HF_TOKEN + GPU
```
