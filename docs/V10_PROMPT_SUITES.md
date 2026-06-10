# V10 Prompt Suites

**Status:** Phase 1 complete — versioned suites committed; validator and tests pass.
**Not a comprehensive public benchmark.** These suites harden ExactKV's controlled
evaluation story before v1.0.0. They do **not** claim external benchmark scores,
production serving readiness, or runtime performance.

---

## 1. Purpose

V10 prompt suites provide a **categorized, versioned, deterministic** evaluation set
for Experiment 012 and later V10 work. They extend — but do not simply duplicate — the
original 34-prompt `core` suite used in Experiments 002–011.

Metrics remain **acceptance, exactness, divergence, and honest memory accounting** only.
No throughput, latency, tokens/sec, speedup, `runtime_seconds`, or
`active_gpu_kv_bytes` fields appear in suite metadata.

---

## 2. Suite list

| Suite file | Suite version prefix | Min prompts | Role |
|---|---|---:|---|
| `benchmarks/prompts/core_v2.jsonl` | `core_v2` | 40 | Regression superset of `core`; mixed categories |
| `benchmarks/prompts/code_structured.jsonl` | `code_structured` | 20 | Syntax-, indentation-, and bracket-sensitive code |
| `benchmarks/prompts/long_context.jsonl` | `long_context` | 15 | Long synthetic prefill stress |
| `benchmarks/prompts/reasoning_math.jsonl` | `reasoning_math` | 15 | Arithmetic and step-wise reasoning |
| `benchmarks/prompts/multilingual.jsonl` | `multilingual` | 15 | Non-English prefill and translation |
| `benchmarks/prompts/retrieval_copy.jsonl` | `retrieval_copy` | 10 | Near-verbatim copy and entity repetition |
| `benchmarks/prompts/tool_json.jsonl` | `tool_json` | 10 | JSON and tool-call shaped prompts |

**Total:** 128 prompts (minimum target ~115).

---

## 3. Prompt counts

Run the validator for live counts and per-category breakdown:

```bash
python scripts/validate_v10_prompt_suites.py
```

As of Phase 1 commit:

| Suite | Count |
|---|---:|
| `core_v2` | 40 |
| `code_structured` | 21 |
| `long_context` | 15 |
| `reasoning_math` | 15 |
| `multilingual` | 15 |
| `retrieval_copy` | 11 |
| `tool_json` | 11 |
| **Total** | **128** |

---

## 4. Category taxonomy

### Primary categories (`primary_category`)

One per prompt:

- `natural_language`
- `code`
- `structured_json`
- `long_context`
- `reasoning_math`
- `multilingual`
- `retrieval_copy`
- `qa_factual`
- `tool_schema`

### Secondary tags (`secondary_tags`, optional list)

- `short_prefill`, `medium_prefill`, `long_prefill`
- `repetition_heavy`, `symbol_heavy`, `whitespace_sensitive`, `numeric_heavy`

### Required JSONL fields

| Field | Required |
|---|---|
| `id` | yes — globally unique across all V10 files |
| `prompt` | yes — non-empty string |
| `primary_category` | yes |
| `suite_version` | yes — must start with suite family prefix (e.g. `core_v2_v1`) |
| `secondary_tags` | no — list when present |
| `source_note` | no — provenance annotation |

V10 suites use this schema. Legacy suites (`smoke`, `core`, …) retain `prompt_id` /
`category` for backward compatibility.

---

## 5. Dataset construction principles

1. **Deterministic and in-repo** — no live web fetch at sweep time.
2. **No copyrighted long passages** — synthetic or short factual prompts only.
3. **No benchmark laundering** — do not claim MMLU, HumanEval, or other external scores.
4. **No compressor cherry-picking** — prompts not selected to favour any compressor.
5. **Regression anchors** — some `core_v2` prompts adapt original `core` themes with
   new IDs; documented in `source_note` where applicable.
6. **Safe and non-sensitive** — no credentials, PII, or harmful content.
7. **Exactness compatible** — greedy decoding and the `exactkv_failures == 0` gate.

---

## 6. How to validate suites

```bash
# Full validation + summary table
python scripts/validate_v10_prompt_suites.py

# Unit tests
pytest tests/test_v10_prompt_suites.py -v
```

The validator checks:

- All required files exist
- JSONL parses; required fields present
- Global `id` uniqueness
- Allowed `primary_category` and `secondary_tags`
- `suite_version` matches suite family prefix
- Minimum per-suite counts
- No forbidden performance field names in metadata
- Summary table by suite and `primary_category`

---

## 7. How these suites differ from the old `core` suite

| Aspect | `core` (V3) | V10 suites |
|---|---|---|
| Count | 34 | 128 across 7 files |
| Schema | `prompt_id`, `category` | `id`, `primary_category`, `suite_version` |
| Stratification | Mixed in one file | Dedicated category suites + `core_v2` superset |
| Versioning | Implicit | Explicit `suite_version` (e.g. `core_v2_v1`) |
| Long context | 4 `long_prompt` rows | 15-row `long_context` suite |
| Multilingual | 4 `translation` rows | 15-row `multilingual` suite |
| Tool/JSON | 4 `json` rows | `tool_json` + `structured_json` in `core_v2` |
| Used in published exps | 002–011 | Experiment 012+ (planned) |

Legacy `core` remains unchanged for historical experiment reproduction.

---

## 8. Limitations

- **Not comprehensive** — 128 prompts do not cover all real-world LLM use cases.
- **Not a public leaderboard benchmark** — controlled engineering evaluation only.
- **Synthetic long context** — `long_context` uses in-repo synthetic text, not
  retrieved documents.
- **Small multilingual panel** — five languages, short sentences; not a translation benchmark.
- **No loader in default registry yet** — Experiment 012 will add sweep integration;
  Phase 1 provides files + validator only.
- **Simulated compressors unchanged** — `_sim` rows remain int8-container simulations.

---

## 9. Experiment 012 readiness

Phase 1 delivers:

- [x] All seven suite files at or above minimum counts
- [x] `scripts/validate_v10_prompt_suites.py`
- [x] `tests/test_v10_prompt_suites.py`
- [x] This document

Experiment 012 (Phase 2) still requires:

- [ ] Sweep script loading V10 suites (or `--suite-file` manifest)
- [ ] Per-category leaderboard rendering
- [ ] Published report with `exactkv_failures == 0`

Do **not** run Experiment 012 until Phase 2 is explicitly approved.

---

## Related

- [`V10_SCOPE_STATEMENT.md`](V10_SCOPE_STATEMENT.md) — V10 phased scope
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) — published experiments 001–011
- [`validate_v10_prompt_suites.py`](../scripts/validate_v10_prompt_suites.py) — validator
