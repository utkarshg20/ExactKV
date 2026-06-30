# External benchmark panels (LongBench, RULER, BFCL, HumanEval, MBPP)

ExactKV external panels measure **token-path drift** on prompts drawn from established long-context and structured-output benchmarks. They do **not** reproduce official LongBench, RULER, BFCL, HumanEval, or MBPP leaderboard scores.

## Claim boundary

| We report | We do **not** claim |
|-----------|---------------------|
| `first_divergence_index`, `divergence_rate`, `acceptance_rate` | LongBench task accuracy |
| `exactkv_failure` under verifier semantics | RULER effective context length |
| Per-category / per-bucket aggregates | BFCL executability pass rate |
| Diagnostic `timing_ms` per cell | HumanEval / MBPP `pass@1` or test execution |

Use language like: *"ExactKV drift panel on LongBench-shaped prompts"* or *"pilot subset from THUDM/LongBench"*, not *"we evaluate on LongBench"* unless you run a documented full export.

## Priority order (paper roadmap)

1. **LongBench** — credibility, multi-task long-context
2. **RULER** — controlled 4K / 8K / 16K / 32K buckets
3. **BFCL** — JSON / tool-call drift case studies
4. **HumanEval / MBPP** — code-generation drift
5. Later: HELMET, InfiniteBench (stress)

## Cell schema

Each cell is keyed by:

```text
(model, compressor, dataset_family, task_category, prompt_id, context_bucket, max_new_tokens)
```

Outputs include the same metrics as the evidence-plus panel plus:

- `dataset_family` (`longbench`, `ruler`, `bfcl`, `humaneval`, `mbpp`)
- `task_type` / `task_category`
- `category_summary` and `bucket_summary` in the report JSON

## Quick start

### Offline / CI (no GPU)

```bash
python3 scripts/run_external_panel.py --family longbench --deterministic-mode --smoke
python3 scripts/run_external_panel.py --family ruler --deterministic-mode --smoke
python3 scripts/run_external_panel.py --family mbpp --deterministic-mode --smoke
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
python3 scripts/plan_next_external_runs.py --input reports/external_panels
pytest tests/test_external_panel.py tests/test_validate_external_panel.py -q
```

### GPU pilot (bundled prompts)

```bash
python3 scripts/run_external_panel.py --family longbench --device cuda --dtype float16 --max-prompts 6 --context-buckets 2048,4096
python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 --context-buckets 4096,8192
python3 scripts/run_external_panel.py --family mbpp --device cuda --dtype float16 --max-prompts 6 --context-buckets 512,1024 --max-new-tokens 16,32
```

Artifacts:

- `reports/external_panels/longbench_raw.json`
- `reports/external_panels/ruler_raw.json`
- `reports/external_panels/mbpp_raw.json`
- matching `*_summary.md`

MBPP panels measure **ExactKV token drift only**. Generated code is **not** executed against `test_list`.

### Artifact validator and run planner

After any external panel run (offline or GPU):

```bash
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
python3 scripts/plan_next_external_runs.py --input reports/external_panels
```

Outputs:

- `reports/external_panels/validation_report.{md,json}` — schema and aggregate consistency checks
- `reports/external_panels/next_run_plan.md` — prioritized next commands (HF LongBench, Mistral rerun, larger BFCL, MBPP smoke, conditional RULER 16K)

The validator checks report/cell fields, `exactkv_failure` consistency, divergence aggregates, `timing_ms` presence, deterministic vs GPU separation, and prompt-source metadata. It does **not** certify official benchmark scores.

### BFCL from Berkeley Function-Calling Leaderboard (BFCL v3)

```bash
python3 scripts/export_bfcl_subset.py --max-per-category 13 --max-total 50 --output benchmarks/prompts/bfcl_export.jsonl
python3 scripts/run_external_panel.py --family bfcl --prompt-source export --device cuda --dtype float16 --max-prompts 50 --context-buckets 1024,2048 --max-new-tokens 16,32
```

Uses `huggingface_hub` to fetch real BFCL v3 JSON (not `datasets.load_dataset`). Claim boundary: ExactKV drift only, not BFCL executability scores.

### LongBench from Hugging Face

```bash
pip install datasets
python3 scripts/export_longbench_subset.py --max-per-subset 2 --output benchmarks/prompts/longbench_export.jsonl
python3 scripts/run_external_panel.py --family longbench --prompt-source hf --max-prompts 12 --device cuda
```

Official dataset: [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench)

### RULER full generation

Bundled `benchmarks/prompts/ruler_pilot.jsonl` provides RULER-**style** tasks for drift measurement. For official RULER task generation at exact token lengths, use [NVIDIA/RULER](https://github.com/NVIDIA/RULER) and export JSONL in the same schema (`prompt_id`, `category`, `prompt`, `declared_context_tokens`).

## Dataset references

| Panel | Repo | Hugging Face / notes |
|-------|------|----------------------|
| LongBench | [THUDM/LongBench](https://github.com/THUDM/LongBench) | `THUDM/LongBench` |
| RULER | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | Generated via repo scripts |
| BFCL | [gorilla BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` |
| HumanEval | [openai/human-eval](https://github.com/openai/human-eval) | `openai/openai_humaneval` |
| MBPP | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) | `google-research-datasets/mbpp`; bundled `mbpp_pilot.jsonl` |

## RunPod

Reuse the evidence-plus setup (`docs/RUNPOD_EVIDENCE_PLUS.md`). Sync repo, then:

```bash
python3 scripts/run_external_panel.py --family longbench --device cuda --dtype float16 --max-prompts 6 --context-buckets 2048,4096 --max-new-tokens 16,32
```

Start with 2048–4096 buckets on A5000 (50 GB volume). Scale to 8192+ after disk and VRAM checks.

## Implementation files

- `exactkv/benchmarks/external_dataset_loaders.py` — HF + bundled JSONL loaders
- `exactkv/benchmarks/external_panel.py` — panel runner (truncate/pad to context bucket)
- `scripts/run_external_panel.py` — CLI
- `scripts/validate_external_panel_artifacts.py` — artifact validator
- `scripts/plan_next_external_runs.py` — next-run planner
- `scripts/export_longbench_subset.py` — HF export helper
- `benchmarks/prompts/*_pilot.jsonl` — offline pilot prompts (includes `mbpp_pilot.jsonl`)

## Paper wording (suggested)

> In addition to the release and evidence-plus panels, we provide external drift panels on LongBench- and RULER-shaped prompt suites, plus BFCL/HumanEval/MBPP diagnostic panels for structured-output and code drift. These panels report ExactKV divergence metrics only, not official benchmark scores.
