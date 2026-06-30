# External panel next-run plan

Generated: 2026-06-26T21:52:36.198468+00:00

Planner recommends ExactKV drift panel commands only. It does not schedule official benchmark score reproduction or code execution for MBPP.

## Recommended commands (priority order)

### 2. Mistral-7B external panel rerun (after disk fix)

Prior RunPod workflow failed with disk quota exceeded after Llama cache.

```bash
# Clear HF cache or expand volume before rerun
rm -rf ~/.cache/huggingface/hub/models--*
bash scripts/run_external_gpu_workflow.sh  # or rerun families one model at a time
```

Blockers:
- RunPod 50 GB volume; run Llama and Mistral sequentially with cache cleanup

### 4. MBPP smoke (bundled pilot, token drift only)

MBPP pilot loader added; offline smoke exists.

```bash
python3 scripts/run_external_panel.py --family mbpp --device cuda --dtype float16 --max-prompts 6 --context-buckets 512,1024 --max-new-tokens 16,32
```

### 5. RULER 16K bucket (conditional on 8K timing)

8192 bucket completed with acceptable timing and zero ExactKV failures.

```bash
python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 --context-buckets 16384 --max-new-tokens 16,32
```

## Maintenance

```bash
python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels
python3 scripts/plan_next_external_runs.py --input reports/external_panels
```
