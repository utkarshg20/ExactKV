# External benchmark GPU panels

**Claim boundary:** These are ExactKV drift panels on external benchmark prompts. They are **not** official LongBench, RULER, BFCL, or HumanEval leaderboard scores.

Generated: 2026-06-26T21:08:07.146973+00:00

## Commands (conservative RunPod A5000 workflow)

```bash
bash scripts/run_external_gpu_workflow.sh
```

Per-step commands:

```bash
python3 scripts/run_external_panel.py --family longbench --device cuda --dtype float16 \
  --max-prompts 6 --context-buckets 2048,4096 --max-new-tokens 16,32

python3 scripts/export_longbench_subset.py --max-per-subset 2
python3 scripts/run_external_panel.py --family longbench --prompt-source hf --device cuda --dtype float16 \
  --max-prompts 12 --context-buckets 2048,4096 --max-new-tokens 16,32

python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 \
  --context-buckets 2048,4096 --max-new-tokens 16,32

python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 \
  --context-buckets 8192 --max-new-tokens 16,32

python3 scripts/run_external_panel.py --family bfcl --device cuda --dtype float16 \
  --max-prompts 25 --context-buckets 1024,2048 --max-new-tokens 16,32

python3 scripts/run_external_panel.py --family humaneval --device cuda --dtype float16 \
  --max-prompts 20 --context-buckets 1024,2048 --max-new-tokens 32
```

Models were run **one at a time** (Llama-3.1-8B, then Mistral-7B) to reduce VRAM/disk pressure.

## Merged GPU results

| Group | Prompt source | Cells (ok/total) | Divergence rate | Mean acceptance | ExactKV failures | Mean ms | P90 ms |
|-------|---------------|------------------:|----------------:|----------------:|-----------------:|--------:|-------:|
| bfcl | pilot | 144/144 | 0.042 | 1.000 | 0 | 4651.801 | 6333.347 |
| humaneval | pilot | 72/72 | 0.000 | 1.000 | 0 | 5699.336 | 6324.916 |
| longbench_pilot | pilot | 216/216 | 0.065 | 0.994 | 0 | 6327.629 | 8682.867 |
| ruler_2048_4096 | pilot | 144/144 | 0.076 | 0.995 | 0 | 6341.886 | 8681.769 |
| ruler_8192 | pilot | 72/72 | 0.097 | 0.995 | 0 | 12054.189 | 13625.374 |

## Per-file artifacts

- `reports/external_panels/longbench_pilot_Llama_3_1_8B_raw.json`: family=longbench source=pilot cells=72/72 div=0.083 accept=0.994 failures=0 mean_ms=6422.296 p90_ms=8759.823
- `reports/external_panels/longbench_pilot_Mistral_7B_Instruct_v0_3_raw.json`: family=longbench source=pilot cells=72/72 div=0.028 accept=0.994 failures=0 mean_ms=6138.296 p90_ms=8384.741
- `reports/external_panels/longbench_pilot_merged_raw.json`: family=longbench source=pilot cells=72/72 div=0.083 accept=0.994 failures=0 mean_ms=6422.296 p90_ms=8759.823
- `reports/external_panels/ruler_2048_4096_Llama_3_1_8B_raw.json`: family=ruler source=pilot cells=48/48 div=0.083 accept=0.993 failures=0 mean_ms=6436.317 p90_ms=8792.586
- `reports/external_panels/ruler_2048_4096_Mistral_7B_Instruct_v0_3_raw.json`: family=ruler source=pilot cells=48/48 div=0.062 accept=1.000 failures=0 mean_ms=6153.025 p90_ms=8397.195
- `reports/external_panels/ruler_2048_4096_merged_raw.json`: family=ruler source=pilot cells=48/48 div=0.083 accept=0.993 failures=0 mean_ms=6436.317 p90_ms=8792.586
- `reports/external_panels/ruler_8192_Llama_3_1_8B_raw.json`: family=ruler source=pilot cells=24/24 div=0.083 accept=0.993 failures=0 mean_ms=12231.604 p90_ms=13639.786
- `reports/external_panels/ruler_8192_Mistral_7B_Instruct_v0_3_raw.json`: family=ruler source=pilot cells=24/24 div=0.125 accept=0.999 failures=0 mean_ms=11699.36 p90_ms=13050.724
- `reports/external_panels/ruler_8192_merged_raw.json`: family=ruler source=pilot cells=24/24 div=0.083 accept=0.993 failures=0 mean_ms=12231.604 p90_ms=13639.786
- `reports/external_panels/bfcl_Llama_3_1_8B_raw.json`: family=bfcl source=pilot cells=48/48 div=0.062 accept=0.999 failures=0 mean_ms=4715.217 p90_ms=6342.236
- `reports/external_panels/bfcl_Mistral_7B_Instruct_v0_3_raw.json`: family=bfcl source=pilot cells=48/48 div=0.000 accept=1.000 failures=0 mean_ms=4524.971 p90_ms=6084.093
- `reports/external_panels/bfcl_merged_raw.json`: family=bfcl source=pilot cells=48/48 div=0.062 accept=0.999 failures=0 mean_ms=4715.217 p90_ms=6342.236
- `reports/external_panels/humaneval_Llama_3_1_8B_raw.json`: family=humaneval source=pilot cells=24/24 div=0.000 accept=1.000 failures=0 mean_ms=5790.775 p90_ms=6382.807
- `reports/external_panels/humaneval_Mistral_7B_Instruct_v0_3_raw.json`: family=humaneval source=pilot cells=24/24 div=0.000 accept=1.000 failures=0 mean_ms=5516.457 p90_ms=6073.265
- `reports/external_panels/humaneval_merged_raw.json`: family=humaneval source=pilot cells=24/24 div=0.000 accept=1.000 failures=0 mean_ms=5790.775 p90_ms=6382.807

## Failed workflow steps

- **longbench_pilot_Mistral_7B_Instruct_v0_3**: exit=1 log=reports/external_panels/logs/longbench_pilot_Mistral_7B_Instruct_v0_3_20260626T191812Z.log
- **ruler_2048_4096_Mistral_7B_Instruct_v0_3**: exit=1 log=reports/external_panels/logs/ruler_2048_4096_Mistral_7B_Instruct_v0_3_20260626T191812Z.log
- **bfcl_Mistral_7B_Instruct_v0_3**: exit=1 log=reports/external_panels/logs/bfcl_Mistral_7B_Instruct_v0_3_20260626T191812Z.log
- **humaneval_Mistral_7B_Instruct_v0_3**: exit=1 log=reports/external_panels/logs/humaneval_Mistral_7B_Instruct_v0_3_20260626T191812Z.log

## Skipped workflow steps

- **longbench_hf_Llama_3_1_8B**: datasets not installed
- **longbench_hf_Mistral_7B_Instruct_v0_3**: datasets not installed

## Merge groups without GPU data

- longbench_hf

## Logs

Workflow logs: `reports/external_panels/logs/`

Summary JSON: `reports/external_panels/summary_all.json`

