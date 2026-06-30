# External panel run quality report

Generated: 2026-06-27T16:07:35.232497+00:00

## Artifact inventory

- `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels/longbench_pilot_merged_raw.json`: ok=72, deterministic=False, failures=0
- `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels/ruler_2048_4096_merged_raw.json`: ok=48, deterministic=False, failures=0
- `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels/ruler_8192_merged_raw.json`: ok=24, deterministic=False, failures=0
- `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels/bfcl_merged_raw.json`: ok=48, deterministic=False, failures=0
- `/Users/utkarshgupta/Documents/ExactKV/reports/external_panels/humaneval_merged_raw.json`: ok=24, deterministic=False, failures=0

## GPU vs offline separation

- **GPU merged cells counted:** 216
- **Non-ok cells in merged files:** 0

Offline deterministic smoke files (excluded from GPU analysis):
- `reports/external_panels/longbench_raw.json`: 4 cells, family=longbench, smoke=True
- `reports/external_panels/humaneval_raw.json`: 4 cells, family=humaneval, smoke=True
- `reports/external_panels/bfcl_raw.json`: 4 cells, family=bfcl, smoke=True
- `reports/external_panels/mbpp_raw.json`: 4 cells, family=mbpp, smoke=True
- `reports/external_panels/ruler_raw.json`: 4 cells, family=ruler, smoke=True

## Validation checklist

| Check | Result |
|-------|--------|
| All merged artifacts `deterministic_mode=false` | True |
| Report-level `exactkv_failures=0` | True |
| Per-cell `exactkv_failure=0` | True |
| Divergence only in int4_sim | True |
| noop/int8 divergence count | 0 |
| int4_sim divergent cells | 15 / 72 (0.208) |
| summary_all cross-check | False |

## summary_all.json cross-check

- **longbench_pilot:** summary cells=216, computed=72, match=False, div summary=0.065, computed=0.083
- **ruler_2048_4096:** summary cells=144, computed=48, match=False, div summary=0.076, computed=0.083
- **ruler_8192:** summary cells=72, computed=24, match=False, div summary=0.097, computed=0.083
- **bfcl:** summary cells=144, computed=48, match=False, div summary=0.042, computed=0.062
- **humaneval:** summary cells=72, computed=24, match=False, div summary=0.000, computed=0.000

## Contradictions

- summary_all cell count mismatch for longbench_pilot
- summary_all cell count mismatch for ruler_2048_4096
- summary_all cell count mismatch for ruler_8192
- summary_all cell count mismatch for bfcl
- summary_all cell count mismatch for humaneval

## Paper update safety

**Resolve contradictions before paper updates.**

## Workflow notes (from README)

- ## Failed workflow steps
- ## Skipped workflow steps
