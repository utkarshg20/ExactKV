# Experiment 086: Claim-Safe Demo Packaging (Phase 17A)

**Status:** run `scripts/research/run_exp086_claim_safe_demo_packaging.py --package-demo`.

> This is claim-safe demo packaging, not new runtime functionality.  
> ExactKV has guarded diagnostic shadow infrastructure, not streaming-attention token-commit integration.  
> Guarded decode-time shadow was tested as diagnostic-only observer work.  
> Shadow output cannot affect token commits in the tested path.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`PHASE_17_CLAIM_SAFE_DEMO.md`](PHASE_17_CLAIM_SAFE_DEMO.md) · `exactkv/demo/phase17_claim_safe_demo.py`

---

## Purpose

Package Phase 16 closeout evidence into demo narrative sections, six demo cards, and Q&A — without new experiments or runtime changes.

---

## Run

```bash
python3 scripts/research/run_exp086_claim_safe_demo_packaging.py --package-demo
```

Report: `reports/experiment_086_claim_safe_demo_packaging.json` (gitignored).

---

## Demo cards

| Card | Focus |
|------|-------|
| `attention_drift_card` | Qwen drift / numerics |
| `tolerance_policy_card` | Policy labels |
| `external_generation_shadow_card` | Post-hoc shadow |
| `live_round_observer_card` | Live snapshots |
| `guarded_decode_shadow_card` | Callback dry-run |
| `claim_freeze_card` | Allowed/forbidden |

---

## Next step

Phase 17B: broader model validation (proposed, explicit approval required).
