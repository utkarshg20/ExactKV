# ExactKV Release Checklist (Phase J)

Pre-launch gate — all items should pass before public announcement.

---

## Evidence & audits

- [ ] `python3 scripts/check_release_evidence.py` — **PASS**
- [ ] `python3 scripts/run_novelty_audit.py` — novelty audit complete
- [ ] `python3 scripts/audit_public_claims.py` — **PASS**
- [ ] `python3 scripts/check_no_secrets.py` — **PASS**
- [ ] `python3 scripts/check_public_release.py` — **PASS**
- [ ] `python3 -m pytest -q` — full suite green

---

## Public release consistency

- [ ] `reports/public_release/README_PUBLIC.md` uses **1500-cell** scale_7b as authoritative evidence
- [ ] Phase A **336-cell** panel labeled historical/internal (not final headline)
- [ ] `release_manifest.json` lists all required `source_artifacts`
- [ ] `leaderboard_final.json` parseable and sourced from scale_7b

---

## Claim safety in public copy

- [ ] Not a production serving system — stated
- [ ] Does not reproduce VeriCache — stated
- [ ] Phase F = kernel microbenchmark only — stated when speedups mentioned
- [ ] Compression ratios = stored tensor byte ratios — stated
- [ ] SpectralQuant = fallback/proxy — stated
- [ ] Shard = probe-first — stated
- [ ] No forbidden phrases: first ever, production ready, end-to-end speedup, active GPU memory savings, etc.

---

## Security

- [ ] No real HF tokens in logs, docs, scripts, or reports
- [ ] Rotate any token exposed in terminal history or RunPod commands

---

## Documentation freeze

- [ ] `docs/QUICKSTART.md`
- [ ] `docs/REPRODUCIBILITY.md`
- [ ] `docs/METRIC_DEFINITIONS.md`
- [ ] `docs/CLAIM_BOUNDARIES.md`
- [ ] `docs/RESULTS_SUMMARY.md`
- [ ] `docs/ARTIFACT_INDEX.md`
- [ ] Launch drafts refreshed: `blog_post.md`, `x_thread.md`, `linkedin_post.md`, `paper_draft.md`

---

## Repro wrapper

- [ ] `python3 scripts/exactkv_repro.py --release-check` passes
- [ ] `python3 scripts/exactkv_repro.py --full` without `--confirm-expensive` refuses expensive inference
- [ ] `reports/repro_manifest.json` written after repro runs

---

## Manual review

- [ ] Generated artifact timestamps reviewed
- [ ] Top leaderboard rows match `reports/scale_7b/leaderboard.json`
- [ ] No unqualified speed / memory / serving claims in README or public_release

---

## Lineage & historical inventory (Release Gate R2)

- [ ] `python3 scripts/build_project_lineage.py` — inventory generated
- [ ] `python3 scripts/check_project_lineage.py` — **PASS**
- [ ] Historical project lineage reviewed (`docs/PROJECT_LINEAGE.md`)
- [ ] Final report does **not** imply ExactKV started at Phase A
- [ ] Pre-A exploratory artifacts clearly separated from release-grade evidence
- [ ] No exploratory/no-go artifact used as a public performance claim
- [ ] V-series/demo claims backed by artifacts or marked illustrative
- [ ] Serving/vLLM/LMCache no-go work framed as claim-boundary evidence, not production support
- [ ] `docs/HISTORICAL_ARTIFACT_INVENTORY.md` and lineage docs reviewed before launch

---

## Final launch sign-off (Phase K)

- [ ] Evidence source of truth confirmed: `reports/scale_7b/raw.json`
- [ ] Public leaderboard has Llama numeric rows
- [ ] Public leaderboard has Mistral numeric rows
- [ ] Historical project lineage reviewed
- [ ] Final report does **not** imply ExactKV started at Phase A
- [ ] Pre-A exploratory artifacts separated from release-grade evidence
- [ ] Claim audit passed (`python3 scripts/audit_public_claims.py`)
- [ ] Secret scan passed (`python3 scripts/check_no_secrets.py`)
- [ ] Evidence checker passed (`python3 scripts/check_release_evidence.py`)
- [ ] Public release validator passed (`python3 scripts/check_public_release.py`)
- [ ] Lineage validator passed (`python3 scripts/check_project_lineage.py`)
- [ ] Launch pack validator passed (`python3 scripts/check_launch_pack.py`)
- [ ] Full pytest passed (`python3 -m pytest -q`)
- [ ] Technical report reviewed (`docs/EXACTKV_TECHNICAL_REPORT.md`)
- [ ] Blog reviewed (`docs/launch_blog_final.md`)
- [ ] X thread reviewed (`docs/launch_x_thread_final.md`)
- [ ] LinkedIn reviewed (`docs/launch_linkedin_final.md`)
- [ ] Demo cards reviewed (`reports/public_release/demo_cards.md`)
- [ ] No HF token in docs/reports/logs
- [ ] HF token rotated if exposed in prior terminal logs
- [ ] SpectralQuant fallback disclosed
- [ ] Shard probe-first disclosed
- [ ] VeriCache caveat included
- [ ] Serving caveat included
- [ ] Kernel microbenchmark caveat included
- [ ] Active GPU memory not claimed
- [ ] End-to-end speedup not claimed
- [ ] Sequential model execution disclosed
- [ ] **Manual approval to publish**

---

## Version lineage (Release Gate R2.1)

- [ ] `python3 scripts/build_project_lineage.py` — V1–V21 version lineage generated
- [ ] `docs/VERSION_LINEAGE.md` reviewed
- [ ] `reports/version_lineage.json` contains V1 through V21
- [ ] V1–V21 lineage reviewed
- [ ] No final material incorrectly says **V1–V13** as the **full** version arc
- [ ] Source-pending V14–V21 entries manually reviewed (partial evidence from Phase 14–21 docs)
- [ ] Version lineage separated from release-grade benchmark evidence (`reports/scale_7b/raw.json`)
