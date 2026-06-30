# ExactKV Final Release Checklist (Release Synthesis)

Run `python3 scripts/validate_final_release_package.py` to verify the mechanical
items automatically.

## Artifacts present

- [x] Paper (Markdown) — `paper/ExactKV_Technical_Report.md`
- [x] Paper (LaTeX) — `paper/ExactKV_Technical_Report.tex`
- [x] Bibliography — `paper/references.bib` (+ `release_synthesis/references.bib`)
- [x] PDF export status — `paper/export_status.json` (PDF pending: no LaTeX/pandoc toolchain)
- [x] Website — `site/index.html`, `styles.css`, `main.js`, `README.md`
- [x] Launch posts — `launch/x_thread.md`, `launch/linkedin_post.md`, `launch/short_announcement.md`, `launch/launch_manifest.json`
- [x] Evidence ledger — `release_synthesis/evidence_ledger.md` (+ `.json`)
- [x] Claim decision table — `release_synthesis/claim_decision_table.md` (+ `.json`)
- [x] Artifact inventory — `release_synthesis/artifact_inventory.md` (+ `.json`/`.csv`)
- [x] Source-of-truth map — `release_synthesis/source_of_truth_map.md`
- [x] Project / version / phase lineage — `release_synthesis/{project,version,phase}_lineage.md`
- [x] Related-work audit — `release_synthesis/related_work_audit.md`
- [x] RELEASE.md

## Claim safety

- [x] No forbidden phrases in public copy (site, launch, paper)
- [x] No unsupported "beats X" claims
- [x] No unsupported "first ever / first and only" claims
- [x] No unsupported active-GPU-memory or end-to-end-speed claims
- [x] Phase F labelled kernel microbenchmark only
- [x] Compression ratios labelled stored tensor byte ratios
- [x] SpectralQuant labelled fallback/proxy
- [x] Shard labelled probe-first
- [x] VeriCache: "does not reproduce" present
- [x] Production: "not a production serving system" present

## Evidence integrity

- [x] Benchmark source of truth = `reports/scale_7b/raw.json` (1500 cells, 0 failures)
- [x] Two timelines documented (V1–V21 vs Phase A–K) as distinct numbering systems
- [x] Historical Phase A (336 cells) marked superseded as public headline
- [x] No secrets/tokens in generated artifacts

## Validation commands

```bash
python3 scripts/validate_final_release_package.py
python3 scripts/check_no_secrets.py
python3 scripts/audit_public_claims.py
python3 scripts/check_site_claims.py
pytest tests/test_final_release_package.py -q
pytest tests/test_site_artifacts.py -q
pytest -q   # full suite: 3085 passed, 110 skipped (2026-06-26)
git diff --check
```

## Manual review (human sign-off)

- [ ] Editorial pass on launch copy tone before posting publicly
- [ ] Confirm GitHub repo URL substituted into `site/` and `launch/` links
- [ ] Build PDF once a LaTeX/pandoc toolchain is available (see `paper/export_status.json`)
- [ ] Confirm model license/access terms for any public redistribution of outputs
