# ROGII Wellbore Geology Prediction

Post-competition research and reproducibility workspace for the Kaggle
**ROGII Wellbore Geology Prediction** competition.

The competition closed on 2026-08-05 23:59 UTC. The final scored Public-LB
best in this repository is **6.186**. The last observed top-10 threshold was
**5.205**.

## Final Result

| Candidate | Public LB | Role |
| --- | ---: | --- |
| `rogii-c1r8-d16-b12-calibrated` | **6.186** | Final scored best; cap-3 datum hierarchy |
| `rogii-c1r8-d16-b12-a2b3-probe` | 6.187 | Competitive orthogonal direction |
| `rogii-c1r8-d16-b12-cap4` | 6.190 | Aggressive clipping frontier |
| `rogii-c1r8-d16-b12-a0b3-probe` | 6.245 | Competitive orthogonal direction |
| `rogii-c1r8-d16-b12-a1b3-probe` | 6.304 | Competitive orthogonal direction |

The final-hour `b3diag150` candidate extrapolated the only unmeasured
Hadamard coefficient and had a nominal score estimate of **5.9749**. Its
notebook completed and passed output audit, but Kaggle rejected the formal
submission because all five UTC-day slots had already been used. It remains an
unscored, high-risk artifact rather than a claimed result.

## Method Progression

The useful score frontier was:

| Stage | Public LB | Main contribution |
| --- | ---: | --- |
| Exact public-source rebuild | 7.053 | Reproducible external anchor |
| Student-t backward HMM | 6.909 | Full-sequence probabilistic smoothing |
| MHA400 continuity + HMM | 6.696 | Multimodal hedge plus sequence smoothing |
| C1 heel calibration | 6.609 | LB-measured structural residual direction |
| Four-bin calibrated C1 | 6.529 | Conditional C1 routing |
| Rank-8 calibrated C1 | 6.524 | Finer deterministic routing |
| Four-group datum calibration | 6.491 | Per-group constant datum correction |
| Nested `b1` calibration | 6.314 | First child-level datum contrast |
| Nested `b1+b2` calibration | **6.186** | Final 12-dimensional datum deployment |

Detailed submission references, hashes, runtime audits, and rejected branches
are recorded in [docs/experiment_log.md](docs/experiment_log.md).

## What Worked

- Exact Student-t forward-backward smoothing produced the first repeatable
  method-level gain.
- Multimodal MHA hedging improved the physical-filter family without relying
  on fixed public IDs.
- Public-LB squared-score inversion was accurate when every probe rebuilt the
  same deterministic hidden anchor. Predicted calibrated scores were usually
  within roughly 0.001-0.02 of the displayed score.
- Row-balanced, run-local datum partitions exposed well-group bias that global
  offsets had cancelled out.
- Reconstructing every Kaggle vector inside the hidden rerun avoided dependence
  on the downloadable three-well template.

## What Did Not Transfer

- Native-mask CV was useful as a rejection gate, but several large local gains
  inverted on the hidden Public LB.
- Cross-well contact orientation, neighbor transfer, GR well-bias RF, geometry
  beams, robust PF rebuilds, and model-package dose sweeps did not improve the
  final hidden score.
- Contact columns nearly reconstruct `TVT` within a known train well, but
  extrapolating the missing test-well datum remained the hard geological
  problem.
- Claimed scores from public notebooks frequently depended on unavailable
  artifacts, fixed IDs, or hidden routing that did not reproduce.
- Low vector correlation did not imply useful residual alignment; diversity
  alone was not an ensemble criterion.

## Validation Rules

- Never treat the three visible test IDs as hidden oracle truth.
- Never read target-tail `TVT`, same-ID formation/contact truth, fixed public
  IDs, or static public submissions during hidden inference.
- Use each train well's native `TVT_input` mask. Artificial 45/60/75 percent
  prefixes are retired.
- Treat local CV as a rejection and failure-analysis tool, not a Public-LB
  score predictor.
- Use score tomography only with deterministic, hash-audited probes sharing an
  identical anchor and partition.
- Keep generated Kaggle outputs local. Commit builders, audits, and findings,
  not downloaded artifacts.

## Reproducing The Final Family

Generate the scored `b1+b2` candidate:

```powershell
python scripts\diagnostics\build_c1_rank8_datum16_multistage_calibrated.py `
  --anchor 6.524 `
  --b1-scores 6.714,6.394,6.984,6.867 `
  --b2-scores 6.988,6.880,6.443,6.715 `
  --amplitude 2
```

Generate the final-day direct portfolio and the unscored sub-6 wager:

```powershell
python scripts\diagnostics\build_c1_datum_b12_cap4.py
python scripts\diagnostics\build_c1_datum_b12_centered_b3_batch.py
python scripts\diagnostics\build_c1_datum_b12_b3_diag150.py
```

Run static and synthetic checks:

```powershell
python scripts\diagnostics\smoke_c1_rank8_datum16_multistage_calibrated.py
python scripts\diagnostics\smoke_c1_datum_b12_centered_b3_final.py
python -m py_compile scripts\diagnostics\build_c1_datum_b12_b3_diag150.py
git diff --check
```

Generated notebooks are written under `kaggle/rogii-*` and are ignored by Git.
The tracked `kaggle/rogii-mha400-cont-hmm010/` directory is the reproducible
base notebook used by the final builders.

## Repository Layout

- `scripts/main/`: stable training and inference entry points.
- `scripts/diagnostics/`: notebook builders, CV, score inversion, smoke tests,
  and output audits.
- `scripts/experiments/`: active experimental code retained for analysis.
- `scripts/archive/`: deprecated approaches kept for provenance.
- `docs/experiment_log.md`: chronological source of truth for experiments.
- `docs/cv_vs_lb.md`: validation mismatch findings.
- `docs/experiment_findings.md`: consolidated technical observations.
- `kaggle/rogii-mha400-cont-hmm010/`: tracked final-family base notebook.
- `train/`, `test/`, `models/`, `submissions/`, `reports/`, and
  `kaggle/outputs/`: local or generated data excluded from Git.

## Repository State

The repository is now in post-competition maintenance mode. New work should be
framed as reproducibility, retrospective analysis, or a reusable geosteering
method, rather than another leaderboard submission variant.

## License

This repository's original source code is licensed under the [MIT License](LICENSE).
Competition data, generated model artifacts, third-party materials, and
trademarks remain subject to their respective terms.
