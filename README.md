# ROGII Wellbore Geology Prediction

This repository is for the ROGII Wellbore Geology Prediction competition. The current goal is not to keep adding random model variants; it is to keep a small active workspace, validate locally before spending submissions, and only revive archived ideas with a concrete hypothesis.

## Current Baselines

| Method | Local signal | Public LB | Status |
| --- | ---: | ---: | --- |
| v23 spatial 3D correction | val around 14.06 in notes | 12.044 | Best known official baseline |
| v13_reg final LGBM | row RMSE 15.08, per-well 11.95 | 12.269 | Stable reference |
| train lookup | visible-ID oracle looked perfect | 15.883 | Not a valid oracle |

Current conclusion: all existing branches are weak. v23 is the best known official baseline, but small retrain/data-pool changes have produced much worse LB scores.

## Active Commands

Root wrappers are kept for the main commands:

```powershell
python lgbm_final_reg_train.py
python run_lgbm_on_test_csv.py
python lgbm_v15_formation_train.py
```

Use local CV before spending daily submissions:

```powershell
python scripts\diagnostics\cv_attenuated_physics.py --folds 5 --seed 42
python scripts\diagnostics\cv_lgbm_rank_replay.py --folds 3 --row-stride 10 --max-rounds 2000 --early-stopping 150
python scripts\main\train_ranked_lgbm_candidates.py
```

## Project Layout

- `scripts/main/`: stable mainline, inference, and Kaggle kernel references.
- `scripts/diagnostics/`: active diagnostics and validation gates.
- `scripts/experiments/`: active experimental scripts only.
- `scripts/archive/`: deprecated or low-priority scripts kept for history.
- `docs/`: project state, experiment findings, CV-vs-LB notes, and strategy.
- `features/`: small checked-in metadata; generated features are ignored.
- `submissions/`, `models/`, `predictions/`, `reports/`: generated outputs, ignored by git.

## Active Validation Notes

- Do not use the three visible test IDs as official oracle truth.
- Held-out well CV is useful for rejecting bad ideas, but current evidence shows it does not fully predict Public LB.
- Any candidate should either beat v13-like CV behavior or reproduce/improve the v23 12.044 path before using submission quota.

## Archive Policy

Archived scripts should stay archived unless they have:

- a named failure mode;
- a target validation gate;
- an expected Public LB effect;
- a clear reason they are better than the current v23/v13 references.
