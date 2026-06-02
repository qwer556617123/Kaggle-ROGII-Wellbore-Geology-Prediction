# ROGII Wellbore Geology Prediction

Kaggle-style project for predicting post-Prediction-Start (post-PS) TVT values for three horizontal wells from trajectory, GR logs, typewell curves, and pre-PS TVT input.

## Current Status

- Best baseline: `lgbm_final_reg_train.py` using the v13_reg LightGBM settings.
- Local validation: RMSE 15.08 row-weighted, 11.95 per-well mean.
- Public LB: 12.269.
- Core target: `target_correction = true_tvt - anchored_physics`.
- Core physics estimate: `anchor_tvt + slope * (Z - Z_anchor)`.
- Current direction: Public LB improvement through test-well-specific analysis, but tracked in a disciplined experiment log.

## Main Commands

These commands are intentionally kept as the stable root-level entry points until the script move can be completed:

```powershell
python lgbm_final_reg_train.py
python run_lgbm_on_test_csv.py
```

Expected output from the final baseline training script:

```text
models/lgbm_final_reg.pkl
submissions/lgbm_final_reg.csv
```

## Project Map

- `train/`, `test/`: local competition data.
- `features/`: checked-in lightweight feature metadata plus ignored generated feature artifacts.
- `models/`, `submissions/`: generated outputs, ignored by git.
- `docs/experiment_log.md`: version history and experiment classification.
- `docs/next_strategy.md`: next Public LB strategy and guardrails.
- `docs/file_inventory.md`: current file roles and planned physical layout.
- `docs/project_state.md`: concise current-state handoff for the project.
- Planned script layout:
  - `scripts/main/`: final training, inference, and Kaggle kernel entry scripts.
  - `scripts/experiments/`: exploratory model and feature scripts.
  - `scripts/diagnostics/`: evaluation and analysis helpers.

## Version Summary

- v1-v4: dTVT/increment modeling; validation around RMSE 17-19 with accumulation issues.
- v5: switched to correction modeling; validation improved to about RMSE 15.76.
- v8: anchored physics plus post-PS trajectory features; validation about RMSE 15.17.
- v13_reg: stronger regularization and fixed final training; current best baseline.
- v14-v24: KNN, formation, xcorr, spatial, GR deviation, and hard-well experiments; useful clues but not yet a stable replacement for v13_reg.

## Current Rules

- Use well-level validation, not row-level random splitting.
- Keep v13_reg as the benchmark until a method clearly improves either validation evidence or Public LB.
- Every new experiment must record:
  - which test well it is meant to fix;
  - expected TVT trend change;
  - expected delta versus v13;
  - validation or LB result;
  - whether it should be continued.

## Notes

- The root script move is planned but currently blocked by the local approval/session state. Until that is resolved, the stable root-level commands remain the source of truth.
- Avoid broad refactoring during score-chasing. The near-term project risk is scattered experimentation, not shared utility duplication.
