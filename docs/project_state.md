# Project State

Last updated: 2026-06-03

## What This Project Is

This project predicts post-Prediction-Start TVT for the ROGII Wellbore Geology Prediction task. The local dataset has 773 training wells and 3 test wells. Each well has:

- a horizontal-well CSV with trajectory, GR, `TVT_input`, and training `TVT` when available;
- a typewell CSV with `TVT`, `GR`, and geology;
- post-PS rows identified by the first blank or missing `TVT_input`.

The submission format is `wellid_rowindex,tvt` for the post-PS rows of the three test wells.

## Current Best Understanding

The best working approach is not to predict TVT from scratch. It builds an anchored physics estimate, then trains LightGBM to predict the residual correction:

```text
anchored_physics = anchor_tvt + slope * (Z - Z_anchor)
target_correction = true_tvt - anchored_physics
prediction = anchored_physics + predicted_correction
```

The current benchmark is `lgbm_final_reg_train.py`, using the v13_reg parameters:

- local row-weighted RMSE: 15.08;
- local per-well mean RMSE: 11.95;
- Public LB: 12.269.

This is the reference point to beat, but it should not be treated as a good solution. Current competition reality is that all tried branches are poor; v13_reg is only the least-bad organized baseline.

## Development History In One Pass

- Early dTVT/increment models were weaker because errors accumulated down the post-PS interval.
- The major improvement came from switching to correction modeling around an anchored physics baseline.
- v8 established the strong anchored-physics plus trajectory-feature baseline.
- v13_reg improved the LightGBM regularization and is the current best baseline.
- v14-v24 explored KNN, formation, GR alignment, spatial neighbors, 3D physics, and GR deviation. These are not random failures; they are useful diagnostic branches, especially for Public LB well-specific strategy.

## Test-Well Interpretation

From `scripts/diagnostics/hard_well_analysis.csv`:

| Well | Key signals | Working interpretation |
| --- | --- | --- |
| `000d7d20` | Post-Z net about +100 ft; test-train analog is `FLAT`; xcorr analog was weak. | Avoid aggressive correction. Compare v13 against anchor-heavy or very small smooth trend changes. |
| `00bbac68` | Post-Z net about +176.5 ft; analog is `COMPLEX`; spatial/formation clues likely matter. | Best candidate for v23/spatial or formation-dip targeted blending. |
| `00e12e8b` | High pre-GR variance, lower post-GR variance; analog is `MODERATE`; stronger xcorr analog. | GR signals may help, but only with smooth, robust corrections. |

## Current Organization State

Completed:

- README rewritten into a readable project summary.
- `docs/experiment_log.md` records baseline, useful clues, low-priority branches, and the required template for new experiments.
- `docs/next_strategy.md` defines the Public LB-focused plan and guardrails.
- `docs/file_inventory.md` maps script roles and their organized locations.
- `scripts/` contains `main/`, `experiments/`, and `diagnostics/`.

Not completed:

- No current script or branch should be treated as an active strong candidate. Existing scripts are organized references until a better strategy is designed.

## Current Working Rules

- Do not refactor model internals while score-chasing.
- Do not start a new experiment without naming the target well and expected TVT trend change.
- Do not globally replace v13 unless the new method explains all three test wells.
- Keep root-level wrapper commands stable after physical script moves.

## Next Best Work

1. Generate a compact v13 trend report for the three test wells:
   - anchor, physics start/end, prediction start/end, min/max, standard deviation, correction range.
2. Generate the same compact report for v23 spatial inference if required artifacts exist locally.
3. Compare the two reports well-by-well and design one small blend candidate at a time.
4. Record every candidate in `docs/experiment_log.md` before submission.

The aim is disciplined Public LB improvement, not another broad feature sweep.
