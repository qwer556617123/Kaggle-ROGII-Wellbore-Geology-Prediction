# File Inventory

This inventory records the current script roles before any physical script move. It is meant to preserve project understanding and make the eventual directory cleanup mechanical.

## Stable Mainline

| File | Role | Notes |
| --- | --- | --- |
| `lgbm_final_reg_train.py` | Final v13_reg training and submission generation | Current best baseline; keep as stable root entrypoint or wrapper target. |
| `run_lgbm_on_test_csv.py` | Local test CSV inference / comparison helper | Currently references v23 spatial model; useful for inspecting Kaggle-like predictions. |
| `kaggle_kernel_inference.py` | Kaggle inference script | Uses precomputed v23 spatial model metadata and neighbor stats. |
| `kaggle_kernel_v3.py` | Kaggle kernel variant | Historical kernel packaging; keep with main/kernel scripts. |

## Baseline And Model Experiments

| File | Role | Current priority |
| --- | --- | --- |
| `lgbm_train.py` | Early parquet-feature LightGBM trainer | Low; superseded by anchored-physics correction flow. |
| `lgbm_v2_train.py` | Early LightGBM experiment | Low. |
| `lgbm_v3_train.py` | Early LightGBM experiment | Low. |
| `lgbm_v8_train.py` | Anchored-physics baseline | High as reference; foundation for v13. |
| `lgbm_v12_train.py` | Intermediate pre-v13 experiment | Low unless needed for history. |
| `lgbm_v13_train.py` | v13 regularization comparison | Medium; documents why v13_reg won. |
| `lgbm_final_train.py` | Earlier final model variant | Medium; compare against final_reg if needed. |
| `lgbm_grid_search.py` | Hyperparameter search | Low unless restarting broad model tuning. |
| `lgbm_v14_knn_train.py` | KNN feature experiment | Low/medium; revisit only with a specific well target. |
| `lgbm_v14_knn_final_train.py` | KNN finalization variant | Low/medium. |
| `lgbm_v15_formation_train.py` | Formation feature experiment | Medium clue for formation-dip strategy. |
| `lgbm_v17_xcorr_train.py` | GR cross-correlation experiment | Medium clue for GR alignment diagnostics. |
| `lgbm_v18_3dphys_train.py` | 3D physics feature experiment | Medium clue. |
| `lgbm_v18b_3dphys_train.py` | 3D physics variant | Medium clue. |
| `lgbm_v19_horiz_train.py` | Horizontal trajectory experiment | Medium clue. |
| `lgbm_v19_final_train.py` | v19 final variant | Medium clue. |
| `lgbm_v20_disp_train.py` | Displacement feature experiment | Medium clue. |
| `lgbm_v20_train.py` | v20 GR/diagnostic-heavy experiment | Medium clue; useful with hard-well analysis. |
| `lgbm_v21_grtvt_train.py` | GR-to-TVT feature experiment | Medium clue. |
| `lgbm_v22_spatial_train.py` | Spatial neighbor feature experiment | High clue for `00bbac68`. |
| `lgbm_v23_spatial_train.py` | Spatial neighbor v8-hyperparam experiment | High clue; compare against v13 trends. |
| `lgbm_v24_grdev_train.py` | GR deviation rolling/trend experiment | High clue for controlled GR correction. |
| `lstm_train.py` | Early LSTM attempt | Low; complex without proven score path. |
| `lstm_v2_train.py` | Later LSTM attempt | Low; keep as experiment history. |
| `dtw_baseline.py` | DTW baseline | Low/diagnostic; monotonic assumptions are risky. |
| `viterbi_tvt.py` | GR/HMM style trajectory search | Medium diagnostic; useful if testing non-monotonic TVT paths. |

## Feature Engineering

| File | Role | Notes |
| --- | --- | --- |
| `feature_engineering.py` | Early feature generation | Historical. |
| `feature_engineering_v2.py` | Extended feature generation | Historical. |
| `feature_engineering_v3.py` | GR/typewell matching and xcorr feature generation | Useful clue but not mainline by itself. |
| `features/test_well_nbr_stats.json` | Test neighbor stats | Used by spatial inference variants. |
| `features/test_well_nbr_stats_658_backup.json` | Backup neighbor stats | Keep for reproducing v22/v23 comparisons. |
| `features/test_well_nbr_stats_full773.json` | Full-pool neighbor stats | Used for full-pool spatial strategy checks. |

## Diagnostics And Analysis

| File | Role | Notes |
| --- | --- | --- |
| `eval_v20.py` | v20 evaluation helper | Move to diagnostics when allowed. |
| `eval_lgbm_xcorr.py` | xcorr model evaluation helper | Move to diagnostics when allowed. |
| `spatial_neighbor_analysis.py` | Spatial neighbor analysis | Important for v22/v23 strategy. |
| `show_importance.py` | Feature importance display | Diagnostics. |
| `hard_well_analysis.csv` | Hard-well comparison table | Keep visible; informs Public LB strategy. |

## Submission Helpers

| File | Role | Notes |
| --- | --- | --- |
| `gen_anchor_submission.py` | Constant-anchor baseline submission | Useful sanity baseline. |
| `gen_train_lookup_submission.py` | Train lookup style submission helper | Low; check leakage risk before use. |
| `gen_v8_submission.py` | v8 submission generator | Useful reference for v8/v13 comparisons. |

## Planned Physical Layout

When move approval is available:

- `scripts/main/`: `lgbm_final_reg_train.py`, `run_lgbm_on_test_csv.py`, Kaggle kernel scripts, plus root wrappers.
- `scripts/experiments/`: model variants, feature engineering scripts, LSTM, DTW, Viterbi.
- `scripts/diagnostics/`: eval scripts, spatial analysis, feature importance, hard-well analysis.

Do not change model logic during the move. The first post-move check should only verify that stable entrypoints still resolve.
