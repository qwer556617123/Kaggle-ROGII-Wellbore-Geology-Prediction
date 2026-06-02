# File Inventory

This inventory records the organized script roles. Root-level wrappers are kept for stable reference commands and convenience, not because any current branch is strong.

## Stable Mainline

| File | Role | Notes |
| --- | --- | --- |
| `lgbm_final_reg_train.py` | Root wrapper for final v13_reg training and submission generation | Current best baseline. |
| `run_lgbm_on_test_csv.py` | Root wrapper for local test CSV inference / comparison helper | Currently references v23 spatial model through organized script. |
| `lgbm_v15_formation_train.py` | Root wrapper for v15 formation script | Weak reference branch; keep easy to rerun, not active. |
| `scripts/main/lgbm_final_reg_train.py` | Final v13_reg implementation | Current best baseline implementation. |
| `scripts/main/run_lgbm_on_test_csv.py` | Local test CSV inference implementation | Useful for inspecting Kaggle-like predictions. |
| `scripts/main/lgbm_v15_formation_train.py` | v15 formation implementation | Weak reference branch. |
| `scripts/main/kaggle_kernel_inference.py` | Kaggle inference script | Uses precomputed v23 spatial model metadata and neighbor stats. |
| `scripts/main/kaggle_kernel_v3.py` | Kaggle kernel variant | Historical kernel packaging; keep with main/kernel scripts. |

## Baseline And Model Experiments

| File | Role | Current priority |
| --- | --- | --- |
| `scripts/experiments/lgbm_train.py` | Early parquet-feature LightGBM trainer | Low; superseded by anchored-physics correction flow. |
| `scripts/experiments/lgbm_v2_train.py` | Early LightGBM experiment | Low. |
| `scripts/experiments/lgbm_v3_train.py` | Early LightGBM experiment | Low. |
| `scripts/experiments/lgbm_v8_train.py` | Anchored-physics baseline | High as reference; foundation for v13. |
| `scripts/experiments/lgbm_v12_train.py` | Intermediate pre-v13 experiment | Low unless needed for history. |
| `scripts/experiments/lgbm_v13_train.py` | v13 regularization comparison | Medium; documents why v13_reg won. |
| `scripts/experiments/lgbm_final_train.py` | Earlier final model variant | Medium; compare against final_reg if needed. |
| `scripts/experiments/lgbm_grid_search.py` | Hyperparameter search | Low unless restarting broad model tuning. |
| `scripts/experiments/lgbm_v14_knn_train.py` | KNN feature experiment | Low/medium; revisit only with a specific well target. |
| `scripts/experiments/lgbm_v14_knn_final_train.py` | KNN finalization variant | Low/medium. |
| `scripts/experiments/lgbm_v17_xcorr_train.py` | GR cross-correlation experiment | Medium clue for GR alignment diagnostics. |
| `scripts/experiments/lgbm_v18_3dphys_train.py` | 3D physics feature experiment | Medium clue. |
| `scripts/experiments/lgbm_v18b_3dphys_train.py` | 3D physics variant | Medium clue. |
| `scripts/experiments/lgbm_v19_horiz_train.py` | Horizontal trajectory experiment | Medium clue. |
| `scripts/experiments/lgbm_v19_final_train.py` | v19 final variant | Medium clue. |
| `scripts/experiments/lgbm_v20_disp_train.py` | Displacement feature experiment | Medium clue. |
| `scripts/experiments/lgbm_v20_train.py` | v20 GR/diagnostic-heavy experiment | Medium clue; useful with hard-well analysis. |
| `scripts/experiments/lgbm_v21_grtvt_train.py` | GR-to-TVT feature experiment | Medium clue. |
| `scripts/experiments/lgbm_v22_spatial_train.py` | Spatial neighbor feature experiment | High clue for `00bbac68`. |
| `scripts/experiments/lgbm_v23_spatial_train.py` | Spatial neighbor v8-hyperparam experiment | High clue; compare against v13 trends. |
| `scripts/experiments/lgbm_v24_grdev_train.py` | GR deviation rolling/trend experiment | High clue for controlled GR correction. |
| `scripts/experiments/lstm_train.py` | Early LSTM attempt | Low; complex without proven score path. |
| `scripts/experiments/lstm_v2_train.py` | Later LSTM attempt | Low; keep as experiment history. |
| `scripts/experiments/dtw_baseline.py` | DTW baseline | Low/diagnostic; monotonic assumptions are risky. |
| `scripts/experiments/viterbi_tvt.py` | GR/HMM style trajectory search | Medium diagnostic; useful if testing non-monotonic TVT paths. |

## Feature Engineering

| File | Role | Notes |
| --- | --- | --- |
| `scripts/experiments/feature_engineering.py` | Early feature generation | Historical. |
| `scripts/experiments/feature_engineering_v2.py` | Extended feature generation | Historical. |
| `scripts/experiments/feature_engineering_v3.py` | GR/typewell matching and xcorr feature generation | Useful clue but not mainline by itself. |
| `features/test_well_nbr_stats.json` | Test neighbor stats | Used by spatial inference variants. |
| `features/test_well_nbr_stats_658_backup.json` | Backup neighbor stats | Keep for reproducing v22/v23 comparisons. |
| `features/test_well_nbr_stats_full773.json` | Full-pool neighbor stats | Used for full-pool spatial strategy checks. |

## Diagnostics And Analysis

| File | Role | Notes |
| --- | --- | --- |
| `scripts/diagnostics/eval_v20.py` | v20 evaluation helper | Diagnostics. |
| `scripts/diagnostics/eval_lgbm_xcorr.py` | xcorr model evaluation helper | Diagnostics. |
| `scripts/diagnostics/spatial_neighbor_analysis.py` | Spatial neighbor analysis | Important for v22/v23 strategy. |
| `scripts/diagnostics/show_importance.py` | Feature importance display | Diagnostics. |
| `scripts/diagnostics/hard_well_analysis.csv` | Hard-well comparison table | Informs Public LB strategy. |

## Submission Helpers

| File | Role | Notes |
| --- | --- | --- |
| `scripts/diagnostics/gen_anchor_submission.py` | Constant-anchor baseline submission | Useful sanity baseline. |
| `scripts/diagnostics/gen_train_lookup_submission.py` | Train lookup style submission helper | Low; check leakage risk before use. |
| `scripts/diagnostics/gen_v8_submission.py` | v8 submission generator | Useful reference for v8/v13 comparisons. |

## Competition Branch Notes

- No current branch is considered strong. v13_reg is the least-bad baseline.
- v15 has a root wrapper plus implementation under `scripts/main/`, but is only a weak reference branch.
- v6 is referenced by project history but has no standalone script in this checkout and should be treated as weak historical context.

Do not change model logic during file organization. Post-move checks should verify that stable root wrappers still resolve.
