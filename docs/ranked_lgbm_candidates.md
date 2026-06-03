# Ranked LGBM Candidates

Last updated: 2026-06-04

## Selected Candidates

The current top two local candidates are:

1. `v13_beta_0p80`
2. `v13_beta_0p75`

Both use the v13-like feature set and LightGBM regularization, trained on all 773 wells from `features/train_features_v2.parquet` for 1254 rounds. The final prediction is blended toward the anchor:

```text
final_tvt = anchor + beta * (lgbm_tvt - anchor)
```

For leaderboard submission this competition requires a completed Kaggle Notebook version, not direct CSV upload. The notebook script lives in `kaggle/rogii-lgbm-final-submit/`.

## Local Evidence

Confirmation CV command:

```powershell
python scripts\diagnostics\cv_lgbm_rank_replay.py --folds 3 --row-stride 10 --max-rounds 2000 --early-stopping 150 --blend-betas 0.7,0.75,0.8,0.9,1.0 --out-prefix reports\cv_lgbm_rank_replay_blend_confirm
```

Top confirmation rows:

| Candidate family | Row RMSE | Per-well RMSE |
| --- | ---: | ---: |
| lgbm_v13_like_base_anchor_beta_0p8 | 13.782 | 11.102 |
| lgbm_v13_like_base_anchor_beta_0p75 | 13.797 | 11.099 |
| lgbm_v13_like_base_anchor_beta_0p9 | 13.803 | 11.153 |
| lgbm_v13_like_base_anchor_beta_0p7 | 13.828 | 11.111 |
| lgbm_v8_like_base_anchor_beta_0p8 | 13.884 | 11.210 |

Tracked CSVs:

- `docs/cv_lgbm_rank_replay_blend_confirm_summary.csv`
- `docs/ranked_lgbm_candidates_report.csv`

## Generated Files

Ignored local artifacts:

- `models/ranked_lgbm_v13_base_1254.pkl`
- `submissions/v13_beta_0p80.csv`
- `submissions/v13_beta_0p75.csv`
- `reports/ranked_lgbm_candidates_report.csv`

Both submission CSVs were validated against `sample_submission.csv`:

- 14,151 rows
- columns: `id,tvt`
- ID order matches sample submission
- no missing predictions

## Submission Order

Submit in this order:

1. `submissions/v13_beta_0p80.csv`
2. `submissions/v13_beta_0p75.csv`

Rationale: beta 0.80 is best by row RMSE; beta 0.75 is nearly tied and best by per-well RMSE among the top two.

## Kaggle Notebook Submissions

| Candidate | Notebook version | Kaggle ref | Status | Public LB |
| --- | ---: | ---: | --- | ---: |
| v13_beta_0p80 | 11 | 53338095 | COMPLETE | 12.548 |
| v13_beta_0p75 | 12 | 53338233 | COMPLETE | 12.650 |

Direct CSV upload returned HTTP 400 because the competition expects notebook-version submission.

Result: neither ranked blend beat the known v13_reg 12.269 or v23 spatial 12.044 baselines. The beta sweep moved in the wrong LB direction: beta 0.75 was slightly better by pseudo-CV per-well RMSE, but worse on Public LB than beta 0.80.
