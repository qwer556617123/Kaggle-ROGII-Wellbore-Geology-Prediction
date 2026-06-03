# CV vs Leaderboard Alignment

Last updated: 2026-06-03

Daily submissions are limited, so local validation must be used as a ranking gate. The goal is not to predict the exact Public LB value; it is to check whether local CV ranks candidate methods in roughly the same order as LB.

When this document says the sample is small, it means the number of leaderboard submissions that also have a comparable local score is small. There are seven submitted leaderboard rows, but only a few currently have a normal validation metric recorded, and the train-lookup row is a different signal type. It should not be mixed with ordinary CV scores.

## Known LB Results

See `docs/lb_history.csv` for the tracked table.

Key points:

- v23 original spatial run is the best known official baseline: 12.044.
- v13_reg is stable but worse: 12.269.
- train lookup scored 15.883, so visible test IDs in train are not an official oracle.
- later v23 retrains scored 17.221-17.482 despite plausible local notes, so implementation/data-pool details matter a lot.

## Current Rank Alignment

Run:

```powershell
python scripts\diagnostics\evaluate_cv_lb_alignment.py
```

Current summary:

| Comparison | n | Spearman | Kendall | Avg abs rank error | Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| val_rmse only | 3 | 0.50 | 0.33 | 0.67 | closest current proxy, but too small for hard tuning |
| comparable CV-like signals | 3 | 0.50 | 0.33 | 0.67 | directional only |
| all local signals | 4 | 0.00 | 0.00 | 1.50 | contaminated by visible-ID oracle |

Generated files:

- `docs/cv_lb_alignment_summary.csv`
- `docs/cv_lb_rank_alignment.csv`

## Current CV Gate

Held-out attenuation CV over all 773 train wells:

| Method | CV row RMSE | CV per-well RMSE |
| --- | ---: | ---: |
| learned global alpha | 15.80 | 12.79 |
| anchor alpha 0.000 | 15.83 | 12.81 |
| alpha 0.040 | 15.96 | 13.02 |
| alpha 0.070 | 16.82 | 13.82 |
| raw physics alpha 1.000 | 108.99 | 95.66 |

This rejects raw physics and weak attenuation variants, but it does not explain why v23 original got 12.044 while v23 retrains got 17+.

## Pseudo-Test Rank Replay

Run:

```powershell
python scripts\diagnostics\cv_lgbm_rank_replay.py --folds 3 --row-stride 10 --max-rounds 800 --early-stopping 80 --out-prefix reports\cv_lgbm_rank_replay_pilot
```

This uses all 773 train wells, split by well id, with every 10th post-PS row kept for a faster pilot. It gives many pseudo-test wells instead of only the three official test wells.

Pilot aggregate:

| Method | Row RMSE | Per-well RMSE | Interpretation |
| --- | ---: | ---: | --- |
| pseudo_oracle_lookup | 0.00 | 0.00 | contamination check only |
| lgbm_v13_like_base | 13.90 | 11.27 | best pilot rank |
| lgbm_v8_like_base | 14.02 | 11.39 | close to v13-like |
| lgbm_extra_features | 14.21 | 11.43 | extra features did not help in this pilot |
| anchor | 15.87 | 12.81 | strong simple baseline |
| alpha_0p07 | 16.85 | 13.81 | worse than anchor here |
| physics | 109.01 | 95.62 | rejected |

Tracked summary: `docs/cv_lgbm_rank_replay_pilot_summary.csv`.

Important: this replay is a local pseudo-test ranking tool. It does not by itself prove the Public LB order, but it provides more local ranking samples than the official test set.

The later blend confirmation is tracked in `docs/cv_lgbm_rank_replay_blend_confirm_summary.csv`. It selected `v13_beta_0p80` and `v13_beta_0p75` as the first two submission candidates. See `docs/ranked_lgbm_candidates.md`.

## Alignment Judgment

Current verdict:

- CV should be judged mainly by rank alignment against LB, especially Spearman/Kendall.
- Current rank alignment is weak-to-moderate, not complete.
- CV is still valid as a rejection gate.
- A candidate that only looks good on visible-ID train truth should be rejected.
- A candidate that beats CV but cannot reproduce the exact v23 12.044 path should still be treated cautiously.

## CV Parameter Policy

- First try to make the CV rank order match LB rank order.
- Prefer well-level splits and row/per-well RMSE reporting.
- Avoid tuning CV parameters to explain only one leaderboard point.
- If repeated LB-backed candidates do not align, fall back to common default well-level CV settings and use CV only as a sanity gate.

## Submission Rule

Before spending one of the five daily submissions, a candidate should satisfy at least one:

- beat v13-style held-out CV behavior;
- improve hard-well behavior without hurting anchor-like wells;
- reproduce the known v23 12.044 path and change only one controlled factor.
