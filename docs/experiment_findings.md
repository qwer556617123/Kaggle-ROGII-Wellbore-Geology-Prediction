# Experiment Findings

Last updated: 2026-06-03

## Submission Reality Check

Kaggle submission history shows:

- Best known official result in this checkout history: v23 spatial, Public LB 12.044.
- v13_reg reference: Public LB 12.269.
- Direct train lookup submission: Public LB 15.883.

Therefore local train truth for the three visible test IDs is not a reliable official oracle. Use it only to inspect code behavior and failure shapes, not to choose final competition predictions blindly.

## Current New Experiments

### GR Path Search

`scripts/experiments/gr_path_search.py` treats TVT prediction as a GR/typewell path-search problem. Local oracle diagnostics were poor:

- `000d7d20`: path RMSE 37.85, anchor RMSE 7.45.
- `00bbac68`: path RMSE 120.99, anchor RMSE 15.26.
- `00e12e8b`: path RMSE 20.55, anchor RMSE 7.92.

Conclusion: direct GR path following is too easily misled for these wells.

### Attenuated Physics

`scripts/experiments/attenuated_physics_sweep.py` tests:

```text
prediction = anchor_tvt + alpha * (anchored_physics - anchor_tvt)
```

Local diagnostic sweep found:

- global best alpha near 0.07, local oracle RMSE 9.42;
- per-well diagnostic alpha: `000d7d20=0.115`, `00bbac68=0.075`, `00e12e8b=0.040`.

Conclusion: the existing models likely over-amplify post-PS TVT trend. This is useful evidence, but local oracle numbers should not be trusted as official LB estimates.

## Generated Candidate Files

- `submissions/attenuated_physics_alpha_0p07.csv`
- `submissions/attenuated_physics_per_well_alpha.csv`
- `submissions/submission.csv` copied from the per-well alpha candidate for Kaggle CLI submission.

The candidate CSV was validated locally:

- columns: `id,tvt`;
- rows: 14151;
- ID order matches `sample_submission.csv`;
- no missing TVT values.

## Kaggle Submit Attempt

Kaggle CLI was available through:

```powershell
D:\anaconda\envs\kaggle-dev\python.exe -m kaggle
```

Two submission attempts uploaded the file but returned:

```text
400 Client Error: Bad Request
```

Because the file validates locally, likely causes are competition/API submission restrictions or account-side limits rather than CSV shape.

## Next Move

Use v23 spatial 12.044 as the official best known baseline, not local oracle RMSE. The next real breakthrough should target why v23 helped official LB while local truth diagnostics disagree. Prioritize:

- reproducing the exact v23 12.044 submission path;
- comparing v23 output trend against attenuated-physics output;
- building a conservative blend around v23, not around train lookup truth.
