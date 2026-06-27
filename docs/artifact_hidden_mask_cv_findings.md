# Artifact Hidden-Mask CV Findings - 2026-06-27

## Result

The v10 artifact dataset was downloaded locally and evaluated on pseudo-hidden
train-well tails with exact coordinate overlap truly disabled by environment
override.

Combined over 12 lb-like wells at known fractions 0.45, 0.60, and 0.75:

| Candidate | Row RMSE | Mask-well mean RMSE | Decision |
| --- | ---: | ---: | --- |
| `blend_pf70` | 5.478 | 4.741 | Best local gate |
| `blend_pf75` | 5.535 | 4.760 | Conservative second candidate |
| `blend_pf80` | 5.601 | 4.790 | Current-family baseline |
| `pf` | 5.955 | 5.037 | PF-only baseline |
| `artifact` | 5.760 | 5.067 | Useful but not enough alone |

## Submission Decision

Submit two CPU-safe notebook versions:

- true no-exact PF/artifact `70/30`;
- true no-exact PF/artifact `75/25`.

This is a materially different test from the previous bracket because the v10
artifact inference code now respects `ROGII_EXACT_OVERLAP=0` instead of allowing
the saved artifact config to re-enable exact-coordinate blending.
