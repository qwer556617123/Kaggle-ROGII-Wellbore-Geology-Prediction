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

## Submission Result

Submitted two CPU-safe notebook versions:

- v33 true no-exact PF/artifact `70/30`: Public LB 8.391.
- v34 true no-exact PF/artifact `75/25`: Public LB 8.328.

Both are worse than the old v16/v21 family around 8.13. The local artifact
hidden-mask CV is useful as a component smoke test, but it is not aligned enough
to choose Public LB blend weights.

This also changes the interpretation of earlier "no-exact" submissions. The
wrapper summary reported `artifact_exact_overlap=0`, but before the 2026-06-27
component fix the saved artifact config could re-enable exact-coordinate
blending inside the v10 artifact component. The old best v16/v21 should be
treated as the practical baseline behavior, not as proof that true no-exact is
best.

## Decision

Reject true no-exact 70/30 and 75/25. Restore the submission default to the
v16-like safe baseline: PF/artifact 80/20 with the artifact component's saved
config controlling exact-coordinate behavior.
