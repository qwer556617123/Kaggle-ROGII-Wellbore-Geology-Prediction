# Best Versions Reassessment - 2026-06-27

## Current Best

| Version | Public LB | Core setting | Decision |
| --- | ---: | --- | --- |
| v16 | 8.130 | PF h0.17 + artifact, 80/20; wrapper reported no-exact but component behavior may follow saved config | Best known |
| v21 | 8.131 | PF h0.17 + artifact, 75/25; same exact-overlap caveat | Statistical near-tie |
| v31 | 8.153 | 85% v16 + 15% v21, equivalent 79.25/20.75 | Worse than both |
| v30 | 8.160 | v16 + 3% contact surface | Worse |
| v32 | 8.408 | v16 - 3% contact surface residual | Strong reject |

## What The Best Two Versions Say

v16 and v21 are not two independent models. They are the same PF and artifact
components with different scalar weights. Their near-tie does not mean more
weight tuning is promising. The failed 77.5/22.5, 79.25/20.75, contact-positive,
and contact-negative probes show that the LB surface is not a smooth scalar
blend curve.

The better interpretation is:

- PF h0.17 is a strong shape prior.
- The artifact model carries a useful but dangerous correction.
- True no-exact artifact is harmful on Public LB; the old "no-exact" label is
  ambiguous because the artifact component could re-enable saved config
  behavior internally.
- Hidden wells likely split into different regimes: some prefer PF-heavy, some
  prefer more artifact, and scalar averaging blurs that split.

## Stop Conditions

Do not spend more submissions on:

- PF/artifact scalar weights between 75/25 and 85/15 on the existing component;
- true no-exact artifact weighting, now rejected by v33/v34;
- same-family weighted ensembles of v16/v21/v30/v31;
- contact surface positive or negative residuals;
- TabICL full reruns without a targeted ablation;
- dynamic rules based only on visible public well IDs or fixed public-test CSVs.

## Breakthrough Hypothesis

The missing move is not another component. It is a rerun-safe regime selector:
choose between PF-heavy, artifact-heavy, and conservative/anchor-like shapes
using only features available from the hidden test prefix.

The selector must be trained and tested on pseudo-hidden train wells. It should
answer, per hidden well:

- Should artifact correction be suppressed, normal, or amplified?
- Should PF be trusted or pulled toward anchor/hold?
- Is the post-start trend likely smooth, flat, or strongly curved?

## Next Experiment

Build a pseudo-hidden selector table over train wells:

1. Generate candidate predictions for many masked train wells:
   - PF h0.17 / 80-20 style;
   - PF/artifact 75-25 style where artifact features are available;
   - PF-only h0.17;
   - anchor/hold;
   - artifact-only or no-exact artifact where feasible.
2. Record prefix-only diagnostics:
   - known fraction, eval length, Z span;
   - prefix GR variance and drift;
   - PF path entropy/std/log-likelihood;
   - PF-artifact gap summary;
   - last-known slope and curvature.
3. Train a constrained well-level selector:
   - target is best candidate per pseudo-hidden well;
   - optimize per-well RMSE first, row RMSE second;
   - prefer simple thresholds or shallow trees for Kaggle stability.
4. Submit only if pseudo-hidden CV shows regime selection beats both fixed
   80/20 and 75/25, not merely by row average but by well-level consistency.

## Practical Submission Direction

The next submission should be a selector notebook, not a new blend scalar. If
artifact OOF generation is too expensive, start with a PF-only selector between
PF h0.17, anchor, and PF/hold variants. If that cannot beat h0.17 locally, do
not submit.
