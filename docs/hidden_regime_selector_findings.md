# Hidden Regime Selector Findings - 2026-06-27

## Result

The prefix-only selector harness is implemented, but it did not pass the local
submission gate.

| Run | Candidate pool | Baseline | Selector OOF | Decision |
| --- | --- | ---: | ---: | --- |
| smoke | PF variants + anchor | 5.037 well RMSE | 6.853 well RMSE | Reject; anchor labels do not generalize |
| pf-only | PF variants only | 5.037 well RMSE | 5.108 well RMSE | Reject; worse than fixed h0.17 |

The gate required the selector to improve fixed h0.17 by at least 0.15 well
RMSE without hurting row RMSE. Both runs failed, so no Kaggle submission should
be made from `prefix_regime_v1` yet.

## Interpretation

The best local pseudo-hidden PF candidate is still fixed `grid_s3_b0_h0p17`.
This supports the current Public LB baseline, but it also means a selector over
PF hold/scale variants is not enough to escape the 8.13 plateau.

The missing validation piece is artifact OOF. Public LB says the artifact stack
adds value, but the current local selector can only compare PF-family shapes.
Without pseudo-hidden artifact predictions, a rule that changes PF/artifact
weight per hidden well is mostly speculation.

## Next Action

Build an artifact pseudo-hidden harness:

- create a temporary competition-like dataset for one masked train well as
  `test`;
- exclude that well from artifact training inputs where feasible;
- run the v10 artifact component in inference mode for a small 12-well smoke;
- compare fixed 80/20, fixed 75/25, and prefix-selected per-well weights using
  true hidden tails.

Keep `ROGII_BLEND_SELECTOR=off` by default until that harness shows a stable
local gain.
