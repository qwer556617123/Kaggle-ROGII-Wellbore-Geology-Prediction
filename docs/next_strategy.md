# Next Public LB Strategy

The next phase is Public LB focused. That means test-well-specific analysis is allowed, but every move must be explicit about overfitting risk.

## Objective

Beat the v13_reg Public LB of 12.269 without returning to broad, untracked experiment churn. Treat all existing branches as weak until proven otherwise; none should be assumed active or promising just because it exists.

The current best public clue is PF/physical modeling rather than another global LGBM feature sweep. Local PF CV is now the primary gate for this branch: `grid_s3_b0_h0p2` beat `public_selector` both locally and on LB, so this validation is useful for ranking PF-selector variants.

## Test Wells

| Well | Known profile from diagnostics | Strategy angle |
| --- | --- | --- |
| `000d7d20` | Test-train analog looked relatively flat; post-Z net about +100 ft; hard-well comparison suggests it may punish over-correction. | Keep v13/anchor behavior as a conservative reference. Test only small smooth trend changes. |
| `00bbac68` | More complex profile; post-Z net about +176.5 ft; spatial/neighbor ideas may matter most here. | Prioritize spatial/formation-dip correction experiments and compare against v13 trend. |
| `00e12e8b` | High pre-GR variance and lower post-GR variance; post-Z net about +144.8 ft. | Use GR-derived signals carefully; prefer robust, smoothed corrections over local GR matching jumps. |

## Next Experiments

1. Treat `grid_s3_b0_h0p17_fast` as the current PF baseline:
   - version 11 reached Public LB 8.534.
   - The gain over h0.18 is real but small, so do not let this pull the project back into endless hold micro-tuning.
2. Runtime is now a first-class constraint:
   - Kaggle GPU is currently off, but this NumPy PF code would not benefit meaningfully from enabling it.
   - Do not re-enable 256 seeds / 500 particles unless the hidden rerun timeout is solved.
   - Skip beam computation whenever beam weight is zero.
3. Treat fixed-hold PF as the new baseline, not the next breakthrough:
   - `grid_s3_b0_h0p17_fast` is the current best LB at 8.534.
   - h0.22 was worse and per-bin was worse on LB, so more hold/selector micro-tuning is low leverage.
   - Keep h0.18 as the default unless a larger method beats it locally and on LB.
4. Shift to larger method changes:
   - Artifact-stack blending: public v10/v11 artifact datasets are visible, but the kojimar helper dataset is not discoverable by dataset search. Next step is a custom inference wrapper/feature builder, not a static CSV blend.
   - `rogii-pf-artifact-blend` now embeds PF h0.17 and v10 artifact inference in one script. Version 5 uses a no-TabICL fallback and 95/5 PF/artifact blend, and improved Public LB to 8.415.
   - Version 6 improved to 8.331 with a 90/10 PF/artifact blend, while version 7 fell to 8.473 with 97.5/2.5. The artifact component is useful, and 2.5% is too conservative.
   - Versions 8 and 9 test 85/15 and 80/20. A quadratic fit to Public LB MSE at 2.5%, 5%, and 10% artifact estimates the optimum near 17% artifact, so version 10 submits 82.5/17.5 as a curve-guided interpolation. If 15%, 17.5%, or 20% beats 8.331, continue around the best point with one midpoint; if all are worse, bracket locally around 90/10 with 92.5/7.5 and 87.5/12.5.
   - Learned PF/meta-selector: generate multiple PF candidates per well and train a local meta-model to choose/blend them using pseudo-hidden wells, instead of hand-coded bins.
   - Full-path ensemble search: sample PF trajectories, score them with global GR/event/shape criteria, and average only the best path families; do this carefully because the first path-rerank attempt was too brittle.
   - Formation/contact residual correction: the first row-level residual model was worse than h0.18, so only revisit with strong regularization or well-level corrections.
5. Avoid another small submission unless it tests one of the larger changes above.
6. Reconstruct v13 predictions for all three test wells and save a compact per-well trend table:
   - start TVT, end TVT, net change, min, max, standard deviation;
   - correction mean/std/range;
   - comparison to anchor and anchored physics.
7. Compare v13 trends against v23 spatial-neighbor inference:
   - only accept v23 influence where it changes the intended target well;
   - avoid global replacement if it worsens flat-looking behavior.
8. Build manual blend candidates by well, not by row:
   - `000d7d20`: v13-heavy or anchor-heavy.
   - `00bbac68`: test v13/v23 spatial blend.
   - `00e12e8b`: test v13 plus small smoothed GR-deviation adjustment.
9. Submit only candidates that have a written expected outcome:
   - which well should improve;
   - whether TVT range should widen, shrink, rise, or fall;
   - how much the prediction differs from v13.

## Guardrails

- Do not start a new feature family unless it targets a named failure mode.
- Do not compare only aggregate validation if the intended target is one of the three Public LB wells.
- Do not replace v13 globally unless the new method explains why all three test wells benefit.
- Keep every submission CSV tied to an experiment-log entry.

## Immediate File Organization Follow-Up

The physical script layout is organized under `scripts/`. Keep root-level wrappers for stable reference commands so existing notes and shell history stay usable.
