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

1. Wait for the pending PF-family notebook submissions:
   - version 9: `grid_s3_b0_h0p22_fast`; bracket above current best.
   - version 10: `grid_s3_b0_h0p18_fast`; bracket below current best.
2. Runtime is now a first-class constraint:
   - Kaggle GPU is currently off, but this NumPy PF code would not benefit meaningfully from enabling it.
   - Do not re-enable 256 seeds / 500 particles unless the hidden rerun timeout is solved.
   - Skip beam computation whenever beam weight is zero.
3. Prioritize fixed PF selector refinement:
   - `grid_s3_b0_h0p2_fast` is the current best LB at 8.564.
   - Per-bin selector looked slightly better locally but lost on LB, so treat it as secondary.
   - Stress-test fixed h0.18-h0.24 on random and hard selections before broader submission.
4. Expand only around the confirmed region:
   - fixed scale 3 with hold 0.10-0.25 and zero beam;
   - code-specific variants for test-like codes 0, 2, and 3;
   - keep beam weight near zero unless hard/random CV proves otherwise.
5. If v7/v8 still time out, lower to 32 seeds / 128 particles before considering a GPU rewrite.
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
