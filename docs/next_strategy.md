# Next Public LB Strategy

The next phase is Public LB focused. That means test-well-specific analysis is allowed, but every move must be explicit about overfitting risk.

## Objective

Beat the v13_reg Public LB of 12.269 without returning to broad, untracked experiment churn. Treat all existing branches as weak until proven otherwise; none should be assumed active or promising just because it exists.

## Test Wells

| Well | Known profile from diagnostics | Strategy angle |
| --- | --- | --- |
| `000d7d20` | Test-train analog looked relatively flat; post-Z net about +100 ft; hard-well comparison suggests it may punish over-correction. | Keep v13/anchor behavior as a conservative reference. Test only small smooth trend changes. |
| `00bbac68` | More complex profile; post-Z net about +176.5 ft; spatial/neighbor ideas may matter most here. | Prioritize spatial/formation-dip correction experiments and compare against v13 trend. |
| `00e12e8b` | High pre-GR variance and lower post-GR variance; post-Z net about +144.8 ft. | Use GR-derived signals carefully; prefer robust, smoothed corrections over local GR matching jumps. |

## Next Experiments

1. Reconstruct v13 predictions for all three test wells and save a compact per-well trend table:
   - start TVT, end TVT, net change, min, max, standard deviation;
   - correction mean/std/range;
   - comparison to anchor and anchored physics.
2. Compare v13 trends against v23 spatial-neighbor inference:
   - only accept v23 influence where it changes the intended target well;
   - avoid global replacement if it worsens flat-looking behavior.
3. Build manual blend candidates by well, not by row:
   - `000d7d20`: v13-heavy or anchor-heavy.
   - `00bbac68`: test v13/v23 spatial blend.
   - `00e12e8b`: test v13 plus small smoothed GR-deviation adjustment.
4. Submit only candidates that have a written expected outcome:
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
