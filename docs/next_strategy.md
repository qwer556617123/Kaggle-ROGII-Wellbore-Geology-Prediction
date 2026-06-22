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
   - Versions 8, 9, and 10 scored 8.258, 8.204, and 8.271 at 15%, 20%, and 17.5% artifact. The best known point is now 80/20, and the LB trend still favors more artifact.
   - Versions 11 and 12 scored 8.231 and 8.322 at 25% and 30% artifact. The upward sweep is now over; 30% is clearly too much artifact.
   - Version 13 scored 8.269 at 21% artifact, and versions 14/15 scored 8.275/8.229 when only `00bbac68` was pushed to 25%/30% artifact. Stop scalar and simple per-well artifact weight tuning.
   - Component ablation worked: version 16 disabled v10 artifact's exact train-coordinate overlap and improved to 8.130, while version 17 reduced the exact blend to 0.10 and scored 8.199. Treat no-exact artifact as the new default.
   - The no-exact weight bracket did not produce a clean curve: 85/15, 82.5/17.5, 77.5/22.5, and 75/25 scored 8.235, 8.262, 8.275, and 8.131. Because 77.5/22.5 is anomalously worse than both neighboring 80/20 and 75/25, audit notebook outputs before spending the remaining quota.
   - Current best remains version 16 at 8.130, with version 21 effectively tied at 8.131. Treat no-exact 80/20 as the default unless audit proves a version mismatch.
   - Add durable `blend_summary.json` output to future notebook versions so PF/artifact weights, exact-overlap settings, and component hashes are auditable after Kaggle reruns.
   - Version 22 completed as an audit-only rerun: `blend_summary.json` confirms no-exact 80/20, empty per-well overrides, PF component hash `936993cfb9ee3cc8112cc4e752fc89369dbe401eceb86836d537ba0ccebc485e`, artifact component hash `92c5bae92dfc257c3e6b93ba425cb01cfb2b0ecbc15f74bc48957ff4f97b18a3`, and submission hash `7e6a4305c420ab4e38a9a8afafcf81b6320b1c4f8e46af01c6dd6c4adb863717`.
   - The next larger change is to keep no-exact 80/20, but run the v10 artifact stack with TabICL enabled on Kaggle GPU. The first push attempt was blocked by Kaggle's 30-hour weekly GPU quota, so keep the notebook default CPU/no-TabICL until quota resets.
   - While GPU is blocked, the only CPU-safe probe worth spending on is per-well no-exact weighting. The v22 component audit shows `00e12e8b` has the largest PF-artifact disagreement, so test `00e12e8b` at 75/25 and 70/30 while keeping the other wells at 80/20.
   - After submitting per-well probes, restore the notebook default to the known best no-exact 80/20 until a pending score proves otherwise.
   - Versions 23 and 24 scored 8.233 and 8.210 for `00e12e8b` at 75/25 and 70/30. This rejects e12-only artifact weight tuning; the 75/25 global near-tie either comes from another well, rerun variance, or a coupled effect that simple per-well weighting is not capturing.
   - Do not spend more submissions on scalar or one-well PF/artifact weights unless a new diagnostic identifies a different mechanism.
   - Next CPU-safe component probe: keep no-exact artifact 80/20 but swap the PF component from fixed h0.17 to `uncertainty_selector`, testing dynamic hold/beam behavior without changing blend weight.
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
