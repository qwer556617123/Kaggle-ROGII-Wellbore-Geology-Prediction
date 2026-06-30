# Next Public LB Strategy

## 2026-06-25 Plateau Reset

The PF/artifact branch is no longer an open-ended tuning space. Versions 16/21
remain the best at about 8.13, while TabICL CUDA v28 scored 8.211 and dynamic
high-gap weighting v29 scored 8.266. Treat blend weights, public well IDs, and
fixed-base probes as exhausted.

Update 2026-06-27: the v30/v31/v32 probes confirmed the plateau. A small
positive contact-surface blend scored 8.160, the conservative v16/v21 ensemble
scored 8.153, and the negative contact-surface residual scored 8.408. Do not
continue same-family blending or contact-surface residuals. Reassess from
`docs/best_versions_reassessment_2026_06_27.md`.

Update 2026-06-28: the first rerun-safe residual probe pair was informative but
not actionable. Max PF/artifact-gap hidden well constant offsets scored 8.247
for +10 ft and 8.236 for -10 ft. The direction weakly favors lowering TVT, but
the pair is below the 0.02 signal gate and both scores are far worse than the
8.13 family. Treat constant offsets on the max-gap well as rejected. The new
priority is to audit the current no-offset wrapper baseline before any further
probe, because v35/v36 used `component_default` artifact behavior while older
v16/v21 labels have exact-overlap ambiguity.

Update 2026-06-30: the no-offset audit returned Public LB 8.230 as notebook
version 37. This confirms baseline drift: the active wrapper default is not the
8.13 v16/v21 family. Stop residual probes, selectors, and shape bases until the
exact historical wrapper/component behavior is recovered. The next submission
must be a controlled baseline-recovery audit, not a new modeling idea.

Follow-up 2026-06-30: downloaded v37 output confirms PF hash matches the v22
audit, while the artifact hash differs. v37 used `component_default`, so it
tested the saved-config exact-on path and failed to recover 8.13. The next
controlled audit is therefore true no-exact 80/20 with
`ROGII_ARTIFACT_EXACT_OVERLAP=0`, dynamic offset off, selector off, and contact
weight 0.

Second follow-up 2026-06-30: v38 true no-exact 80/20 scored Public LB 8.279,
worse than v37. This rejects the wrapper-level exact override as the recovery
switch. The artifact source itself drifted: historical v16-v31 used embedded
v10 artifact source hash `51c2409f...`, while v38 used `81d15d7d...`. The next
controlled audit is legacy artifact source `51c2409f...`, PF/artifact 80/20,
selector off, contact off, dynamic offsets off. Do not submit contact-shape
probes until this source-level baseline is tested.

This competition reruns notebooks with substituted hidden test data, so local
public test well IDs are not the Public LB target. Future work should focus on
hidden-rerun-style CV over train wells and learned path/meta-selection. See
`docs/plateau_reset_2026_06_25.md`.

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
   - The TabICL/GPU probe was blocked by Kaggle runtime: metadata accepted T4x2, but torch inside the selected image was CPU-only and failed at `device="cuda"`. Do not retry TabICL until the docker image/runtime mismatch is fixed.
   - Fallback CPU-safe component probe: keep no-exact artifact 80/20 but swap fixed h0.17 PF for `bin_lb_safe`, testing per-bin PF meta-selection without changing blend weight.
   - Versions 25 and 27 scored 8.373 and 8.282, so PF selector swaps are rejected. The public notebooks explain the current 8.x family, but do not by themselves explain a 6.x score.
   - GPU quota reset path: retry TabICL using the public kojimar CUDA docker image (`gcr.io/kaggle-private-byod/python@sha256:57e612b...`) and `NvidiaTeslaT4`, because the prior v26 failure was a torch/CUDA image mismatch rather than a modeling result.
   - The no-offset audit v37 scored 8.230, so the active wrapper baseline is
     not the historical 8.13 baseline. Public LB residual probing is paused
     until this is fixed.
   - Do not continue symmetric zero-mean shape probes yet. Their interpretation
     would be tied to the wrong base prediction.
   - Recover the exact v16/v21 baseline behavior first. v37 rejected the new
     source component-default path, and v38 rejected the new source true-noexact
     path. The immediate candidate is legacy artifact source `51c2409f...` with
     the historical inference handoff.
   - Learned PF/meta-selector: generate multiple PF candidates per well and train a local meta-model to choose/blend them using pseudo-hidden wells, instead of hand-coded bins.
   - Full-path ensemble search: sample PF trajectories, score them with global GR/event/shape criteria, and average only the best path families; do this carefully because the first path-rerank attempt was too brittle.
   - Formation/contact residual correction: the first row-level residual model was worse than h0.18, so only revisit with strong regularization or well-level corrections.
   - Contact geometry diagnostic reset: actual train formation contact columns plus a known-segment offset reconstruct TVT almost exactly (`~0.006 ft` local RMSE), but hidden/test files do not expose those columns. The practical task is therefore reconstructing missing contact surfaces from train X/Y before applying the offset formula.
   - First contact-surface results: KNN surface imputation is usable but not yet best (`EGFDL_k64_tail_mean` smoke row RMSE `9.087`); LightGBM surface extrapolation is rejected (`20.336`). Submit one KNN surface probe only to measure hidden LB signal, not as a claimed best model.
5. Avoid another small submission unless it tests one of the larger changes above.
   - Current larger change: build a rerun-safe regime selector that chooses
     PF-heavy, artifact-heavy, or conservative/anchor-like behavior from
     prefix-only diagnostics. Scalar PF/artifact weights are no longer a valid
     experiment class.
   - The first local selector harness is implemented, but the smoke tests
     rejected PF-only selection: fixed h0.17 scored 5.037 well RMSE, while the
     OOF selector scored 5.108 without anchor and 6.853 with anchor. Keep
     `ROGII_BLEND_SELECTOR=off` until artifact OOF is available.
   - Next implementation target: create pseudo-hidden artifact predictions for
     masked train wells, then validate per-well 80/20 vs 75/25 vs selector
     weights against true hidden tails. This is the first selector test that
     actually matches the Public LB blend mechanism.
   - The artifact hidden-mask harness was built and suggested true no-exact
     70/30 or 75/25, but Kaggle rejected both: v33 scored 8.391 and v34 scored
     8.328. Treat that harness as a diagnostic only, not a submit gate. Restore
     the practical v16-like baseline: 80/20 with the artifact component's saved
     config controlling exact-coordinate behavior.
   - New interpretation: prior "no-exact" labels were partly misleading because
     the wrapper-level env could be overridden inside the artifact component
     before the 2026-06-27 fix. True no-exact is now empirically worse.
   - The v35/v36 max-gap constant-offset probe added another rejection: simple
     per-well TVT offsets are too blunt even when the well is chosen
     rerun-safely from component disagreement. The next probe, if any, must be
     shape-aware and preceded by a recovered 8.13 no-offset audit.
   - The v37 no-offset audit scored 8.230, proving the current wrapper is not
     the right base for probing. The immediate task is baseline recovery, not
     more residual probing.
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
