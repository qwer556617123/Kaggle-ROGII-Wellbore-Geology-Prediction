# Plateau Reset - 2026-06-25

## Current Read

The PF/artifact family is now a proven 8.x plateau:

- best known: no-exact PF/artifact 80/20, Public LB 8.130;
- no-exact 75/25 tied but did not improve, Public LB 8.131;
- TabICL on a CUDA image scored 8.211;
- dynamic high-gap well weighting scored 8.266;
- fixed public-test base probing is invalid for this code competition because Kaggle reruns notebooks with substituted hidden test data.

This means the missing 6.x signal is not another scalar PF/artifact blend weight, not another single-well public-ID correction, and not the current TabICL branch.

## Hard Stop

Do not spend more submissions on:

- scalar PF/artifact weights;
- public test well IDs;
- fixed public-test submission CSVs as notebook inputs;
- PF selector swaps around h0.17/h0.20;
- full TabICL reruns without a smaller ablation proving value.

## Correct Target

The notebook must generalize to hidden rerun wells. Local public test files are useful only for smoke tests and profile diagnostics. They are not the Public LB target.

The local validation target should become pseudo-hidden train wells with rerun-like masks:

- split by well, not row;
- score per-well RMSE and row RMSE;
- sample cut points to imitate hidden rerun prediction-start lengths;
- report rank alignment against the Kaggle submissions that are truly comparable.

## Next Large Moves

1. Hidden-rerun CV harness
   - Build pseudo-hidden masks from train wells instead of relying only on existing `TVT_input` masks.
   - Match distributions of known fraction, eval length, Z span, GR missingness, and tail GR variance.
   - Use this as a rejection gate before any new submission.

2. Learned path/meta-selector
   - Generate a matrix of candidate paths per pseudo-hidden well: PF scales, hold values, beam variants, anchor, formation/contact, artifact-style features where available.
   - Train a well-level selector or constrained stacker using only pre-PS features and candidate diagnostics.
   - Optimize per-well RMSE, with a penalty for unstable row-level jumps.

3. Shape-aware path scorer
   - Stop averaging paths only by online PF likelihood.
   - Score candidate paths by multi-scale GR alignment, smoothness, physical slope plausibility, contact consistency, and post-PS shape statistics.
   - Select or weight a small set of plausible complete trajectories.

4. Smaller model ablations
   - If using TabICL again, run one small branch only: one context size, one seed, no full A+B run.
   - Submit only if local pseudo-hidden CV shows a different error profile from the current artifact stack.

## Immediate Action

The initial hidden-rerun CV harness now lives at `scripts/diagnostics/evaluate_hidden_rerun_masks.py`.

Next, run a broader comparison over more wells and masks. Until that table exists, additional submissions are mostly blind. The next submitted notebook should be backed by a local pseudo-hidden rank table, not by another public-LB guess.
