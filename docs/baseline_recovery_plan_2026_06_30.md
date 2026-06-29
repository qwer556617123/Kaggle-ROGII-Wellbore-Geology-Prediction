# Baseline Recovery Plan - 2026-06-30

## Why This Exists

Version 37 ran the current `rogii-pf-artifact-blend` wrapper with no dynamic
offset and scored Public LB `8.230`. That is too close to v35/v36 (`8.247` and
`8.236`) and too far from v16/v21 (`8.130` and `8.131`). The working baseline
has drifted.

Do not spend more submissions on residual probes until a no-offset notebook
again reproduces the 8.13 family.

## Current Interpretation

- v35/v36 rejected the max-gap constant-offset basis.
- v37 rejected the active wrapper as the base prediction for further probing.
- The old "no-exact" labels are not sufficient, because artifact exact-overlap
  behavior changed at both the wrapper-env and component-inference handoff
  layers.
- The next target is not a better model; it is a reproducible historical model.

## Recovery Hypotheses

1. The v16/v21 submissions depended on pre-fix artifact exact-overlap handoff
   behavior, where saved component config could override wrapper-level
   `ROGII_EXACT_OVERLAP`.
2. The current `component_default` path is not identical to the old v16/v21
   path, even if its high-level settings look similar.
3. A component source/config/hash mismatch is more likely than a small blend
   weight issue, because many nearby scalar and per-well blend probes have
   already failed.

## Required Local Audit Before The Next Submission

Build a compact audit table for v16, v21, v22, and v37:

- wrapper git commit and notebook version;
- `PF_VARIANT`, `PF_WEIGHT`, and artifact weight;
- wrapper-level `ARTIFACT_EXACT_OVERLAP` default;
- artifact component inference behavior for `ROGII_EXACT_OVERLAP`;
- PF component hash;
- artifact component hash;
- final submission hash when available.

Then inspect the exact code transition around the artifact handoff fix and
identify the closest reproducible candidate to v16/v21.

## Next Submission Candidate

Submit only one baseline-recovery audit after the local table is complete:

`legacy_exact_handoff_80_20`

- PF: `grid_s3_b0_h0p17`;
- PF/artifact weight: `80/20`;
- selector: off;
- contact weight: `0`;
- dynamic final offset: off;
- artifact exact behavior: reproduce the historical v16-era handoff as closely
  as possible, especially whether saved component config can override the
  wrapper env.

Acceptance gate:

- if this returns near `8.13`, resume residual probing from this recovered base;
- if this returns near `8.23`, the missing behavior is not only the exact handoff
  and the next audit must move to component source/config hashes;
- if it is worse than `8.23`, revert the recovery candidate and do not stack new
  changes on it.
