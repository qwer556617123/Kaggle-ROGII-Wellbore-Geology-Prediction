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
- v38 scored `8.279` with true wrapper-level no-exact 80/20. This rejects the
  immediate recovery candidate and shows that the current artifact source plus
  forced no-exact is worse than the v37 component-default path.
- The old "no-exact" labels are not sufficient, because artifact exact-overlap
  behavior changed at both the wrapper-env and component-inference handoff
  layers.
- The next target is not a better model; it is a reproducible historical model.
- The downloaded v37 `blend_summary.json` shows PF hash
  `936993cfb9ee3cc8112cc4e752fc89369dbe401eceb86836d537ba0ccebc485e`, matching
  the v22 audit PF hash, but artifact hash
  `f67409151654218cd50ecd31c4ca6cc11ae12cf8af6bcbceb06b97dee3c5e693`, differing
  from the v22 audit artifact hash
  `92c5bae92dfc257c3e6b93ba425cb01cfb2b0ecbc15f74bc48957ff4f97b18a3`.
- Since v37 used `component_default`, it effectively tested the saved-config
  exact-on path and scored 8.230. The v38 audit then forced true no-exact 80/20
  and scored 8.279, so wrapper-level exact handoff is not the missing recovery
  switch.
- After v38, the better diagnosis is source-level drift. Historical v16-v31
  wrappers embedded v10 artifact source hash `51c2409f...`; v38 embedded
  `81d15d7d...`. The only substantive source differences are the sklearn RMSE
  fallback and the inference-only exact-overlap env handoff. The next audit
  should restore the legacy `51c2409f...` artifact source and then rerun the
  no-offset 80/20 baseline.

## Recovery Hypotheses

1. The v16/v21 submissions depended on true no-exact 80/20 behavior, but the
   only durable no-exact 80/20 output, v22, was never Public-LB submitted.
2. The current `component_default` path is not identical to the old v16/v21
   path, and v37 shows it is not good enough.
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

`legacy_source_80_20_recovery`

- PF: `grid_s3_b0_h0p17`;
- PF/artifact weight: `80/20`;
- selector: off;
- contact weight: `0`;
- dynamic final offset: off;
- artifact source: restore historical v10 source hash `51c2409f...`;
- artifact exact behavior: preserve the legacy inference handoff rather than
  forcing the new wrapper override.

Acceptance gate:

- if this returns near `8.13`, resume residual probing from this recovered base;
- if this returns near `8.23`, legacy source plus saved-config behavior is close
  but still not exact, so inspect PF source/runtime and artifact dataset config;
- if it is worse than `8.23`, revert the recovery candidate and do not stack new
  changes on it.
