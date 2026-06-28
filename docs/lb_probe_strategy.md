# Public LB Probe Strategy

The current PF/artifact family is stuck around Public LB 8.13. Public notebooks explain this 8.x family, but not a 6.x jump. The next strategy is to use the Public LB score itself as a low-dimensional residual signal.

## Rationale

For a base prediction `p0`, a perturbation basis `b`, and score `S(a)` for `p0 + a*b`:

```text
S(a)^2 - S(-a)^2 = 4a * mean(b * (p0 - y))
```

So a symmetric `+a` / `-a` submission pair estimates whether the base residual points with or against that basis. This is much more informative than submitting another uncalibrated model family.

## Probe Bases

Start with interpretable, low-risk bases:

- per-well constant offset: one well shifted up/down by a fixed number of feet;
- per-well linear trend: zero-mean ramp, changing start/end trend without moving the well average;
- later, if useful, add two-piece trend basis split at mid measured depth.

Do not use fixed public-test CSVs for this competition. Rerun-safe probes must
be applied inside the Kaggle notebook after it recomputes predictions for the
current hidden sample.

## First Probe Pair Result

First rerun-safe pair:

- version 35: max PF/artifact component-gap hidden well, constant `+10 ft`,
  Public LB `8.247`, ref `54118191`;
- version 36: same basis, constant `-10 ft`, Public LB `8.236`, ref
  `54118269`.

The direction weakly favors lowering TVT on the max-gap well, but the symmetric
score gap is only `0.011`, below the `0.02` signal gate. Both probes are also
well worse than the 8.13 baseline. Do not submit a calibrated offset from this
pair.

## Reassessment After v35/v36

The important result is not the tiny preference for `-10`; it is that a blunt
constant offset on the max-gap well is the wrong basis. The symmetric projection
estimate is only about `0.0045`, while the average squared-score penalty versus
the 8.130 baseline is about `1.83`. In plain terms, this probe mostly added
variance and only found a weak direction.

This pair also exposes an audit gap: v35/v36 used the current wrapper default
with artifact `component_default`, while the historical v16/v21 labels have
known exact-overlap ambiguity. Before spending more residual probes, run one
no-offset audit version from the current wrapper and compare it against 8.13.

Decision tree:

1. If the no-offset audit returns near 8.13, then v35/v36 reject only the
   constant-offset basis. Move to a zero-mean shape basis, preferably a smooth
   linear or two-piece trend on the same rerun-safe max-gap well.
2. If the no-offset audit returns near 8.23-8.25, then the active wrapper
   baseline no longer matches the historical best. Stop probing and first
   recover the exact v16/v21 behavior from component config and hashes.
3. Do not submit calibrated offsets from v35/v36, and do not test more scalar
   constant offsets on the same basis.

## Guardrails

- Always submit symmetric pairs for probing.
- Do not mix multiple new bases in one probe submission.
- Keep all probe outputs tied to their base submission hash.
- Treat this as Public-LB optimization; document overfit risk explicitly.
- Do not use a fixed public-test submission as a Kaggle notebook input. This competition
  reruns notebooks with a substituted test set, so fixed public IDs can silently miss
  the rerun sample and produce invalid fallback predictions.
