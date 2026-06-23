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

Use `scripts/diagnostics/make_lb_probe_submission.py` to generate candidate CSVs from an existing base output.

## First Probe Pair

Recommended first pair:

- `00e12e8b` constant `+10 ft`
- `00e12e8b` constant `-10 ft`

Reason: prior artifact-weight probes moved `00e12e8b` downward and got worse, so the next question is whether the base is systematically too low on that well.

If this pair shows signal, convert the score delta into an estimated optimal offset and submit the calibrated offset. If not, move to `00bbac68` linear trend, then `000d7d20` linear trend.

## Guardrails

- Always submit symmetric pairs for probing.
- Do not mix multiple new bases in one probe submission.
- Keep all probe outputs tied to their base submission hash.
- Treat this as Public-LB optimization; document overfit risk explicitly.
