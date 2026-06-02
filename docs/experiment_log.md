# Experiment Log

This log keeps the project from drifting into random trial-and-error. Every experiment should be tied to a specific hypothesis, a target well or failure mode, and a measurable outcome.

## Baseline

| Version | Category | Main idea | Local result | Public LB | Keep going? |
| --- | --- | --- | --- | --- | --- |
| v13_reg / `lgbm_final_reg_train.py` | Best baseline | Anchored physics plus LightGBM correction with stronger regularization | RMSE 15.08 row-weighted, 11.95 per-well mean | 12.269 | Yes, benchmark |

## Useful Clues

| Version | Category | Main idea | Result / note | Keep going? |
| --- | --- | --- | --- | --- |
| v8 | Useful baseline | Anchored physics plus post-PS trajectory features | RMSE about 15.17; foundation for v13 | Yes, as reference |
| v15 | Weak branch | Formation top features | Not good enough; keep as organized reference only | No, unless a new targeted hypothesis revives it |
| v6 | Weak historical branch | GR matching features | Not good enough; no standalone v6 script in this checkout | No, historical context only |
| v20 | Useful clue | GR/xcorr-derived corrections and hard-well diagnostics | More diagnostic value than score value | Selectively |
| v21 | Useful clue | GR-to-TVT features | Did not clearly beat v13 | Only if tied to a test-well hypothesis |
| v22-v23 | Useful clue | Spatial neighbor / 3D correction features | Interesting for test-well-specific trend correction; not stable enough as mainline | Selectively |
| v24 | Useful clue | GR deviation rolling and trend features | Hypothesis useful, score not yet proven | Selectively |
| `hard_well_analysis.csv` | Diagnostic | Compares hard validation wells with test wells | Useful for matching target failure modes | Yes |

## Low Priority Branches

| Version / script | Why low priority |
| --- | --- |
| v1-v4 style dTVT models | Accumulation error and weaker validation range. |
| early `lgbm_train.py` / generated parquet flow | Superseded by anchored-physics correction models. |
| broad LSTM attempts | Higher complexity without a proven score path. |
| generic DTW-only approaches | Risky because test wells can have decreasing or near-flat TVT behavior. |
| one-off submission generators | Useful for comparisons, but not a strategy by themselves. |

## Required Entry For New Experiments

Use this template before running or submitting a new idea:

```text
Experiment:
Target well(s):
Hypothesis:
Expected TVT trend change:
Expected delta versus v13:
Validation check:
Public LB result:
Decision:
```

## Current Open Questions

- What failure mode explains why all current approaches remain poor?
- Does `00bbac68` require a stronger spatial or formation-dip correction than v13 predicts?
- Is `000d7d20` close to a flat/anchor-like regime where aggressive corrections hurt?
- Can `00e12e8b` use GR deviation safely without overreacting to high pre-GR variance?
