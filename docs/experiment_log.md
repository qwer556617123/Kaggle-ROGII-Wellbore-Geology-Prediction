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

## PF-Family Public Notebook Track

These experiments are inspired by the public physical/PF notebooks. Local CV uses train wells as pseudo-hidden wells by scoring the original `TVT_input` tail against `TVT`; the default cut below is 36 `lb_like` wells, 24 PF seeds, and 120 particles.

| Experiment | Main idea | Local result | Public LB | Decision |
| --- | --- | --- | --- | --- |
| `public_selector` | Reproduce the PF scale / beam / hold selector style as a stable baseline | row RMSE 7.255, per-well mean 6.252 | 8.844, ref 53344707 | PF baseline; CV/LB alignment confirmed |
| `grid_s3_b0_h0p2` | Fixed PF scale 3 with 0 beam and 0.20 anchor hold | row RMSE 7.134, per-well mean 6.098 | 8.752, ref 53344688 | Best known LB; local rank matched LB |
| `uncertainty_selector` | Dynamic hold/beam weights from PF uncertainty and GR missingness | row RMSE 7.618, per-well mean 6.326 | not submitted | Hold; did not beat public selector locally |
| `path_rerank` | PF path library reranked by full-sequence GR score | row RMSE 11.573, per-well mean 7.793; one well failed badly | 9.099, ref 53339812 | Worse than selector; stop |
| `event_beam` | Event-weighted beam plus PF blend | smoke row RMSE 12.184 on 18 wells | not submitted | Drop for now |
| `bin_less_aggressive` | Per-bin selector from 100-well CV: code0 s3/h0.10, code2 s8/h0.10, code3 s3/h0.15, code5 s12/h0.15 | row RMSE 10.368, per-well mean 7.732 on 100 `lb_like` wells | timeout/no score, ref 53361269 | Too slow at 256 seeds / 500 particles |
| `grid_s3_b0_h0p15` | Fixed PF scale 3 with 0 beam and 0.15 hold | row RMSE 10.389, per-well mean 7.821 on 100 `lb_like` wells | timeout/no score, ref 53361314 | Too slow at 256 seeds / 500 particles |
| `bin_less_aggressive_fast` | Same per-bin selector, but 64 seeds / 160 particles and skip beam when weight is zero | row RMSE 7.844, per-well mean 6.368 on 36 `lb_like` wells | pending, ref 53511157 | Submitted to avoid hidden rerun timeout |
| `grid_s3_b0_h0p2_fast` | Same fixed selector as current best LB, but 64 seeds / 160 particles and zero-beam shortcut | row RMSE 7.860, per-well mean 6.571 on 36 `lb_like` wells | pending, ref 53511214 | Submitted as fast safety version |

Top selector-grid rows are recorded in `docs/pf_selector_grid_summary.csv`; per-well detail is in `docs/pf_selector_grid_details.csv`.
The larger 100-well selector comparison is recorded in `docs/pf_bin_selector_cv_summary.csv` and `docs/pf_bin_selector_cv_details.csv`.
The fast runtime check is recorded in `docs/pf_fast_selector_cv_summary.csv` and `docs/pf_fast_selector_cv_details.csv`.

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
- Does PF-style CV align better with LB than prior LGBM CV? The public-selector submission will calibrate this.
- Does `00bbac68` require a stronger spatial or formation-dip correction than v13 predicts?
- Is `000d7d20` close to a flat/anchor-like regime where aggressive corrections hurt?
- Can `00e12e8b` use GR deviation safely without overreacting to high pre-GR variance?
