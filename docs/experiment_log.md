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
| `bin_less_aggressive_fast` | Same per-bin selector, but 64 seeds / 160 particles and skip beam when weight is zero | row RMSE 7.844, per-well mean 6.368 on 36 `lb_like` wells | 8.578, ref 53511157 | Valid but slightly worse than fixed h0.20 |
| `grid_s3_b0_h0p2_fast` | Same fixed selector as current best LB, but 64 seeds / 160 particles and zero-beam shortcut | row RMSE 7.860, per-well mean 6.571 on 36 `lb_like` wells | 8.564, ref 53511214 | Best known LB; fixed selector beats per-bin on LB |
| `grid_s3_b0_h0p22_fast` | Hold sweep neighbor above h0.20 | row RMSE 7.861 on 36 `lb_like` wells | 8.598, ref 53545690 | Worse; higher hold is not promising |
| `grid_s3_b0_h0p18_fast` | Hold sweep neighbor below h0.20 | row RMSE 7.869 on 36 `lb_like` wells | 8.541, ref 53545708 | Valid but superseded by h0.17 |
| `grid_s3_b0_h0p17_fast` | Hold sweep neighbor below h0.18 | not separately scored in CV; bracket from LB trend | 8.534, ref 53576928 | Best known LB; h0.17 is current default |
| `pf_residual_correction` | Learn residual on top of h0.18 from row/well features | base row RMSE 7.869; best tested residual model was worse at 8.307+ on 36 wells | not submitted | Reject current design; residual correction overfits pseudo-hidden wells |
| `artifact_stack_probe` | Inspect public v10/v11 artifact datasets as a possible PF blend component | v11 artifact scores around 10.44 OOF; helper dataset for kojimar blend was not discoverable by Kaggle dataset search | not submitted | Possible future blend component, but needs custom inference wrapper/feature builder |
| `pf_artifact_blend_v5` | Dynamic PF h0.17 plus v10 artifact inference, no-TabICL fallback, 95/5 PF/artifact blend | Kaggle notebook version 5 completed; blend output verified as 5% artifact delta, no NaNs | 8.415, ref 53612723 | Best known LB; artifact signal is useful |
| `pf_artifact_blend_v6` | Same components as v5, 90/10 PF/artifact blend | Weight bracket above 5% artifact | 8.331, ref pending | Best known LB; artifact signal is strongest so far around 10% |
| `pf_artifact_blend_v7` | Same components as v5, 97.5/2.5 PF/artifact blend | Weight bracket below 5% artifact | 8.473, ref pending | Worse than 5% and 10%; too little artifact |
| `pf_artifact_blend_v8` | Same components as v5, 85/15 PF/artifact blend | Weight bracket above 10% artifact | 8.258 | Better than 90/10, but not best |
| `pf_artifact_blend_v9` | Same components as v5, 80/20 PF/artifact blend | Weight bracket above 15% artifact | 8.204 | Best known LB; artifact can carry at least 20% |
| `pf_artifact_blend_v10` | Same components as v5, 82.5/17.5 PF/artifact blend | Public-LB quadratic interpolation from 2.5%, 5%, and 10% artifact scores estimated optimum near 17% artifact | 8.271, ref 53625463 | Worse than 15% and 20%; curve estimate was too conservative |
| `pf_artifact_blend_v11` | Same components as v5, 75/25 PF/artifact blend | Public-LB quadratic fit through known 2.5%-20% artifact scores estimates optimum near 30% artifact | 8.231 | Worse than 80/20; 25% artifact is too high |
| `pf_artifact_blend_v12` | Same components as v5, 70/30 PF/artifact blend | Direct test near the quadratic optimum estimate | 8.322 | Clear over-blend; stop upward sweep |
| `pf_artifact_blend_v13` | Same components as v5, 79/21 PF/artifact blend | Local curve through 20/25/30 and 15/20/25 estimates optimum around 20.5%-21% artifact | 8.269, ref 53660371 | Worse than 80/20; stop scalar weight fine-tuning |
| `pf_artifact_blend_v14` | Scalar 80/20, but `00bbac68` uses 75/25 | Per-well probe: keep `00e12e8b` at 20% artifact to avoid over-lowering, raise only `00bbac68` | 8.275, ref 53660387 | Worse; this per-well artifact increase does not help |
| `pf_artifact_blend_v15` | Scalar 80/20, but `00bbac68` uses 70/30 | Stronger version of v14; tests whether `00bbac68` benefits from more artifact when `00e12e8b` is protected | 8.229, ref 53660388 | Worse; stop `00bbac68`-only artifact increase |
| `pf_artifact_blend_v16` | Same as v9 80/20, but v10 artifact exact overlap disabled | Component ablation: direct train lookup was poor, so test whether artifact exact-coordinate blend is hurting | 8.130, ref 53664697 | Best known LB; exact overlap is hurting this blend |
| `pf_artifact_blend_v17` | Same as v9 80/20, but v10 artifact exact overlap weight lowered to 0.10 | Softer component ablation: keep exact overlap but reduce default 0.28 strength | 8.199, ref 53664696 | Slight gain vs v9 but worse than disabling exact overlap |
| `pf_artifact_blend_v18` | No-exact artifact, 85/15 PF/artifact blend | Re-bracket the blend weight after removing exact overlap | 8.235, ref 53706983 | Worse than v16; too little artifact |
| `pf_artifact_blend_v19` | No-exact artifact, 82.5/17.5 PF/artifact blend | Re-bracket the blend weight after removing exact overlap | 8.262, ref 53706980 | Worse than v16; too little artifact |
| `pf_artifact_blend_v20` | No-exact artifact, 77.5/22.5 PF/artifact blend | Re-bracket the blend weight after removing exact overlap | 8.275, ref 53706981 | Anomalously worse than both 80/20 and 75/25; audit before trusting curve |
| `pf_artifact_blend_v21` | No-exact artifact, 75/25 PF/artifact blend | Wider right-side check for the no-exact artifact optimum | 8.131, ref 53706984 | Near-tie with v16; no clear improvement |
| `pf_artifact_blend_v22` | Audit rerun of no-exact 80/20 with durable output summary | Confirm the actual Kaggle-rerun settings and component hashes after the non-convex v18-v21 bracket | not submitted | Output verified: `artifact_exact_overlap=0`, PF/artifact 80/20, submission SHA256 `7e6a4305c420ab4e38a9a8afafcf81b6320b1c4f8e46af01c6dd6c4adb863717`; use as audit baseline, not a new score |
| `tabicl_gpu_attempt` | No-exact 80/20, but enable the artifact TabICL branch on Kaggle GPU | Test whether the artifact-stack component can change the residual structure rather than only changing scalar blend weights | blocked before run | Kaggle rejected GPU push because the 30-hour weekly GPU quota is exhausted; revisit after quota reset |
| `pf_artifact_blend_v23` | No-exact artifact; scalar 80/20 except `00e12e8b` at 75/25 | Component-gap audit shows PF is about 21 ft above artifact on `00e12e8b`, while the other two wells have much smaller component gaps | 8.233, ref 53747821 | Worse than v16/v21; e12-only artifact increase does not explain the global 75/25 near-tie |
| `pf_artifact_blend_v24` | No-exact artifact; scalar 80/20 except `00e12e8b` at 70/30 | Stronger e12-only version of v23 | 8.210, ref 53747937 | Still worse; stop e12-only artifact weight tuning |
| `pf_artifact_blend_v25` | No-exact artifact 80/20, but replace fixed h0.17 PF with `uncertainty_selector` PF | Test whether dynamic PF hold/beam uncertainty changes the three public wells in a useful way when artifact stack remains fixed | pending | CPU-safe new component probe; not a scalar weight tune |
| `pf_artifact_blend_v26` | Fixed h0.17 PF plus no-exact artifact 80/20, but enable TabICL on Kaggle GPU | Test whether the artifact stack's TabICL component changes residual structure enough to escape the 8.13 plateau | blocked before submission | Kaggle accepted GPU metadata but ran a CPU-only torch build; TabICL failed with `Torch not compiled with CUDA enabled` |
| `pf_artifact_blend_v27` | No-exact artifact 80/20, but replace fixed h0.17 PF with `bin_lb_safe` PF | Test a per-bin PF meta-selector as a second CPU-safe component change | pending | Fallback second strategy after TabICL/GPU failed in Kaggle runtime |

Top selector-grid rows are recorded in `docs/pf_selector_grid_summary.csv`; per-well detail is in `docs/pf_selector_grid_details.csv`.
The larger 100-well selector comparison is recorded in `docs/pf_bin_selector_cv_summary.csv` and `docs/pf_bin_selector_cv_details.csv`.
The fast runtime check is recorded in `docs/pf_fast_selector_cv_summary.csv` and `docs/pf_fast_selector_cv_details.csv`.
The fast hold sweep is recorded in `docs/pf_fast_hold_sweep_summary.csv` and `docs/pf_fast_hold_sweep_details.csv`.
The rejected residual correction check is recorded in `docs/pf_residual_summary.csv`.

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
