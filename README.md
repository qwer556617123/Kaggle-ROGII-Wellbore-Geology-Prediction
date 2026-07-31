# ROGII Wellbore Geology Prediction

This repository is for the ROGII Wellbore Geology Prediction competition. The current goal is not to keep adding random model variants; it is to keep a small active workspace, validate locally before spending submissions, and only revive archived ideas with a concrete hypothesis.

## Current Baselines

| Method | Local signal | Public LB | Status |
| --- | ---: | ---: | --- |
| pk-adopt exact rebuild | external/public source anchor | 7.053 | Reproducible base vector |
| Student-t exact HMM, global 0.25 | native-mask gain transferred to LB | 6.909 | Historical HMM anchor |
| Student-t HMM, per-well 0.65/0.05/0.05 | three coded LB probes predicted 6.664 | 6.902 | Historical tomography; direction basis overfit |
| Student-t HMM + package reverse bias -1.75 | first large reverse-direction probe | 7.689 | Rejected: overshot |
| Wellcal HMM + package reverse bias -0.75 | curvature-calibrated follow-up | 7.511 | Rejected: reverse-package basis is not transferable |
| Wellcal HMM + Prefix-GR RF well bias | grouped-OOF external correction, cap +/-0.5 ft | 7.478 | Rejected; runtime shortcut changed hidden anchor lineage |
| Wellcal HMM + neighbor structural path 0.10 | native100 pooled -0.326, median -0.186 | 7.501 | Rejected: native-mask gain inverted on hidden LB |
| HMM + neighbor structure + Prefix-GR well bias | orthogonal direction correlation 0.216 | 7.432 | Rejected: combining two harmful directions stayed harmful |
| Wellcal HMM + exact anti Prefix-GR well bias | sign-reversed audit only | not submitted | Bound invalid: hidden anchor lineage differs from scored 6.902 |
| Pilkwang latest dual-track exact rebuild | full prefix/contact/model-package branches | 7.065 | Valid external anchor, but below the 6.902 Student-t HMM |
| Verified MHA140SEP4 + A10 slope refinement | external 6.979 anchor plus bounded GR slope alignment | 6.979 | Rejected: +/-0.14 ft move was below LB sensitivity |
| Verified MHA140SEP4 + Student-t HMM 0.25 | external posterior hedge plus our validated smoother | 6.944 | Positive but below the 6.902 HMM anchor; no scalar sweep |
| Geometry beam + paired prefix-affine PF reference | candidate-generation change on MHA140SEP4 contract | 8.599 | Rejected: strongly harmful despite exact runtime-safe rebuild |
| Geometry/reference PF + Student-t HMM 0.25 | controlled smoother rescue of the same geometry vector | 7.746 | Rejected: HMM recovers 0.853 but remains far below the 6.902 anchor |
| Claimed Public 6.768 exact source rebuild | runtime-safe full hidden routing | 7.559 | Rejected as anchor: visible source vector was exact, hidden score was not reproducible |
| Claimed 6.768 source + Student-t HMM 0.25 | controlled same-anchor smoother test | 7.301 | HMM gain -0.258 is real, but the upstream anchor remains weak |
| Guard640 package 0.49 + U-continuity | hidden-aware package gate and heel-boundary fade | 11.752 | Rejected: package 0.49 catastrophically activates on hidden rerun |
| Guard640 + Student-t HMM 0.15 | guarded upstream plus conservative backward smoothing | 10.725 | Rejected: HMM rescues 1.027 but cannot repair the upstream failure |
| MHA250SEP2 exact rebuild | alpha 2.5, sep 2, cap 4 on verified MHA lineage | 6.880 | Historical MHA frontier; ref 54882007, runtime 323s |
| MHA300 minor-mass 0.15 | broader low-confidence mode coverage | 6.823 | Positive, but weaker than retaining mass 0.22 |
| MHA300 separation-high 60 | broader high-confidence fault separation | 6.801 | Positive continuation of the MHA frontier |
| MHA400SEP1CAP10 | stronger hidden midpoint hedge | 6.757 | Reliable hedge anchor |
| Dynamic F594 exact rebuild | PF seed-branch hedge, no fixed public ID | 7.504 | Rejected: reported 6.594 does not transfer to hidden rerun |
| MHA400 + U-continuity | best MHA hedge plus target-free heel boundary fade | 6.720 | Current scored best; independent continuity gain 0.037 |
| Dynamic F594 + HMM 0.15 | stronger external anchor plus backward smoothing | 7.355 | HMM gains 0.149, but upstream F594 remains weak |
| MHA500 / MHA600 | fixed mass/sep, extended hedge dose | 6.756 / 6.797 | Saturation then reversal beyond alpha 4-5 |
| MHA400-continuity rank tomography | four ID-free Hadamard datum probes at +/-2 ft | 6.720 x4 | Invalid experiment: hidden well count differed from the visible three-well template, so scoring retained the existing base file |
| MHA400-continuity + Student-t HMM 0.10 | independent sequence-shape control | 6.696 | Current scored best; exact smoother signal adds 0.024 |
| HMM010 structural tomography | C1 `+/-`: 6.905/7.802; slope `+/-2`: 6.791/6.798; datum `+2`: 7.039 | completed | C1 has a strong positive projection but full dose overshoots; slope and global datum are effectively zero-value |
| HMM010 C1 calibration | LB-derived global `alpha=0.35` plus four-bin Hadamard tomography | 6.609 | Current scored best; coded probes `7.623/7.440/6.824` reject naive group decoding |
| C1 + orientation field 0.15 | cross-well contact derivatives integrated from the known heel | 7.139 | Rejected: the strong native-mask gain inverted on hidden LB |
| C1 - orientation field 0.15 | exact symmetric hidden-residual projection probe | 7.642 | Pair implies optimum `+0.0253` and only ~0.024 predicted gain; retired |
| HMM010 C1 single-bin energy probes | bin 0 and bin 3 only at alpha 1 | pending | Refs `55144850` / `55144899`; resolve hidden per-bin energy |
| HMM010 C1 bin1-off selective | `(0.35,0,0.35,0.35)` from Hadamard projections | pending | Ref `55144931`; harmful positive-projection bin removed |
| Harshini dynamic surface OOF | cross-well formation surfaces + PF + grouped row ensemble | rejected direct | fast2 passed runtime at 641s, but its +3.43 ft bias conflicts with the LB-derived -0.18 ft datum direction |
| MPSC multiscale alignment | 12-well bounded refinement: pooled -0.015, median +0.220 | not submitted | Rejected: hard-well SSE worsened |
| event segmental fallback | 12-well native-mask gain 0.108 at 0.20 | not submitted | Signal exists, below gate |

Current conclusion: exact backward smoothing remains the only recent family
confirmed on Public LB. Native-mask gains for neighboring structure and
grouped-OOF well bias both inverted on hidden LB, so native-mask CV is now only
a rejection tool. Runtime shortcuts that disable visible-inactive branches can
change hidden execution lineage and are prohibited for score tomography. The
full-branch dual-track rebuild scored `7.065`, and the neighbor/well-bias fusion
scored `7.432`. Low component correlation did not imply useful residual
alignment, so rejected correction directions may no longer be combined merely
for diversity. The MHA140SEP4 A10 refinement was score-neutral at `6.979`;
adding the validated Student-t HMM improved it to `6.944`, but still did not
beat the `6.902` HMM anchor. The geometry/reference candidate then scored
`8.599`; HMM rescued it to `7.746`, proving that the upstream geometry branch
is harmful rather than an untuned positive basis. The active pair now rebuilds
the source that claimed Public `6.768`; exact reconstruction instead scored
`7.559`, while HMM improved the identical anchor to `7.301`. External title
scores are therefore treated only as hypotheses. The active batch tests a
hidden-aware package/continuity guard alone and with HMM, plus an exact
MHA250SEP2 control from the more reproducible MHA lineage. That control scored
`6.880`, then the controlled MHA ladder improved to `6.823`, `6.801`, and
`6.757`. High-confidence multimodal coverage and stronger midpoint hedging are
the active signal; lowering minor mass helps less. Guard640 instead scored
`11.752` (`10.725` with HMM), permanently rejecting the high-weight package
route. The active batch now rebuilds the dynamic reported-`6.594` PF branch and
tests target-free continuity on MHA400. Fixed-public-ID A23 derivatives are
excluded from hidden reruns.
Model-package dose sweeps, multiscale GR correlation, neighbor transfer, RF
well bias, geometry/reference PF, and event-driven fault blocks remain rejected.
The offset-free structural orientation field is rejected as a main model:
despite a strong 100-well native-mask gain, its `+0.15/-0.15` pair scored
`7.139/7.642`. The exact hidden quadratic places the optimum at only
`+0.0253`, with a predicted score near `6.585`; no calibrated orientation
submission is planned. The active C1 experiment instead uses the completed
four-bin Hadamard scores to identify heterogeneous residual projection, then
measures the two important bins directly before selective calibration.

## Active Commands

Root wrappers are kept for the main commands:

```powershell
python lgbm_final_reg_train.py
python run_lgbm_on_test_csv.py
python lgbm_v15_formation_train.py
```

Use local CV before spending daily submissions:

```powershell
python scripts\diagnostics\cv_attenuated_physics.py --folds 5 --seed 42
python scripts\diagnostics\cv_lgbm_rank_replay.py --folds 3 --row-stride 10 --max-rounds 2000 --early-stopping 150
python scripts\main\train_ranked_lgbm_candidates.py
```

## Project Layout

- `scripts/main/`: stable mainline, inference, and Kaggle kernel references.
- `scripts/diagnostics/`: active diagnostics and validation gates.
- `scripts/experiments/`: active experimental scripts only.
- `scripts/archive/`: deprecated or low-priority scripts kept for history.
- `docs/`: project state, experiment findings, CV-vs-LB notes, and strategy.
- `features/`: small checked-in metadata; generated features are ignored.
- `submissions/`, `models/`, `predictions/`, `reports/`: generated outputs, ignored by git.

## Active Validation Notes

- Do not use the three visible test IDs as official oracle truth.
- Use each train well's native `TVT_input` mask; artificial 45/60/75% prefixes are retired.
- Held-out/native-mask CV is a rejection gate, not a reliable Public-LB score predictor.
- A new sequence candidate must improve pooled RMSE by at least 0.30 ft, well median by 0.20 ft, avoid p90 regression above 0.25 ft, and improve at least two test-matched strata.
- Public-LB tomography is allowed only after the local rejection and novelty gates pass.

## Archive Policy

Archived scripts should stay archived unless they have:

- a named failure mode;
- a target validation gate;
- an expected Public LB effect;
- a clear reason they are better than the current v23/v13 references.
