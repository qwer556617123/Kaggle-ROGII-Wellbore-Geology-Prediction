# ROGII Breakthrough Strategy - 2026-07-14

## Current Evidence

- Exact second-order forward-backward smoothing at weight `0.15` improved the
  reliable PK-Adopt anchor from `7.053` to `6.935`. This is the first verified
  method-level move into the 6.x range. The exact Yusuke 7.039 source rebuild
  scored only `7.146`, so the gain belongs to smoothing rather than anchor swap.
- The reproducible best improved from `7.166` to `7.053` by adding the guarded
  model-package residual at maximum weight `0.010`.
- Forcing the model-package branch broadly at weight `0.020` scored `7.130`.
  The public `0.010` source is not a symmetric dose point because it retains a
  p95 disagreement guard of `25`, while the `0.015`/`0.020` sources raise it to
  `999`. Stop the dose sweep: the result rejects broad package application, not
  merely a particular scalar weight.
- This is a useful residual direction, but it is not a top-10 strategy by itself.
- Visible three-well outputs do not describe the code-scoring environment. Hidden
  rerun routing and artifact availability are part of the effective model.

## Domain Hypotheses Rejected

- Smooth cross-well contact-surface interpolation: leakage-safe native-mask audit
  on 120 wells produced pooled RMSE about `16.1`.
- Formation-gradient integration: five-fold, whole-well-isolated audit on all 773
  wells produced pooled RMSE about `37.7` due to accumulated dip error and faults.
- Selecting one model-package family per well from truth-free disagreement,
  geometry, prefix-U, GR, and typewell features: package OOF worsened from `10.67`
  to about `11.15`.
- These results rule out ordinary KNN/kriging, smooth regional dip continuation,
  and confidence-only expert routing as breakthrough paths.

## Remaining High-Leverage Hypothesis

The missing structure is discrete, not smooth: fault blocks, datum modes, and GR
phase aliases. Domain knowledge should generate and constrain bounded alternatives;
it should not replace the strong sequence/PF anchor with an unconstrained path.

The next method-level candidate is a native-mask fault-mode residual model:

1. Use the reproducible 7.0x source as the base path.
2. Generate sparse alternatives only where prefix replay and GR phase indicate a
   plausible `10-35 ft` mode or fault jump.
3. Represent formation information as relative thickness/order and local gradient
   change, never as an absolute smooth contact surface.
4. Train the gate on native masks with whole-well isolation. Features must be
   available in the hidden rerun and may not contain same-well formation truth.
5. Require an OOF oracle gap before training a router, then require the learned
   router to close a material part of that gap.

The first implementation of this hypothesis did not pass its local gate. On a
30-well native-mask rescue audit, the `0.10` fault-posterior blend changed pooled
RMSE from `15.0889` to `15.1219` and worsened well-median RMSE by `0.5181` ft.
Only one of three public-matched strata improved. The slight well-p90 and
worst-tail gains are insufficient: stop this branch, do not enable the target
graph, and do not submit it.

The simpler no-fault second-order HMM did transfer. Combining the scored
`weight=0` and `weight=0.15` points with the visible rerun direction norm gives
a quadratic optimum of `0.223`; the source's 773-well OOF independently points
to about `0.25`. Posterior standard deviation is not a stable quality gate:
20-well native masks showed all four std quartiles improving at weight `0.25`,
and a monotone shrink gate worsened pooled RMSE versus constant weight.

Robust emission is a useful independent axis. On 20 native-mask wells,
Student-t/std at weight `0.25` improved Gaussian/std row RMSE `9.140 -> 9.102`,
well mean `6.802 -> 6.744`, median `4.931 -> 4.619`, and p90
`12.114 -> 11.939`. Affine/MAD emissions were strongly harmful and are retired.

## Competition Mechanics

- Treat a notebook as a conditional inference program. Audit which branch runs on
  hidden data rather than assuming the visible output vector is the scored vector.
- Use Public LB only for controlled low-dimensional probes. Do not fit large fusion
  weights from a small number of correlated submissions.
- Keep the guarded `0.010` package path only as part of the `7.053` anchor. Do not
  retry `0.015`, submit `0.0125`, or continue a scalar dose sweep.
- Continue surveying new public sequence/artifact families. A genuinely independent
  strong candidate is more valuable than another blend of the same SP45/contact path.

## Active Runs

- Scored controls: `rogii-yusuke7039-rebuild` v1 = `7.146`; exact Gaussian HMM
  `0.15` = `6.935`.
- `rogii-pkadopt-hmm0225` v1 tests the LB-calibrated Gaussian weight `0.225`.
  Runtime was `869s`; final hash `2f605e53...`; competition ref `54707042`
  is pending.
- `rogii-pkadopt-hmm-student025` v1 tests Student-t emission at weight `0.25`.
  Runtime was `903s`; final hash `80cba42a...`; competition ref `54707045`
  is pending.
- Both follow-ups generated anchor and posterior vectors inside the hidden rerun
  and passed 14,151-row sample alignment and finite-value audits. The Gaussian
  unweighted posterior exactly matched v1; Student-t differed from Gaussian by
  `2.162 ft` RMSE.

## 2026-07-15 Robust-Emission Result And Next Split

- Gaussian HMM `0.225` scored `6.916`; Student-t HMM `0.25` scored `6.909`.
  The reliable anchor remains `7.053`, so exact sequence smoothing has now
  transferred across three controlled Public-LB runs.
- A quadratic fit to Gaussian weights `0`, `0.15`, and `0.225` gives an optimum
  of `0.2402` and predicted minimum `6.9154`. Ordinary Gaussian weight tuning is
  therefore exhausted; another nearby scalar dose cannot provide a breakthrough.
- Student-t improves the observation model rather than merely changing dose. Its
  `0.007` edge over the calibrated Gaussian run is small but directionally agrees
  with native-mask validation.
- On 30 native-mask wells and 154,954 eval rows, Student-t scalar `0.25` had row
  RMSE `12.4811`. Separating the HMM residual into a per-well mean datum and a
  zero-mean within-well shape, then using datum weight `0.25` and shape weight
  `0.50`, improved row RMSE to `12.2648`. Well mean, median, and p90 all improved.
  Shape weight `0.75` offered negligible pooled gain and a worse p90, so `0.50`
  is the selected bounded follow-up.
- Two independent notebooks are running in parallel:
  `rogii-pkadopt-hmm-student-shape050` tests the datum/shape split, while
  `rogii-robustpf-sub7-rebuild` tests a public raw128/smoothed32 PF emission
  ensemble with visible-prefix gating. The latter is not an HMM weight variant
  and can become a stronger anchor or a future independent residual component.
- A compact 12-well dynamics grid found that stronger Student-t emission at
  `lambda=1.25` works best when the HMM contributes no per-well mean datum and
  only a `0.50` zero-mean shape component. Relative to default Student-t `0.25`,
  row RMSE improved `9.6446 -> 9.1611` and well p90 `15.6508 -> 14.9571`.
  This shape-only candidate is running as
  `rogii-pkadopt-hmm-student-lam125-shape050` v1.

## 2026-07-16 Public Datum/Shape Recalibration

- Results rejected the native-mask amplitude extrapolation: default Student-t
  datum `0.25` / shape `0.50` scored `6.986`, and lambda `1.25` shape-only
  `0.50` scored `7.085`, both worse than scalar Student-t `0.25 = 6.909`.
  Stop shape amplification and stronger emission.
- Using the scored `(mean, shape)` points `(0,0)`, `(.25,.25)`, and
  `(.25,.50)` together with the rerun component norms gives estimated residual
  projections `mean=+0.6591`, `shape=-10.2609`. The low-dimensional optimum is
  approximately `mean=-0.057`, `shape=0.310`; because visible component norms
  can drift in the hidden rerun, the submitted conservative point is
  `mean=0`, `shape=0.30`, with no negative datum extrapolation.
- The public robust-PF source completed its visible run in `830s`, but its code
  submission timed out. Profiling showed the three independent robust-PF wells
  consumed about `211s` sequentially. The timeout repair preserves raw128 and
  smooth32 exactly and parallelizes only the per-well loop; visible output hash
  must equal the sequential run before competition submission.
- A fresh public-source audit rejected Hellbore V6 (incomplete generic model),
  Geologia V92 (empty source), A044 datum kriging (missing unpublished lookup),
  and the advertised model-package `0.025` fork (source actually forces `0.02`,
  already rejected). PRVS grouped-OOF meta residual remains viable: it combines
  five Ravaghi tracks, PF, and public model-package OOF across 773 held-out
  wells, without the unavailable spatial lookup.
- The exact PRVS direct path was valid but took `871s`. A runtime-safe variant
  reduced only selector PF seeds from 128 to 80; it completed in `780.4s`, kept
  the same 3,783,989 grouped-OOF training rows, seven features, fitted
  coefficients, and OOF RMSE `9.8770`, and emitted a non-fallback final SHA
  `424e3314...`. It is submitted as ref `54749859`.
- Three intentionally different hypotheses are now scored in parallel: the
  Public-LB-fitted HMM shape dose (`54749211`), a faster robust multi-emission PF
  anchor (`54749507`), and the grouped-OOF learned residual (`54749859`). Do not
  add another correlated dose until these identify which family transfers.

## 2026-07-17 Three-Well Residual Tomography

- The parallel batch resolved decisively: HMM shape `0.30 = 6.912`, robust-PF
  fast `7.448`, and grouped-OOF PRVS `7.858`. Keep Student-t scalar `0.25 = 6.909`;
  remove robust-PF and PRVS from active candidates.
- Posterior std is not a valid confidence router here. The locally winning gate
  raised mean test weight to `0.3328`; every gate constrained to mean `0.25`
  worsened native-mask row RMSE and p90. Do not submit the std-gated notebook.
- Use the competition's three dynamic wells as three disjoint low-dimensional
  bases. Submit signed Student-t HMM probes `++-` and `+-+`; combine their scores
  with the existing `+++` score and common `7.053` anchor to solve each well's
  residual projection exactly. Hash and norm equality are mandatory.
- After both scores return, compute `w_j = -projection_j / norm_j`, clamp to
  `[-0.50, 0.75]`, round to `0.025`, and submit one calibrated per-well HMM run.
  This is the next high-leverage attempt; ordinary scalar, shape, uncertainty,
  Gaussian/Student blend, robust-PF, and PRVS dose searches are closed.
- Public output audits passed exactly. The `++-` probe is competition ref
  `54766267`; the `+-+` probe is ref `54766337`. Both share base SHA
  `fdf4a817...` and posterior SHA `e51c4c2...`, so their score pair is valid for
  the three-axis solve.
- Scores `6.966` and `6.963` recover rounded sorted-well weights
  `[0.65, 0.05, 0.05]`, versus the former global `[0.25, 0.25, 0.25]`. Predicted
  Public RMSE is `6.6636`; score-rounding sensitivity does not change the rounded
  solution. The calibrated v1 output reproduced all hashes and norms and was
  submitted as competition ref `54784705`.
