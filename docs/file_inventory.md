# File Inventory

Last updated: 2026-06-03

## Root

Root should stay small:

- `README.md`
- `lgbm_final_reg_train.py` wrapper
- `run_lgbm_on_test_csv.py` wrapper
- `lgbm_v15_formation_train.py` wrapper
- data folders: `train/`, `test/`, `features/`
- project folders: `docs/`, `scripts/`, `.github/`

## Active Scripts

### `scripts/main/`

Stable references and inference entrypoints:

- `lgbm_final_reg_train.py`
- `run_lgbm_on_test_csv.py`
- `lgbm_v15_formation_train.py`
- `train_ranked_lgbm_candidates.py`
- `kaggle_kernel_inference.py`
- `kaggle_kernel_v3.py`

### `scripts/diagnostics/`

Active diagnostics and validation gates:

- `cv_attenuated_physics.py`: held-out well CV for anchor/attenuation baselines.
- `evaluate_cv_lb_alignment.py`: quick check of whether local signals track Public LB.
- `cv_lgbm_rank_replay.py`: grouped pseudo-test replay for candidate rank checks.
- `hard_well_analysis.csv`: hard-well and visible test-well profile summary.

### `scripts/experiments/`

Active experiments kept visible because they are recent and directly tied to current strategy:

- `attenuated_physics_sweep.py`
- `gr_path_search.py`

## Archived Scripts

Deprecated or historical scripts live under `scripts/archive/`:

- `scripts/archive/experiments/`: old LGBM versions, feature engineering variants, LSTM, DTW, Viterbi.
- `scripts/archive/diagnostics/`: old eval scripts, one-off submission helpers, feature importance helpers, spatial analysis.
- `scripts/archive/helpers/`: reserved for archived utilities.

Archived scripts are preserved for history but should not be used by default.

## Output Policy

Generated outputs should not be committed:

- `models/`
- `submissions/`
- `predictions/`
- `reports/`
- `kernel_output*/`

If a result matters, summarize it in `docs/experiment_findings.md` or `docs/cv_vs_lb.md`.
