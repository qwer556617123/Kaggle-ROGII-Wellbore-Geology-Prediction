# ROGII Wellbore Geology Prediction

## Git & Version Control

This project uses local git (initialized at repo root). **Proactively commit after milestone achievements:**
- New model variant with improved val RMSE
- New feature set or significant experiment
- New submission generated
- Bug fixes to training/inference scripts
- Updates to copilot-instructions.md

Use semantic commit messages: `feat:`, `fix:`, `chore:`, `exp:` (experiment)  
Always include `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.

## Competition Overview

Kaggle competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction  
**Task**: Predict TVT (True Vertical Thickness) for horizontal well points beyond the Prediction Start (PS) point  
**Metric**: RMSE of (manualTVT - predictedTVT)

## Domain Knowledge

**TVT (True Vertical Thickness)**: The true vertical depth in the geological formation reference frame. As a horizontal well drills through layers, TVT increases when drilling deeper, decreases when shallowing, or stays constant when drilling horizontally within a single layer.

**GR (Gamma Ray)**: Primary log measurement. Higher GR = shale, Lower GR = sand/limestone. The GR pattern in the horizontal well mirrors the typewell GR, making it the key feature for TVT correlation.

**Prediction Start (PS)**: The point where `TVT_input` becomes empty. The model must predict TVT from the PS point onward.

**Key Insight (from PPT Slide 9)**: The GR from the horizontal well section *before* PS point correlates better with deeper typewell GR than the typewell GR itself (due to higher resolution). Use pre-PS GR + known TVT as anchor for post-PS prediction.

**Geological Dip**: Nearby wells share similar dipping behavior. Offset wells can improve TVT predictions for the current well.

## Data Structure

```
E:\Code\ROGII - Wellbore Geology Prediction\
├── train\
│   ├── {id}.png                      # Visual reference image
│   ├── {id}__horizontal_well.csv     # MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input
│   └── {id}__typewell.csv            # TVT,GR,Geology
├── test\
│   ├── {id}__horizontal_well.csv     # MD,X,Y,Z,GR,TVT_input  (TVT column absent)
│   └── {id}__typewell.csv            # TVT,GR  (Geology column absent)
└── sample_submission.csv             # id ({well_id}_{row_index}), tvt
```

**Training wells**: ~773 unique IDs (3 files each)  
**Test wells**: 3 unique IDs  
**Submission rows**: only rows where `TVT_input` is empty (after PS point)  
**Submission ID format**: `{well_id}_{row_index_0based}` e.g. `000d7d20_1442`

## Geological Formation Columns (train horizontal_well only)

- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`: TVT depth of each formation top at that MD point
- `Geology` values in typewell: ANCC, ASTNL, ASTNU, BUDA, EGFDL, EGFDU, LBHL, LTGT, LTHL, MNSS

## Environment

- **Python**: 3.11.15 (conda env: `kaggle-dev` at `D:\anaconda\envs\kaggle-dev\`)
- **GPU**: NVIDIA GeForce RTX 3060 Ti, CUDA 11.8
- **PyTorch**: 2.7.1+cu118
- **LightGBM**: 4.6.0
- **Pandas**: 3.0.3, **NumPy**: latest
- Direct python: `D:\anaconda\envs\kaggle-dev\python.exe`
- No Kaggle compute available — local GPU only
- **pandas 3.0 gotcha**: use `.bfill()` / `.ffill()` not `fillna(method=...)`

## Best Model: LightGBM v13_reg / lgbm_final_reg (Direct Physics Correction)

**Key approach** (`lgbm_v8_train.py` / `lgbm_final_reg_train.py`):
- **Target**: `correction = true_tvt - anchored_physics` (not dTVT increments)
- **Anchored physics**: `anchor_tvt + slope × (Z - Z_anchor)` — always 0 at PS
- **v8 val RMSE**: 15.17 ft (row-weighted), 11.98 ft (per-well mean), 7983 rounds
- **v13_reg val RMSE**: **15.08 ft (row-weighted), 11.95 ft (per-well mean), 13077 rounds** ← BEST
- **Final submission**: `lgbm_final_reg.csv` — v13_reg params on all 773 wells

**v13_reg key params** (v8 params changed):
- `feature_fraction`: 0.8 → **0.7**
- `bagging_fraction`: 0.8 → **0.7**
- `min_child_samples`: 20 → **50**
- `lambda_l1`: 0.05 → **0.3**
- `lambda_l2`: 0.05 → **0.3**

**Top features** (in order):
1. `physics_vs_lkt` — anchored physics deviation from last known TVT (dominant, 10× others)
2. `physics_tvt_at_end` — total expected TVT change at well end
3. `rows_since_ps` — position in post-PS sequence
4. `total_z_change_post` — total Z excursion after PS
5. `post_dz_rate` — mean Z rate post-PS
6. `post_ps_frac` — fractional progress in post-PS section

**Val RMSE progression**:
| Version | Val RMSE | Notes |
|---------|----------|-------|
| v1-v4   | 17-19 ft | dTVT (increment) target — integration errors accumulate |
| **v5**  | **15.76 ft** | BREAKTHROUGH: direct correction target |
| v6      | 15.73 ft | +GR matching features (marginal) |
| v7      | 16.17 ft | +global GR stats (hurt) |
| **v8**  | **15.17 ft** | +anchored physics + post-PS trajectory features |
| v9-v12  | 15.25-15.40 ft | all worse (overfit) |
| **v13_reg** | **15.08 ft** | +stronger regularization (BEST) |

**Features that HURT** (do not add):
- Global post-PS GR statistics (v7) — overfitting
- `physics_attenuation`, `post_pre_dz_ratio` (v11) — hurt despite high importance
- GR matching correction search ±60 ft (v12) — marginal at best
- Huber loss / stronger regularization (v13) — worse

## Physics Model Details

Pre-PS linear fit: `TVT = slope × Z + intercept` (R² = 0.98-1.0)
- slope ≈ −1.0 for most wells
- **Critical**: use anchored form `anchor_tvt + slope × (Z − Z_anchor)`, NOT `intercept + slope × Z`
- Post-PS trajectory (Z, MD) is fully known at test time — use for global features

## Hard Wells (train val)
- `1b1eba53`: 66 ft RMSE — non-monotonic correction (±42 ft oscillation) due to complex geology
- `ba48188d`: 54 ft RMSE
- `389ae58f`: 41 ft RMSE — physics over-predicts (attenuation = 0.34, well became more horizontal)

- Always separate wells into train/validation split (not row-split — use held-out wells)
- The PS row index divides known TVT (TVT_input not empty) from prediction zone
- Feature windows: use sliding GR windows around each MD point
- Evaluate per-well RMSE and overall RMSE on validation set
- Save model checkpoints to `models/` directory
- Save predictions to `predictions/` directory, submissions to `submissions/`

## Code Style

- Use pandas for data loading, numpy for computation
- Use `pathlib.Path` for file paths, not string concatenation
- Type hints where practical
- Prefer LightGBM for tabular baselines; PyTorch LSTM/Transformer for sequence models
- Seed everything: `np.random.seed(42)`, `torch.manual_seed(42)`
