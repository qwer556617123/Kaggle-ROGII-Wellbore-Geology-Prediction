# Script Layout

This directory is the planned home for project scripts once local move approval is available.

## Intended Categories

- `main/`: stable training, inference, and Kaggle kernel scripts.
- `experiments/`: exploratory model and feature scripts.
- `diagnostics/`: evaluation, hard-well analysis, and inspection helpers.

See `docs/file_inventory.md` for the current root-level file classification and the planned move map.

## Stable Root Entrypoints

Until the move is completed, these root-level commands remain canonical:

```powershell
python lgbm_final_reg_train.py
python run_lgbm_on_test_csv.py
```

After the move, keep root-level wrappers for both commands so existing workflow and notes do not break.
