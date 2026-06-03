# Script Layout

## Intended Categories

- `main/`: stable training, inference, and Kaggle kernel references.
- `diagnostics/`: active evaluation and validation gates.
- `experiments/`: active experiments only.
- `archive/`: deprecated, weak, or historical scripts.

See `docs/file_inventory.md` for the current root-level file classification and archive policy.

## Stable Root Entrypoints

Until the move is completed, these root-level commands remain canonical:

```powershell
python lgbm_final_reg_train.py
python run_lgbm_on_test_csv.py
python lgbm_v15_formation_train.py
```

Root-level wrappers are kept for these commands so existing workflow and notes do not break.
