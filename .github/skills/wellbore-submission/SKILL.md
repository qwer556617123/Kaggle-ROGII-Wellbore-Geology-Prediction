---
name: wellbore-submission
description: "Create Kaggle submission for ROGII Wellbore Geology Prediction. Use when generating a submission CSV from TVT predictions, validating submission format, or checking prediction coverage for test wells 000d7d20, 00bbac68, 00e12e8b."
argument-hint: "Path to predictions file or model name (e.g. 'predictions/lgbm_v1.pkl')"
---

# Wellbore Submission Skill

## When to Use
- Converting model predictions into a Kaggle-format submission CSV
- Validating submission completeness (all required row IDs covered)
- Comparing multiple submission files
- Post-processing predictions (clipping, smoothing)

## Test Wells
- `000d7d20` — 3836 rows to predict
- `00bbac68` — check sample_submission.csv for row count
- `00e12e8b` — check sample_submission.csv for row count
- **Total submission rows**: 14151

## Submission Format
```
id,tvt
000d7d20_1442,<float>
000d7d20_1443,<float>
...
```

## Creating a Submission

```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TEST_DIR  = DATA_DIR / "test"
SUBS_DIR  = DATA_DIR / "submissions"
SUBS_DIR.mkdir(exist_ok=True)

def make_submission(predictions: dict, name: str):
    """
    predictions: {well_id: np.array of TVT values for post-PS rows (in order)}
    name: submission filename stem (e.g. 'lgbm_v1')
    """
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    for well_id, pred_tvt in predictions.items():
        # Get row indices that correspond to this well's missing TVT_input rows
        hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
        empty_mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str) == "")
        row_indices = hw.index[empty_mask].tolist()
        
        assert len(row_indices) == len(pred_tvt), \
            f"Length mismatch for {well_id}: {len(row_indices)} vs {len(pred_tvt)}"
        
        ids = [f"{well_id}_{i}" for i in row_indices]
        pred_series = pd.Series(pred_tvt, index=ids)
        sub.loc[sub["id"].isin(ids), "tvt"] = sub.loc[sub["id"].isin(ids), "id"].map(pred_series)
    
    # Validate
    assert sub["tvt"].notna().all(), "Submission has NaN values!"
    
    out_path = SUBS_DIR / f"{name}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(sub)} rows)")
    return out_path

def validate_submission(path: str):
    """Check submission file for issues."""
    sub = pd.read_csv(path)
    ref = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    missing = set(ref["id"]) - set(sub["id"])
    extra   = set(sub["id"]) - set(ref["id"])
    nan_count = sub["tvt"].isna().sum()
    
    print(f"Rows: {len(sub)} / {len(ref)}")
    print(f"Missing IDs: {len(missing)}")
    print(f"Extra IDs:   {len(extra)}")
    print(f"NaN values:  {nan_count}")
    print(f"TVT range:   [{sub['tvt'].min():.2f}, {sub['tvt'].max():.2f}]")
```

## Post-processing (Optional)

```python
def smooth_predictions(tvt_array, window=5):
    """Apply light Gaussian smoothing to reduce jitter."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(tvt_array.astype(float), sigma=window/4)

def clip_tvt_range(tvt_array, tw_tvt):
    """Clip TVT to typewell range + small buffer."""
    lo, hi = tw_tvt.min() - 50, tw_tvt.max() + 50
    return np.clip(tvt_array, lo, hi)
```
