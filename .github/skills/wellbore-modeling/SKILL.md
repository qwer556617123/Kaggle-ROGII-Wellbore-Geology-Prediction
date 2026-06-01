---
name: wellbore-modeling
description: "ML and deep learning modeling for TVT prediction in ROGII Wellbore Geology Prediction. Use when training models, feature engineering GR logs, implementing DTW correlation, LightGBM regression, LSTM sequence prediction, or evaluating TVT RMSE on validation wells."
argument-hint: "Optional: model type to use (dtw | lgbm | lstm | transformer | ensemble)"
---

# Wellbore Modeling Skill

## When to Use
- Implementing TVT prediction pipeline
- Feature engineering from GR logs (sliding windows, DTW features)
- Training LightGBM, LSTM, or Transformer models
- Evaluating RMSE on held-out validation wells
- Tuning hyperparameters for local GPU

## Approaches (in order of complexity)

### 1. DTW Baseline (No GPU, ~minutes)
Match horizontal well GR to typewell GR using Dynamic Time Warping.
- Align horizontal GR (projected on known TVT) with typewell GR
- Extend the alignment forward to predict post-PS TVT
- See [dtw reference](./references/dtw-approach.md)

### 2. LightGBM Regression (CPU/GPU, ~minutes)
Tabular features per MD point → predict TVT.
- See [lgbm reference](./references/lgbm-features.md)

### 3. LSTM Sequence Model (GPU, ~hours)
Sequence of GR values → sequence of dTVT values.
- See [lstm reference](./references/lstm-approach.md)

### 4. Transformer / Attention Model (GPU, ~hours)
Cross-attention between horizontal GR and typewell GR.
- See [transformer reference](./references/transformer-approach.md)

## Standard Pipeline

```python
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

def get_all_well_ids(split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return sorted(set(
        f.stem.split("__")[0]
        for f in d.glob("*__horizontal_well.csv")
    ))

def load_well(well_id, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(d / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{well_id}__typewell.csv")
    return hw, tw

def get_ps_index(hw):
    """Row index of first empty TVT_input (Prediction Start)."""
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str) == "")
    return int(mask.idxmax()) if mask.any() else len(hw)
```

## Feature Engineering (LightGBM)

```python
def make_features(hw, tw, ps_idx):
    """
    Create per-row features for the horizontal well.
    For post-PS rows: use GR windows + typewell GR similarity.
    """
    gr = hw["GR"].values
    n = len(hw)
    feats = pd.DataFrame(index=hw.index)
    
    # Raw signals
    feats["gr"] = gr
    feats["md"] = hw["MD"].astype(float)
    feats["z"] = hw["Z"].astype(float)
    
    # Sliding window stats (window=21 points)
    W = 21
    gr_series = pd.Series(gr).fillna(method="ffill").fillna(method="bfill")
    for w in [5, 11, 21, 51]:
        feats[f"gr_mean_{w}"] = gr_series.rolling(w, center=True, min_periods=1).mean()
        feats[f"gr_std_{w}"]  = gr_series.rolling(w, center=True, min_periods=1).std()
    
    # Position in typewell space (last known TVT as anchor)
    known_tvt = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = known_tvt.ffill().values
    feats["last_known_tvt"] = last_known_tvt
    feats["rows_since_ps"] = np.maximum(0, np.arange(n) - ps_idx)
    
    return feats
```

## Validation Strategy

```python
# Well-level split — NEVER row-level split
all_ids = get_all_well_ids("train")
np.random.seed(42)
val_ids = set(np.random.choice(all_ids, size=int(len(all_ids)*0.15), replace=False))
train_ids = [i for i in all_ids if i not in val_ids]
```

## Evaluation

```python
def evaluate_rmse(predictions: dict, well_ids: list) -> float:
    """
    predictions: {well_id: np.array of predicted TVT for post-PS rows}
    """
    errors = []
    for wid in well_ids:
        hw, _ = load_well(wid)
        ps = get_ps_index(hw)
        truth = hw.iloc[ps:]["TVT"].astype(float).values
        pred  = predictions[wid]
        errors.extend((truth - pred).tolist())
    return np.sqrt(np.mean(np.array(errors)**2))
```

## Output Paths
```python
MODELS_DIR      = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
for d in [MODELS_DIR, PREDICTIONS_DIR, SUBMISSIONS_DIR]:
    d.mkdir(exist_ok=True)
```
