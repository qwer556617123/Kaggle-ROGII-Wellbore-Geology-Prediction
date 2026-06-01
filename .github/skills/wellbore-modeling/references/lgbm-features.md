# LightGBM Feature Engineering

## Feature Groups

### 1. GR Rolling Statistics
```python
for window in [5, 11, 21, 51, 101]:
    gr_mean_{window}   # rolling mean
    gr_std_{window}    # rolling std
    gr_min_{window}    # rolling min
    gr_max_{window}    # rolling max
    gr_range_{window}  # max - min
```

### 2. Depth Features
```python
md             # Measured depth
z              # Z coordinate (negative = depth)
dz_dmd         # dZ/dMD (inclination proxy)
dx_dmd         # dX/dMD
dy_dmd         # dY/dMD
last_known_tvt # Last TVT_input before current row (forward-filled)
rows_since_ps  # Rows elapsed since Prediction Start
```

### 3. Typewell GR Matching (Critical)
```python
# For each row, find the typewell GR value closest to current GR
# within a TVT search window around last_known_tvt
tw_matched_tvt        # Best-matching TVT in typewell
tw_matched_gr_diff    # GR difference at best match
tw_match_rank         # Rank among top-5 typewell GR matches
```

### 4. Delta Features
```python
dgr_1    # GR[i] - GR[i-1]
dgr_5    # GR[i] - GR[i-5]
dtvt_input_1  # TVT_input[i] - TVT_input[i-1] (in known region)
tvt_trend_10  # Linear trend of last 10 known TVT values
```

### 5. Neighbor Well Features (Advanced)
```python
# Find N nearest wells by X,Y at same MD
# Use their TVT values as spatial context
neighbor_tvt_mean_3   # Mean TVT of 3 nearest wells
neighbor_tvt_std_3    # Std TVT of 3 nearest wells
neighbor_gr_corr      # GR correlation with nearest neighbor
```

## Training Config (LightGBM)

```python
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "device": "gpu",       # RTX 3060 Ti
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "seed": 42,
}
```

## Target Engineering

Predict **dTVT** (delta from last known TVT) rather than raw TVT:
```python
# Target: TVT[i] - last_known_TVT
# This normalizes scale differences between wells
target = hw["TVT"].astype(float) - hw["TVT_input"].astype(float).ffill()
```
