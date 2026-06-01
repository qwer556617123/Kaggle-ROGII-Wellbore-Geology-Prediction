# DTW (Dynamic Time Warping) Approach for TVT Prediction

## Core Idea

Use DTW to align the GR log from the horizontal well (projected onto TVT space) with the typewell GR log.
The DTW warping path gives a TVT → TVT mapping that extends beyond the PS point.

## Algorithm

```python
from dtaidistance import dtw_ndim
import numpy as np

def predict_tvt_dtw(hw, tw, ps_idx, search_window=200):
    """
    Predict TVT for post-PS rows using GR-DTW correlation.
    
    Strategy:
    1. Take pre-PS GR of horizontal well (higher quality reference)
    2. Find best alignment with typewell GR around last known TVT
    3. Extend alignment to predict post-PS TVT
    """
    # Pre-PS section: known GR + known TVT
    pre_gr  = hw.iloc[:ps_idx]["GR"].fillna(0).values.astype(float)
    pre_tvt = hw.iloc[:ps_idx]["TVT_input"].astype(float).values
    
    # Typewell GR (uniform 0.5 ft steps)
    tw_gr  = tw["GR"].fillna(0).values.astype(float)
    tw_tvt = tw["TVT"].astype(float).values
    
    # Find typewell TVT range overlapping with pre-PS TVT
    tvt_start = pre_tvt[-min(100, len(pre_tvt))]  # anchor window
    tvt_end   = pre_tvt[-1]
    tw_mask   = (tw_tvt >= tvt_start - 50) & (tw_tvt <= tvt_end + search_window)
    
    # Post-PS GR
    post_gr = hw.iloc[ps_idx:]["GR"].fillna(0).values.astype(float)
    
    # Concatenate pre-anchor + post GR as query
    anchor_len = min(50, ps_idx)
    query_gr = np.concatenate([pre_gr[-anchor_len:], post_gr])
    
    # DTW alignment against typewell window
    tw_window_gr  = tw_gr[tw_mask]
    tw_window_tvt = tw_tvt[tw_mask]
    
    path = dtw_ndim.warping_path(
        query_gr.reshape(-1, 1),
        tw_window_gr.reshape(-1, 1)
    )
    
    # Map query indices to typewell TVT
    path_dict = {}
    for qi, ti in path:
        path_dict.setdefault(qi, []).append(ti)
    
    # Average mapped typewell TVT for each query position
    query_tvt = np.array([
        tw_window_tvt[np.mean(path_dict[i]).astype(int)]
        if i in path_dict else np.nan
        for i in range(len(query_gr))
    ])
    
    # Return only post-PS portion (skip the anchor)
    return query_tvt[anchor_len:]
```

## Dependencies

```
pip install dtaidistance
```

## Notes
- DTW is O(n²) — for long wells, use `window` parameter to constrain search
- The pre-PS GR provides higher-quality anchor than typewell GR
- Consider using GR smoothing (Savitzky-Golay) before DTW to reduce noise
