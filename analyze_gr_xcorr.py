"""
Strategic analysis: Compare DTW-based TVT vs training TVT to understand evaluation gap.
Also test GR-typewell cross-correlation approach.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
import warnings; warnings.filterwarnings('ignore')

TRAIN_DIR = Path('train')
TEST_DIR  = Path('test')

def fill_gr(s): return s.ffill().bfill().astype(float).values

def get_ps(hw):
    mask = hw['TVT_input'].isna() | (hw['TVT_input'].astype(str).str.strip() == '')
    return int(mask.idxmax()) if mask.any() else len(hw)

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

# ============================================================
# 1. Sliding-window GR cross-correlation TVT estimate
# ============================================================
def gr_xcorr_tvt(hw_gr, tw_gr, tw_tvt, physics_tvt, ps, window=150, search_range=80):
    """
    For each post-PS row, find typewell TVT that maximizes GR cross-correlation
    in a local window.
    
    Returns: array of TVT estimates for ALL rows (pre-PS are physics TVT)
    """
    n = len(hw_gr)
    hw_gr_s = gaussian_filter1d(fill_gr(pd.Series(hw_gr)), sigma=2.0)
    tw_gr_s = gaussian_filter1d(fill_gr(pd.Series(tw_gr)), sigma=1.0)
    
    tvt_pred = physics_tvt.copy()
    
    for i in range(ps, n):
        # Window of horizontal well GR
        w_start = max(0, i - window)
        w_end   = min(n, i + window)
        hw_win  = hw_gr_s[w_start:w_end]
        
        # Physics TVT estimate at this row
        phys_tvt_i = physics_tvt[i]
        
        # Search typewell TVT range
        tw_center_idx = np.searchsorted(tw_tvt, phys_tvt_i)
        
        best_corr = -np.inf
        best_tvt  = phys_tvt_i
        
        for delta in range(-search_range, search_range + 1, 2):
            tw_target_tvt = phys_tvt_i + delta
            tw_idx = np.searchsorted(tw_tvt, tw_target_tvt)
            
            # Get matching typewell window (same length as hw_win)
            # Center the typewell window at the test position
            hw_half = len(hw_win) // 2
            tw_start = max(0, tw_idx - hw_half)
            tw_end   = min(len(tw_gr_s), tw_start + len(hw_win))
            tw_win   = tw_gr_s[tw_start:tw_end]
            
            if len(tw_win) < 10:
                continue
            
            min_len = min(len(hw_win), len(tw_win))
            h_seg = hw_win[:min_len]
            t_seg = tw_win[:min_len]
            
            # Pearson correlation
            h_std = h_seg.std()
            t_std = t_seg.std()
            if h_std < 1e-9 or t_std < 1e-9:
                continue
            
            corr = np.corrcoef(h_seg, t_seg)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_tvt  = tw_target_tvt
        
        tvt_pred[i] = best_tvt
    
    return tvt_pred


# ============================================================
# 2. Test on training wells to validate approach
# ============================================================
from sklearn.linear_model import LinearRegression

def get_physics_tvt(hw, ps):
    """Compute physics-based TVT from well trajectory."""
    z = hw['Z'].astype(float).values
    tvt_inp = hw['TVT_input'].astype(str).replace('', np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().values
    
    known = tvt_inp.values[:ps]
    valid = ~np.isnan(known)
    pre_z = z[:ps][valid]
    pre_tvt = known[valid]
    
    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
    else:
        slope = -1.0
    
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)])
    Z_anchor   = z[max(0, ps-1)]
    physics_tvt = anchor_tvt + slope * (z - Z_anchor)
    return physics_tvt


# Test on the 3 test wells (which have training truth)
print("=== Cross-correlation TVT vs Training TVT (test wells) ===\n")
for wid in ['000d7d20', '00bbac68', '00e12e8b']:
    hw_test  = pd.read_csv(TEST_DIR / f'{wid}__horizontal_well.csv')
    hw_train = pd.read_csv(TRAIN_DIR / f'{wid}__horizontal_well.csv')
    tw       = pd.read_csv(TEST_DIR / f'{wid}__typewell.csv')
    
    ps = get_ps(hw_test)
    n  = len(hw_test)
    
    # Ground truth (training TVT)
    gt_tvt = hw_train['TVT'].astype(float).values
    
    # Physics TVT
    phys_tvt = get_physics_tvt(hw_test, ps)
    
    # Physics RMSE (post-PS)
    phys_rmse = np.sqrt(np.mean((phys_tvt[ps:] - gt_tvt[ps:])**2))
    
    # Compute cross-correlation TVT
    hw_gr = fill_gr(hw_test['GR'])
    tw_gr = tw['GR'].values
    tw_tvt = tw['TVT'].astype(float).values
    
    xcorr_tvt = gr_xcorr_tvt(hw_gr, tw_gr, tw_tvt, phys_tvt, ps, window=100, search_range=60)
    xcorr_tvt_post = xcorr_tvt[ps:]
    xcorr_tvt_post_smooth = gaussian_filter1d(xcorr_tvt_post, sigma=3.0)
    
    xcorr_rmse = np.sqrt(np.mean((xcorr_tvt_post - gt_tvt[ps:])**2))
    xcorr_smooth_rmse = np.sqrt(np.mean((xcorr_tvt_post_smooth - gt_tvt[ps:])**2))
    
    print(f'Well {wid}:')
    print(f'  Post-PS rows: {n-ps}')
    print(f'  Training TVT range: [{gt_tvt[ps:].min():.2f}, {gt_tvt[ps:].max():.2f}]')
    print(f'  Physics TVT RMSE vs training:     {phys_rmse:.3f} ft')
    print(f'  XCorr TVT RMSE vs training:       {xcorr_rmse:.3f} ft')
    print(f'  XCorr+smooth RMSE vs training:    {xcorr_smooth_rmse:.3f} ft')
    print()
