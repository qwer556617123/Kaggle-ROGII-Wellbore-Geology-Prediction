"""
Analysis Task 2: Sliding Window GR Cross-Correlation
Analysis Task 6: DTW Baseline on train wells
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
FEAT_DIR  = DATA_DIR / "features"

np.random.seed(42)

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def load_well(wid):
    return (pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv"),
            pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv"))

def get_physics(hw, tw):
    ps  = get_ps(hw)
    z   = hw["Z"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    pre_z   = z[:ps]
    pre_tvt = hw["TVT"].astype(float).values[:ps]
    reg = LinearRegression().fit(pre_z.reshape(-1,1), pre_tvt)
    slope = float(reg.coef_[0])
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)])
    Z_anchor   = z[max(0, ps-1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)
    return ps, slope, anchored_physics

vr = pd.read_csv(FEAT_DIR / "viterbi_val_results.csv")
HARD_WELLS = ["1b1eba53", "ba48188d", "389ae58f", "81bf5923"]
easy_wells = vr[vr["lgbm"] < 5].head(5)["wid"].tolist()

# ════════════════════════════════════════════════════════════════════════════
# TASK 2: Sliding Window GR Cross-Correlation
# ════════════════════════════════════════════════════════════════════════════
print("="*70)
print("TASK 2: Sliding Window GR Cross-Correlation")
print("="*70)

def xcorr_tvt_estimate(hw_gr, tw_gr, tw_tvt, physics_tvt_row, search_width=100,
                        half_win=50):
    """Cross-correlate a window of HW GR with a typewell section.
    Returns the TVT offset that gives peak correlation for this window center."""
    n_tw = len(tw_tvt)
    # Typewell search range (by TVT index)
    lo = physics_tvt_row - search_width
    hi = physics_tvt_row + search_width
    tw_mask = (tw_tvt >= lo) & (tw_tvt <= hi)
    if tw_mask.sum() < half_win * 2:
        return physics_tvt_row  # fallback
    tw_sub_tvt = tw_tvt[tw_mask]
    tw_sub_gr  = tw_gr[tw_mask]
    query = hw_gr  # the window (already centered)
    # Normalize
    def norm(x): return (x - x.mean()) / (x.std() + 1e-9)
    q_n   = norm(query)
    tw_n  = norm(tw_sub_gr)
    # Cross-correlation via scipy
    corr = correlate(tw_n, q_n, mode='valid')
    if len(corr) == 0:
        return physics_tvt_row
    # Peak offset
    best_offset = int(np.argmax(corr))
    # Center of query maps to: tw_sub start + best_offset + half_win
    center_idx = best_offset + half_win
    center_idx = min(center_idx, len(tw_sub_tvt)-1)
    return float(tw_sub_tvt[center_idx])

def windowed_gr_tvt(hw, tw, physics_tvt, ps, half_win=50, search_width=100):
    gr_hw   = fill_arr(hw["GR"])
    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr_raw = fill_arr(tw["GR"])
    # Smooth
    tw_gr   = gaussian_filter1d(tw_gr_raw, sigma=1.5)
    hw_gr_s = gaussian_filter1d(gr_hw, sigma=1.0)
    n       = len(hw)
    n_post  = n - ps
    pred    = np.zeros(n_post)
    for i in range(n_post):
        row    = ps + i
        hw_lo  = max(0, row - half_win)
        hw_hi  = min(n, row + half_win)
        window = hw_gr_s[hw_lo:hw_hi]
        p_tvt  = physics_tvt[row]
        pred[i] = xcorr_tvt_estimate(window, tw_gr, tw_tvt, p_tvt,
                                      search_width=search_width, half_win=half_win)
    return pred

t2_rows = []
for wid in HARD_WELLS + easy_wells[:2]:
    try:
        hw, tw = load_well(wid)
        ps, slope, phys = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        post_true = tvt_true[ps:]
        post_phys = phys[ps:]
        rmse_phys = float(np.sqrt(np.mean((post_phys - post_true)**2)))
        results = {"well": wid, "type": "HARD" if wid in HARD_WELLS else "EASY",
                   "physics_rmse": round(rmse_phys,2)}
        # Test different window sizes
        for hw_win in [25, 50, 100]:
            pred = windowed_gr_tvt(hw, tw, phys, ps, half_win=hw_win, search_width=100)
            rmse = float(np.sqrt(np.mean((pred - post_true)**2)))
            results[f"xcorr_win{hw_win*2}_rmse"] = round(rmse, 2)
            results[f"xcorr_win{hw_win*2}_delta"] = round(rmse - rmse_phys, 2)
        t2_rows.append(results)
        print(f"  [{wid}] {results['type']}: phys={rmse_phys:.1f}  "
              f"xcorr50={results.get('xcorr_win50_rmse','?'):.1f}  "
              f"xcorr100={results.get('xcorr_win100_rmse','?'):.1f}  "
              f"xcorr200={results.get('xcorr_win200_rmse','?'):.1f}")
    except Exception as e:
        print(f"  FAIL {wid}: {e}")
        import traceback; traceback.print_exc()

t2 = pd.DataFrame(t2_rows)
print("\nFull table:")
print(t2.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# TASK 6: DTW Baseline on Training Wells
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 6: DTW Baseline on Training Wells")
print("="*70)

try:
    from dtaidistance import dtw as dtaidtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("  dtaidistance not available, using manual DTW")

def smooth_gr(arr, sigma=2.0):
    return gaussian_filter1d(arr, sigma=sigma)

def predict_tvt_dtw_train(wid):
    hw, tw = load_well(wid)
    ps     = get_ps(hw)
    hw_gr_raw = fill_arr(hw["GR"])
    hw_gr     = smooth_gr(hw_gr_raw, sigma=1.5)
    tvt_inp   = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = float(tvt_inp.ffill().iloc[ps-1])
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    tw_gr  = smooth_gr(tw_gr, sigma=1.0)
    ANCHOR_LEN = min(300, ps)
    anchor_gr  = hw_gr[ps-ANCHOR_LEN:ps]
    post_ps_gr = hw_gr[ps:]
    query_gr   = np.concatenate([anchor_gr, post_ps_gr])
    n_post     = len(post_ps_gr)
    anchor_tvt_start = last_known_tvt - (ANCHOR_LEN * 0.5)
    tw_start_idx = max(0, np.searchsorted(tw_tvt, anchor_tvt_start) - 20)
    tw_end_idx   = min(len(tw_tvt), np.searchsorted(tw_tvt, last_known_tvt) + 3000)
    tw_window_gr  = tw_gr[tw_start_idx:tw_end_idx]
    tw_window_tvt = tw_tvt[tw_start_idx:tw_end_idx]
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)
    q_norm  = norm(query_gr).astype(np.double)
    tw_norm = norm(tw_window_gr).astype(np.double)
    window_sz = max(50, int(max(len(q_norm), len(tw_norm)) * 0.15))
    if DTW_AVAILABLE:
        path = dtaidtw.warping_path(q_norm, tw_norm, window=window_sz)
    else:
        raise RuntimeError("DTW not available")
    qi_to_ti = {}
    for qi, ti in path:
        qi_to_ti.setdefault(qi, []).append(ti)
    all_q_tvt = np.zeros(len(query_gr))
    for qi, tis in qi_to_ti.items():
        ti_mean = int(round(np.mean(tis)))
        ti_mean = np.clip(ti_mean, 0, len(tw_window_tvt)-1)
        all_q_tvt[qi] = tw_window_tvt[ti_mean]
    nans = all_q_tvt == 0
    if nans.any():
        idx = np.arange(len(all_q_tvt))
        good = ~nans
        if good.sum() > 1:
            all_q_tvt[nans] = np.interp(idx[nans], idx[good], all_q_tvt[good])
    post_tvt_pred = smooth_gr(all_q_tvt[ANCHOR_LEN:], sigma=2.0)
    return post_tvt_pred, n_post

t6_rows = []
for wid in HARD_WELLS + easy_wells[:3]:
    try:
        hw, tw = load_well(wid)
        ps, slope, phys = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        post_true = tvt_true[ps:]
        post_phys = phys[ps:]
        rmse_phys = float(np.sqrt(np.mean((post_phys - post_true)**2)))
        pred_dtw, n_post = predict_tvt_dtw_train(wid)
        L = min(len(pred_dtw), len(post_true))
        rmse_dtw = float(np.sqrt(np.mean((pred_dtw[:L] - post_true[:L])**2)))
        delta = rmse_dtw - rmse_phys
        label = "HARD" if wid in HARD_WELLS else "EASY"
        t6_rows.append({
            "well": wid, "type": label,
            "physics_rmse": round(rmse_phys, 2),
            "dtw_rmse": round(rmse_dtw, 2),
            "dtw_vs_phys": round(delta, 2),
        })
        print(f"  [{wid}] {label}: physics={rmse_phys:.1f}ft  DTW={rmse_dtw:.1f}ft  delta={delta:+.1f}ft")
    except Exception as e:
        print(f"  FAIL {wid}: {e}")
        import traceback; traceback.print_exc()

t6 = pd.DataFrame(t6_rows)
print("\nFull table:")
print(t6.to_string(index=False))
print("\nMean by type:")
print(t6.groupby("type")[["physics_rmse","dtw_rmse","dtw_vs_phys"]].mean().round(2))
