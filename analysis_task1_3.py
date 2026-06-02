"""
Analysis Tasks 1, 3, 4, 5: GR match quality, physics error sources,
typewell informativeness, correction shape.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
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
    pre_pred = reg.predict(pre_z.reshape(-1,1))
    pre_r2   = float(1 - np.var(pre_tvt - pre_pred) / (np.var(pre_tvt)+1e-12))
    pre_resid_std = float(np.std(pre_tvt - pre_pred))
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)])
    Z_anchor   = z[max(0, ps-1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)
    return ps, slope, pre_r2, pre_resid_std, anchored_physics, anchor_tvt, Z_anchor

# ── Get hard and easy well lists ────────────────────────────────────────────
vr = pd.read_csv(FEAT_DIR / "viterbi_val_results.csv")
HARD_WELLS = ["1b1eba53", "ba48188d", "389ae58f", "81bf5923"]
easy_wells = vr[vr["lgbm"] < 5].head(5)["wid"].tolist()
ALL_WELLS  = HARD_WELLS + easy_wells
print(f"Hard wells: {HARD_WELLS}")
print(f"Easy wells: {easy_wells}")


# ════════════════════════════════════════════════════════════════════════════
# TASK 1: Pointwise GR match
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1: GR-Typewell Match Quality (pointwise ±100ft window)")
print("="*70)

def pointwise_gr_match(hw, tw, physics_tvt, ps, search_width=100):
    """For each post-PS row, find typewell TVT in [physics-W, physics+W] that
    minimises |gr_hw - gr_tw(TVT)|. Returns predicted TVT array."""
    gr_hw   = fill_arr(hw["GR"])
    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr   = fill_arr(tw["GR"])
    n_post  = len(hw) - ps
    pred    = np.zeros(n_post)
    for i in range(n_post):
        row   = ps + i
        p_tvt = physics_tvt[row]
        lo    = p_tvt - search_width
        hi    = p_tvt + search_width
        mask  = (tw_tvt >= lo) & (tw_tvt <= hi)
        if mask.sum() == 0:
            pred[i] = p_tvt
            continue
        tw_sub  = tw_tvt[mask]
        twgr_sub = tw_gr[mask]
        gr_query = gr_hw[row]
        best_i   = np.argmin(np.abs(twgr_sub - gr_query))
        pred[i]  = tw_sub[best_i]
    return pred

t1_rows = []
for wid in ALL_WELLS:
    try:
        hw, tw = load_well(wid)
        ps, slope, pre_r2, pre_resid_std, phys, anchor_tvt, Z_anchor = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        post_true = tvt_true[ps:]
        post_phys = phys[ps:]
        gr_pred   = pointwise_gr_match(hw, tw, phys, ps, search_width=100)
        rmse_phys = float(np.sqrt(np.mean((post_phys - post_true)**2)))
        rmse_gr   = float(np.sqrt(np.mean((gr_pred   - post_true)**2)))
        label = "HARD" if wid in HARD_WELLS else "EASY"
        t1_rows.append({
            "well": wid, "type": label,
            "pre_resid_std": round(pre_resid_std, 1),
            "physics_rmse": round(rmse_phys, 2),
            "gr_match_rmse": round(rmse_gr, 2),
            "gr_vs_phys": round(rmse_gr - rmse_phys, 2),
        })
        print(f"  [{wid}] {label}: physics={rmse_phys:.1f}ft  gr_match={rmse_gr:.1f}ft  delta={rmse_gr-rmse_phys:+.1f}ft")
    except Exception as e:
        print(f"  FAIL {wid}: {e}")

t1 = pd.DataFrame(t1_rows)
print("\nSummary by type:")
print(t1.groupby("type")[["physics_rmse","gr_match_rmse","gr_vs_phys"]].mean().round(2))


# ════════════════════════════════════════════════════════════════════════════
# TASK 3: Physics Error Sources
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 3: Physics Error Sources")
print("="*70)

t3_rows = []
for wid in ALL_WELLS:
    try:
        hw, tw = load_well(wid)
        ps, slope, pre_r2, pre_resid_std, phys, anchor_tvt, Z_anchor = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        x = hw["X"].astype(float).values
        y = hw["Y"].astype(float).values
        z = hw["Z"].astype(float).values
        md = hw["MD"].astype(float).values
        post_z  = z[ps:]
        post_x  = x[ps:]
        post_y  = y[ps:]
        post_true = tvt_true[ps:]
        post_phys = phys[ps:]
        post_z_range = float(post_z.max() - post_z.min()) if len(post_z)>0 else 0
        post_xy_range = float(np.sqrt((post_x.max()-post_x.min())**2 + (post_y.max()-post_y.min())**2)) if len(post_x)>0 else 0
        phys_err_max = float(np.max(np.abs(post_phys - post_true))) if len(post_true)>0 else 0
        phys_err_mean = float(np.mean(np.abs(post_phys - post_true))) if len(post_true)>0 else 0
        # post-PS Z total drop (signed)
        z_total = float(post_z[-1] - post_z[0]) if len(post_z)>1 else 0
        total_md = float(md[ps:].max() - md[ps:].min()) if ps < len(md)-1 else 1.
        has_ancc = not hw["ANCC"].isna().all()
        label = "HARD" if wid in HARD_WELLS else "EASY"
        t3_rows.append({
            "well": wid, "type": label,
            "slope": round(slope, 3),
            "pre_r2": round(pre_r2, 4),
            "pre_resid_std": round(pre_resid_std, 2),
            "post_z_range": round(post_z_range, 1),
            "post_xy_range": round(post_xy_range, 1),
            "z_total_post": round(z_total, 1),
            "has_ancc": has_ancc,
            "phys_err_max": round(phys_err_max, 1),
            "phys_err_mean": round(phys_err_mean, 2),
        })
    except Exception as e:
        print(f"  FAIL {wid}: {e}")

t3 = pd.DataFrame(t3_rows)
print(t3.to_string(index=False))
print("\nMean by type:")
print(t3.groupby("type")[["slope","pre_r2","pre_resid_std","post_z_range","post_xy_range","phys_err_max","phys_err_mean"]].mean().round(3))


# ════════════════════════════════════════════════════════════════════════════
# TASK 4: Typewell Informativeness
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 4: Typewell Informativeness")
print("="*70)

def stretch_correlation(hw_gr, tw_gr, tw_tvt, tvt_arr):
    """Check if horizontal well GR ~ stretched typewell GR by computing
    Pearson correlation after interpolating typewell GR onto hw TVT."""
    tw_gr_at_tvt = np.interp(tvt_arr, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    # normalize both
    def norm(x):
        s = x.std()
        return (x - x.mean()) / (s + 1e-9)
    r = np.corrcoef(norm(hw_gr), norm(tw_gr_at_tvt))[0,1]
    return float(r)

t4_rows = []
for wid in ALL_WELLS:
    try:
        hw, tw = load_well(wid)
        ps, slope, pre_r2, pre_resid_std, phys, anchor_tvt, Z_anchor = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        gr_hw    = fill_arr(hw["GR"])
        tw_tvt   = tw["TVT"].astype(float).values
        tw_gr    = fill_arr(tw["GR"])
        # Typewell range
        tw_range = float(tw_tvt.max() - tw_tvt.min())
        # Pre-PS TVT range
        pre_tvt_range = float(tvt_true[:ps].max() - tvt_true[:ps].min()) if ps > 0 else 0
        # Post-PS TVT range (true)
        post_tvt = tvt_true[ps:]
        post_tvt_range = float(post_tvt.max() - post_tvt.min()) if len(post_tvt) > 0 else 0
        # Fraction of typewell explored
        pre_frac  = pre_tvt_range / (tw_range + 1e-9)
        post_frac = post_tvt_range / (tw_range + 1e-9)
        # Post-PS anchor: what fraction beyond anchor TVT does the well explore?
        post_explored_frac = (post_tvt.max() - anchor_tvt) / (tw_range + 1e-9) if len(post_tvt)>0 else 0
        # Stretch correlation: use pre-PS GR vs typewell
        pre_gr   = gr_hw[:ps]
        pre_tvt_arr = tvt_true[:ps]
        r_pre = stretch_correlation(pre_gr, tw_gr, tw_tvt, pre_tvt_arr)
        # Post-PS
        post_gr   = gr_hw[ps:]
        post_tvt_arr = tvt_true[ps:]
        r_post = stretch_correlation(post_gr, tw_gr, tw_tvt, post_tvt_arr)
        label = "HARD" if wid in HARD_WELLS else "EASY"
        t4_rows.append({
            "well": wid, "type": label,
            "tw_range_ft": round(tw_range, 1),
            "pre_tvt_range": round(pre_tvt_range, 1),
            "post_tvt_range": round(post_tvt_range, 1),
            "anchor_tvt": round(anchor_tvt, 1),
            "tw_range_pre_frac": round(pre_frac, 3),
            "tw_range_post_frac": round(post_frac, 3),
            "r_pre_gr_tw": round(r_pre, 3),
            "r_post_gr_tw": round(r_post, 3),
        })
    except Exception as e:
        print(f"  FAIL {wid}: {e}")

t4 = pd.DataFrame(t4_rows)
print(t4.to_string(index=False))
print("\nMean by type:")
print(t4.groupby("type")[["tw_range_ft","pre_tvt_range","post_tvt_range",
                           "tw_range_pre_frac","tw_range_post_frac",
                           "r_pre_gr_tw","r_post_gr_tw"]].mean().round(3))


# ════════════════════════════════════════════════════════════════════════════
# TASK 5: Correction Shape Analysis
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 5: Correction Shape Analysis")
print("="*70)

def is_monotone(arr):
    if len(arr) < 2: return "n/a"
    diffs = np.diff(arr)
    pct_pos = (diffs > 0).mean()
    if pct_pos > 0.9: return "mono_inc"
    if pct_pos < 0.1: return "mono_dec"
    return "non_mono"

t5_rows = []
for wid in ALL_WELLS:
    try:
        hw, tw = load_well(wid)
        ps, slope, pre_r2, pre_resid_std, phys, anchor_tvt, Z_anchor = get_physics(hw, tw)
        tvt_true = hw["TVT"].astype(float).values
        x = hw["X"].astype(float).values
        y = hw["Y"].astype(float).values
        correction = tvt_true - phys
        post_corr  = correction[ps:]
        post_x     = x[ps:] - x[max(0,ps-1)]
        post_y     = y[ps:] - y[max(0,ps-1)]
        # Stats
        corr_range   = float(post_corr.max() - post_corr.min())
        corr_std     = float(post_corr.std())
        corr_mean    = float(post_corr.mean())
        # Monotonicity
        mono = is_monotone(post_corr)
        # Fit correction ~ a*X + b*Y
        if len(post_corr) > 10:
            XY = np.column_stack([post_x, post_y])
            reg_xy = LinearRegression().fit(XY, post_corr)
            xy_pred = reg_xy.predict(XY)
            ss_res = np.sum((post_corr - xy_pred)**2)
            ss_tot = np.sum((post_corr - post_corr.mean())**2)
            r2_xy  = float(1 - ss_res/(ss_tot+1e-12))
            # Fit correction ~ linear in post index
            t_arr = np.arange(len(post_corr)).reshape(-1,1)
            reg_t = LinearRegression().fit(t_arr, post_corr)
            t_pred = reg_t.predict(t_arr)
            ss_res_t = np.sum((post_corr - t_pred)**2)
            r2_t = float(1 - ss_res_t/(ss_tot+1e-12))
            # Fit correction ~ a*X + b*Y + c*Z
            z = hw["Z"].astype(float).values
            post_z = z[ps:] - z[max(0,ps-1)]
            XYZ = np.column_stack([post_x, post_y, post_z])
            reg_xyz = LinearRegression().fit(XYZ, post_corr)
            xyz_pred = reg_xyz.predict(XYZ)
            ss_res_xyz = np.sum((post_corr - xyz_pred)**2)
            r2_xyz = float(1 - ss_res_xyz/(ss_tot+1e-12))
            slope_x = round(reg_xy.coef_[0], 4)
            slope_y = round(reg_xy.coef_[1], 4)
        else:
            r2_xy = r2_t = r2_xyz = 0.0
            slope_x = slope_y = 0.0
        label = "HARD" if wid in HARD_WELLS else "EASY"
        t5_rows.append({
            "well": wid, "type": label,
            "corr_mean": round(corr_mean, 1),
            "corr_std": round(corr_std, 1),
            "corr_range": round(corr_range, 1),
            "monotone": mono,
            "R2_vs_linear_t": round(r2_t, 3),
            "R2_vs_XY": round(r2_xy, 3),
            "R2_vs_XYZ": round(r2_xyz, 3),
            "slope_x": slope_x,
            "slope_y": slope_y,
        })
        print(f"  [{wid}] {label}: corr_range={corr_range:.1f}ft  R2_t={r2_t:.3f}  R2_XY={r2_xy:.3f}  R2_XYZ={r2_xyz:.3f}  shape={mono}")
    except Exception as e:
        print(f"  FAIL {wid}: {e}")

t5 = pd.DataFrame(t5_rows)
print("\nFull table:")
print(t5.to_string(index=False))
print("\nMean by type:")
print(t5.groupby("type")[["corr_mean","corr_std","corr_range",
                           "R2_vs_linear_t","R2_vs_XY","R2_vs_XYZ"]].mean().round(3))

print("\n\n=== INTERPRETATION NOTES ===")
print("Task 1: GR pointwise match  - gr_vs_phys < 0 means GR match BETTER than physics")
print("Task 3: Physics errors      - large pre_resid_std → physics slope unreliable")
print("Task 4: Typewell info       - r_post_gr_tw near 1 → GR matching viable")
print("Task 5: Correction shape    - R2_XY near 1 → dip correction viable")
