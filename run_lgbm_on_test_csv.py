"""
Run v23 LGBM inference on test CSVs (as Kaggle kernel does).
This tells us what the actual Kaggle predictions are.
"""
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

DATA  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
FEAT  = DATA / "features"
TEST  = DATA / "test"
TRAIN = DATA / "train"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

with open(DATA / "models" / "lgbm_v23_spatial.pkl", "rb") as f:
    md = pickle.load(f)
model, feat_cols = md["model"], md["feature_cols"]

nbr_stats = json.load(open(FEAT / "test_well_nbr_stats.json"))

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values

def build_test_features(hw, tw, wid, nbr_x, nbr_y):
    """Build features for a test-mode well (no TVT column)."""
    n   = len(hw)
    ps  = get_ps(hw)
    gr  = fill_arr(hw["GR"])
    md  = hw["MD"].astype(float).values
    z   = hw["Z"].astype(float).values
    x   = hw["X"].astype(float).values
    y   = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().values
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    gr_s   = pd.Series(gr)
    
    # Pre-PS slope from TVT_input
    known  = tvt_inp[:ps].values
    valid  = ~np.isnan(known)
    pre_z  = z[:ps][valid]
    pre_tvt = known[valid]
    
    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / (np.var(pre_tvt) + 1e-12))
    else:
        slope, pre_r2 = -1.0, 0.0
    
    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)
    
    post_z   = z[ps:] if ps < n else z[:]
    post_md  = md[ps:] if ps < n else md[:]
    n_post   = len(post_z)
    z_end    = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z  = z_end - Z_anchor
    total_md = float(post_md[-1] - post_md[0]) if n_post > 1 else 1.0
    post_dz_rate  = total_z / total_md
    phys_tvt_end  = slope * total_z
    pre_dz_md     = (z[ps-1] - z[0]) / (md[ps-1] - md[0]) if ps > 1 and abs(md[ps-1] - md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md
    
    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev        = gr - gr_at_physics
    
    feat = {"gr": gr, "gr_diff": np.gradient(gr), "gr_diff2": np.gradient(np.gradient(gr))}
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values
    
    feat.update({
        "md": md, "z": z,
        "dz_dmd": np.gradient(z, md), "dx_dmd": np.gradient(x, md), "dy_dmd": np.gradient(y, md),
        "inclination": np.arctan2(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
                                   np.abs(np.gradient(z)) + 1e-9) * 180 / np.pi,
        "raw_dz": np.diff(z, prepend=z[0]), "raw_dmd": np.diff(md, prepend=md[0]),
        "last_known_tvt": lkt,
        "rows_since_ps":  np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps": np.maximum(0, ps - np.arange(n)).astype(float),
        "ps_idx": float(ps),
        "tvt_z_slope": np.full(n, slope),
        "physics_tvt": anchored_physics,
        "physics_vs_lkt": anchored_physics - lkt,
        "physics_dtvt": slope * np.diff(z, prepend=z[0]),
        "gr_at_physics": gr_at_physics,
        "gr_dev_physics": gr_dev,
        "total_z_change_post": np.full(n, total_z),
        "physics_tvt_at_end": np.full(n, phys_tvt_end),
        "post_dz_rate": np.full(n, post_dz_rate),
        "dz_rate_change": np.full(n, dz_rate_change),
        "n_post_ps": np.full(n, float(n_post)),
        "post_ps_frac": np.where(n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.).astype(float),
        "pre_r2": np.full(n, pre_r2),
    })
    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt   = tvt_inp[:ps].values; vtvt = vtvt[~np.isnan(vtvt)]
        pdtvt  = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.)
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)
    
    df = pd.DataFrame(feat)
    df["well_id"] = wid; df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    df["anchored_physics_col"] = anchored_physics.astype(np.float32)
    df["_x"] = x; df["_y"] = y
    
    # Add spatial features
    x_ps = x[ps] if ps < n else x[-1]
    y_ps = y[ps] if ps < n else y[-1]
    dx = x - x_ps; dy = y - y_ps
    s = nbr_stats.get(wid, {})
    nbr_ax = s.get("nbr_alpha_x", 0.0)
    nbr_ay = s.get("nbr_alpha_y", 0.0)
    df["nbr_alpha_x"]            = np.float32(nbr_ax)
    df["nbr_alpha_y"]            = np.float32(nbr_ay)
    df["nbr_3d_corr"]            = (nbr_ax * dx + nbr_ay * dy).astype(np.float32)
    df["nbr_dist_min"]           = np.float32(s.get("nbr_dist_min", 9999.0))
    df["nbr_r2_mean"]            = np.float32(s.get("nbr_r2_mean", 0.0))
    df["nbr_count_consistent_x"] = np.float32(s.get("nbr_count_consistent_x", 0.5))
    
    return df, anchored_physics, anchor_tvt, ps, slope


print("="*60)
print("Running v23 LGBM on TEST CSV (as Kaggle kernel does):")
print("="*60)
for wid in TEST_WELLS:
    hw_te = pd.read_csv(TEST  / f"{wid}__horizontal_well.csv")
    hw_tr = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw_te = pd.read_csv(TEST  / f"{wid}__typewell.csv")
    
    ps = get_ps(hw_te)
    tvt_true = hw_tr["TVT"].astype(float).values[ps:]
    
    nbr_x = nbr_stats.get(wid, {}).get("nbr_alpha_x", 0.0)
    nbr_y = nbr_stats.get(wid, {}).get("nbr_alpha_y", 0.0)
    
    df, anchored_physics, anchor, ps_, slope = build_test_features(hw_te, tw_te, wid, nbr_x, nbr_y)
    
    # Inference on post-PS rows
    post_mask = df["is_post_ps"] == 1
    X_test = df[post_mask][feat_cols].values
    correction = model.predict(X_test)
    
    tvt_pred = anchored_physics[ps:] + correction
    tvt_pred = gaussian_filter1d(tvt_pred, sigma=1.0)
    
    rmse_anchor = float(np.sqrt(np.mean((anchor   - tvt_true)**2)))
    rmse_pred   = float(np.sqrt(np.mean((tvt_pred - tvt_true)**2)))
    
    print(f"\n{wid}:")
    print(f"  anchor        = {anchor:.2f}  slope={slope:.4f}")
    print(f"  anchored_phys end: {anchored_physics[-1]:.2f}  (total_change={anchored_physics[-1]-anchor:.2f})")
    print(f"  LGBM correction: mean={correction.mean():.3f}  std={correction.std():.3f}  range=[{correction.min():.2f},{correction.max():.2f}]")
    print(f"  prediction: mean={tvt_pred.mean():.2f} std={tvt_pred.std():.3f} trend={tvt_pred[-1]-tvt_pred[0]:.2f}")
    print(f"  training TVT: mean={tvt_true.mean():.2f} std={tvt_true.std():.3f}")
    print(f"  RMSE: anchor={rmse_anchor:.3f}  LGBM={rmse_pred:.3f}")
    
    # Show key features
    post_df = df[post_mask]
    print(f"  physics_vs_lkt: mean={post_df['physics_vs_lkt'].mean():.2f} range=[{post_df['physics_vs_lkt'].min():.2f},{post_df['physics_vs_lkt'].max():.2f}]")
    print(f"  nbr_3d_corr:    mean={post_df['nbr_3d_corr'].mean():.2f} range=[{post_df['nbr_3d_corr'].min():.2f},{post_df['nbr_3d_corr'].max():.2f}]")
