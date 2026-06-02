"""
Test GR xcorr TVT refinement using LGBM v23 predictions on val wells.
Strategy: Use LGBM predicted TVT to bin GR, apply xcorr with typewell GR,
get per-well TVT shift, apply constant shift to improve predictions.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d

DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
FEAT_DIR  = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# Load trained model
pkg = pickle.load(open(MODELS_DIR / "lgbm_v23_spatial.pkl", "rb"))
model       = pkg["model"]
feature_cols = pkg["feature_cols"]

# Load val IDs
val_ids = pd.read_csv(FEAT_DIR / "val_ids.csv", header=None)[0].tolist()

# ── Load v23 spatial infrastructure (dip coeffs + neighbor pool) ──────────────
dip_df     = pd.read_csv(FEAT_DIR / "well_dip_coeffs.csv")
train_ids  = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
K_NEIGHBORS = 10
MIN_DIST_CAP = 100.0
MIN_R2_QUALITY = 0.3

train_pool = dip_df[
    dip_df["wid"].isin(train_ids) & (dip_df["r2_xy"] >= MIN_R2_QUALITY)
].copy().reset_index(drop=True)

pool_cx = train_pool["cx"].values
pool_cy = train_pool["cy"].values
pool_ax = train_pool["alpha_x"].values
pool_ay = train_pool["alpha_y"].values
pool_r2 = train_pool["r2_xy"].values
pool_wids = train_pool["wid"].values
dip_lookup = dip_df.set_index("wid")[["cx", "cy"]].to_dict("index")


def compute_nbr_stats(wid, cx_well, cy_well):
    dist = np.sqrt((pool_cx - cx_well)**2 + (pool_cy - cy_well)**2)
    dist[pool_wids == wid] = np.inf
    k = min(K_NEIGHBORS, int(np.isfinite(dist).sum()))
    idx = np.argpartition(dist, k)[:k]
    idx = idx[np.argsort(dist[idx])]
    w = 1.0 / np.maximum(dist[idx], MIN_DIST_CAP)
    w /= w.sum()
    nbr_ax = float(np.dot(w, pool_ax[idx]))
    nbr_ay = float(np.dot(w, pool_ay[idx]))
    return {
        "nbr_alpha_x": nbr_ax,
        "nbr_alpha_y": nbr_ay,
        "nbr_dist_min": float(dist[idx[0]]),
        "nbr_r2_mean": float(np.mean(pool_r2[idx])),
        "nbr_count_consistent_x": float(np.mean(np.sign(pool_ax[idx]) == np.sign(nbr_ax)))
        if nbr_ax != 0 else 0.5,
    }


def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values


def build_and_predict(wid):
    """Build features, run LGBM, return (tvt_pred_post, tvt_true_post, gr_post, tw)."""
    hw = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv")

    n = len(hw)
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    ps = int(mask.idxmax()) if mask.any() else n

    gr = fill_arr(hw["GR"])
    md = hw["MD"].astype(float).values
    z  = hw["Z"].astype(float).values
    x  = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().values
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    gr_s   = pd.Series(gr)

    pre_z   = z[:ps]
    pre_tvt = hw["TVT"].astype(float).values[:ps]

    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.0
    else:
        slope = -1.0
        pre_r2 = 0.0

    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    post_z  = z[ps:]
    post_md = md[ps:]
    n_post  = len(post_z)
    z_end   = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z      = z_end - Z_anchor
    total_md     = float(post_md[-1] - post_md[0]) if n_post > 1 else 1.0
    post_dz_rate = total_z / total_md
    phys_tvt_end = slope * total_z
    pre_dz_md    = (z[ps-1] - z[0]) / (md[ps-1] - md[0]) if ps > 1 and abs(md[ps-1] - md[0]) > 0.01 else -0.01
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
        "dz_dmd": np.gradient(z, md),
        "dx_dmd": np.gradient(x, md),
        "dy_dmd": np.gradient(y, md),
        "inclination": np.arctan2(
            np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
            np.abs(np.gradient(z)) + 1e-9) * 180 / np.pi,
        "raw_dz":  np.diff(z,  prepend=z[0]),
        "raw_dmd": np.diff(md, prepend=md[0]),
        "last_known_tvt":  lkt,
        "rows_since_ps":   np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps":  np.maximum(0, ps - np.arange(n)).astype(float),
        "ps_idx":          float(ps),
        "tvt_z_slope":     np.full(n, slope),
        "physics_tvt":     anchored_physics,
        "physics_vs_lkt":  anchored_physics - lkt,
        "physics_dtvt":    slope * np.diff(z, prepend=z[0]),
        "gr_at_physics":   gr_at_physics,
        "gr_dev_physics":  gr_dev,
        "total_z_change_post":   np.full(n, total_z),
        "physics_tvt_at_end":    np.full(n, phys_tvt_end),
        "post_dz_rate":          np.full(n, post_dz_rate),
        "dz_rate_change":        np.full(n, dz_rate_change),
        "n_post_ps":             np.full(n, float(n_post)),
        "post_ps_frac":          np.where(n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.).astype(float),
        "pre_r2": np.full(n, pre_r2),
    })

    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt   = tvt_inp[:ps].values
        vtvt   = vtvt[~np.isnan(vtvt)]
        pdtvt  = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.)
    else:
        feat["pre_ps_dz_slope"] = np.zeros(n)
        feat["pre_ps_dtvt_mean"] = np.zeros(n)
        feat["pre_ps_dtvt_std"]  = np.zeros(n)

    df = pd.DataFrame(feat)
    df["well_id"] = wid
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    df["anchored_physics_col"] = anchored_physics.astype(np.float32)
    df["_x"] = x
    df["_y"] = y

    # Add spatial features
    if wid in dip_lookup:
        cx = dip_lookup[wid]["cx"]
        cy = dip_lookup[wid]["cy"]
        stats = compute_nbr_stats(wid, cx, cy)
    else:
        stats = {"nbr_alpha_x": 0., "nbr_alpha_y": 0., "nbr_dist_min": 9999., "nbr_r2_mean": 0., "nbr_count_consistent_x": 0.5}

    nbr_ax = stats["nbr_alpha_x"]
    nbr_ay = stats["nbr_alpha_y"]
    x_ps = x[ps] if ps < n else x[-1]
    y_ps = y[ps] if ps < n else y[-1]
    df["nbr_alpha_x"]            = np.float32(nbr_ax)
    df["nbr_alpha_y"]            = np.float32(nbr_ay)
    df["nbr_3d_corr"]            = (nbr_ax * (x - x_ps) + nbr_ay * (y - y_ps)).astype(np.float32)
    df["nbr_dist_min"]           = np.float32(stats["nbr_dist_min"])
    df["nbr_r2_mean"]            = np.float32(stats["nbr_r2_mean"])
    df["nbr_count_consistent_x"] = np.float32(stats["nbr_count_consistent_x"])

    # Predict
    X_feat = df[feature_cols].values
    corr_pred = model.predict(X_feat, num_iteration=model.best_iteration)
    tvt_pred_all = anchored_physics + corr_pred

    # Return post-PS only
    post_mask = np.arange(n) >= ps
    tvt_pred_post = tvt_pred_all[post_mask]
    tvt_true_post = hw["TVT"].astype(float).values[post_mask]
    gr_post = gr[post_mask]

    return tvt_pred_post, tvt_true_post, gr_post, tw, ps


def gr_xcorr_lag(tvt_est, gr_post, tw, step=0.5, search_range=30.0):
    """
    Bin horizontal GR by estimated TVT, find xcorr lag vs typewell GR.
    Returns lag (ft) — positive means tvt_est is too low by that amount.
    """
    tvt_range = [tvt_est.min() - 2, tvt_est.max() + 2]
    span = tvt_est.max() - tvt_est.min()
    if span < 5:
        return np.nan, 0.0

    bins = np.arange(tvt_range[0], tvt_range[1] + step, step)
    if len(bins) < 8:
        return np.nan, 0.0
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bin_idx = np.digitize(tvt_est, bins)
    bin_gr = np.array([
        np.nanmean(gr_post[bin_idx == i]) if (bin_idx == i).any() else np.nan
        for i in range(1, len(bins))
    ])

    # Extend typewell GR range for search
    tw_tvt_range = [tvt_range[0] - search_range, tvt_range[1] + search_range]
    tw_sorted = tw.sort_values("TVT")
    tw_mask = (tw_sorted["TVT"] >= tw_tvt_range[0]) & (tw_sorted["TVT"] <= tw_tvt_range[1])
    if tw_mask.sum() < 5:
        return np.nan, 0.0
    tw_sub = tw_sorted[tw_mask]
    tw_interp = interp1d(tw_sub["TVT"], tw_sub["GR"], kind="linear", fill_value="extrapolate")
    tw_gr = tw_interp(bin_centers)

    valid = ~np.isnan(bin_gr)
    if valid.sum() < 8:
        return np.nan, 0.0

    hw_sig = bin_gr.copy()
    hw_sig[~valid] = np.nanmean(bin_gr[valid])
    hw_sig -= hw_sig.mean()
    tw_sig = tw_gr - tw_gr.mean()

    xcorr = correlate(tw_sig, hw_sig, mode="full")
    lags = np.arange(-(len(bin_centers) - 1), len(bin_centers)) * step
    # Only search within ±search_range
    valid_lag_mask = np.abs(lags) <= search_range
    best_lag = lags[valid_lag_mask][np.argmax(xcorr[valid_lag_mask])]

    # Correlation quality
    hw_norm = hw_sig[valid] - hw_sig[valid].mean()
    tw_at_bins = tw_gr[valid] - tw_gr[valid].mean()
    corr_quality = np.corrcoef(hw_norm, tw_at_bins)[0, 1] if hw_norm.std() > 0 else 0.0

    return best_lag, corr_quality


# ── Main evaluation ─────────────────────────────────────────────────────────
print("Computing LGBM predictions + GR xcorr correction for val wells...")
print(f"Using {len(val_ids)} val wells")

results = []
for i, wid in enumerate(val_ids):
    try:
        tvt_pred, tvt_true, gr_post, tw, ps = build_and_predict(wid)
        rmse_lgbm = float(np.sqrt(np.mean((tvt_pred - tvt_true)**2)))

        # GR xcorr: use LGBM prediction as starting TVT estimate
        lag, qual = gr_xcorr_lag(tvt_pred, gr_post, tw, step=0.5, search_range=20.0)

        if not np.isnan(lag):
            tvt_corrected = tvt_pred + lag
            rmse_corr = float(np.sqrt(np.mean((tvt_corrected - tvt_true)**2)))
        else:
            rmse_corr = rmse_lgbm

        results.append({
            "wid": wid,
            "n_rows": len(tvt_true),
            "tvt_span": tvt_true.max() - tvt_true.min(),
            "rmse_lgbm": rmse_lgbm,
            "xcorr_lag": lag,
            "xcorr_qual": qual,
            "rmse_xcorr": rmse_corr,
            "improvement": rmse_lgbm - rmse_corr,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(val_ids)} done", flush=True)
    except Exception as e:
        print(f"  SKIP {wid}: {e}")

df = pd.DataFrame(results)
print(f"\n{'='*60}")
print(f"Val wells analyzed: {len(df)}")

# Row-weighted RMSE (matches competition metric more closely)
total_rows = df.n_rows.sum()
rw_lgbm = float(np.sqrt(np.sum(df.n_rows * df.rmse_lgbm**2) / total_rows))
rw_xcorr = float(np.sqrt(np.sum(df.n_rows * df.rmse_xcorr**2) / total_rows))
print(f"\nRow-weighted RMSE  — LGBM: {rw_lgbm:.4f} ft  |  LGBM+xcorr: {rw_xcorr:.4f} ft")
print(f"Per-well mean RMSE — LGBM: {df.rmse_lgbm.mean():.4f} ft  |  LGBM+xcorr: {df.rmse_xcorr.mean():.4f} ft")

# Where does xcorr help?
better = df[df.improvement > 0.5]
worse  = df[df.improvement < -0.5]
print(f"\nXcorr improved: {len(better)} wells (mean improvement {better.improvement.mean():.2f} ft)")
print(f"Xcorr hurt:     {len(worse)} wells (mean degradation {worse.improvement.mean():.2f} ft)")
print(f"Xcorr corr quality (mean): {df.xcorr_qual.mean():.3f}")

print("\nBottom 10 by xcorr improvement:")
print(df.nsmallest(10, 'improvement')[['wid','tvt_span','xcorr_lag','xcorr_qual','rmse_lgbm','rmse_xcorr','improvement']].to_string(index=False))
