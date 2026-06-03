"""
LightGBM v17 — xcorr features
New features added over v8:
  - xcorr_best_offset: TVT offset that maximizes GR-typewell correlation
  - xcorr_peak_corr: peak correlation at best offset
  - xcorr_corr_at_0: correlation at physics TVT (0 offset)
  - xcorr_corr_std: std of correlation profile (peakiness)
  - xcorr_corr_rel: peak_corr - corr_at_0 (improvement from searching)
  - pre_ps_gr_tw_corr: pre-PS GR vs typewell GR correlation
  - gr_at_xcorr: per-row typewell GR at (physics_TVT + xcorr_best_offset)
  - gr_dev_xcorr: per-row horizontal GR - gr_at_xcorr
  - post_gr_mean / post_gr_std / post_gr_iqr: post-PS GR statistics
  - pre_gr_mean / pre_gr_std: pre-PS GR statistics
  - gr_mean_ratio_post_pre: post_gr_mean / (pre_gr_mean + 1)
Expected: ~56 features (v8: 46 + 10 new)
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
FEAT_DIR   = DATA_DIR / "features"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]


def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values


def load_well(wid, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (pd.read_csv(d / f"{wid}__horizontal_well.csv"),
            pd.read_csv(d / f"{wid}__typewell.csv"))


def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def compute_xcorr_features(gr, phys, tw_tvt, tw_gr, ps,
                            offset_range=120, step=2.0):
    """Compute per-well GR-typewell cross-correlation features."""
    gr_post = gr[ps:]; phys_post = phys[ps:]
    n_post = len(gr_post)
    if n_post < 10:
        return {k: 0. for k in [
            "xcorr_best_offset", "xcorr_peak_corr", "xcorr_corr_at_0",
            "xcorr_corr_std", "xcorr_corr_rel", "pre_ps_gr_tw_corr",
            "post_gr_mean", "post_gr_std", "post_gr_iqr",
            "pre_gr_mean", "pre_gr_std", "gr_mean_ratio_post_pre"
        ]}
    gr_post_sm = gaussian_filter1d(gr_post, sigma=5).astype(float)
    offsets = np.arange(-offset_range, offset_range + step, step)
    mid_idx = len(offsets) // 2
    corrs = []
    for delta in offsets:
        tw_interp = np.interp(phys_post + delta, tw_tvt, tw_gr,
                              left=tw_gr[0], right=tw_gr[-1])
        c = np.corrcoef(gr_post_sm, tw_interp)[0, 1]
        corrs.append(float(c) if not np.isnan(c) else 0.)
    corrs = np.array(corrs)
    best_idx = int(np.argmax(corrs))
    # Pre-PS GR-typewell correlation (formation match quality)
    if ps > 5:
        gr_pre_sm = gaussian_filter1d(gr[:ps], sigma=3)
        tw_pre = np.interp(phys[:ps], tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
        c_pre = float(np.corrcoef(gr_pre_sm, tw_pre)[0, 1])
        c_pre = c_pre if not np.isnan(c_pre) else 0.
    else:
        c_pre = 0.
    # GR statistics
    gr_post_f = gr_post.astype(float)
    q25, q75 = np.percentile(gr_post_f, [25, 75])
    pre_gr = gr[:ps].astype(float) if ps > 0 else np.array([0.])
    return {
        "xcorr_best_offset":    float(offsets[best_idx]),
        "xcorr_peak_corr":      float(corrs[best_idx]),
        "xcorr_corr_at_0":      float(corrs[mid_idx]),
        "xcorr_corr_std":       float(corrs.std()),
        "xcorr_corr_rel":       float(corrs[best_idx] - corrs[mid_idx]),
        "pre_ps_gr_tw_corr":    c_pre,
        "post_gr_mean":         float(gr_post_f.mean()),
        "post_gr_std":          float(gr_post_f.std()),
        "post_gr_iqr":          float(q75 - q25),
        "pre_gr_mean":          float(pre_gr.mean()),
        "pre_gr_std":           float(pre_gr.std()),
        "gr_mean_ratio_post_pre": float(gr_post_f.mean() / (pre_gr.mean() + 1.)),
    }


def build(hw, tw, well_id, split="train"):
    """Build feature dataframe for one well (v17 = v8 + xcorr features)."""
    n = len(hw)
    ps = get_ps(hw)
    gr = fill_arr(hw["GR"])
    md = hw["MD"].astype(float).values
    z  = hw["Z"].astype(float).values
    x  = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt     = tvt_inp.ffill().bfill().values
    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr   = fill_arr(tw["GR"])
    gr_s    = pd.Series(gr)

    # Pre-PS linear fit
    if split == "train" and "TVT" in hw.columns:
        pre_z   = z[:ps]
        pre_tvt = hw["TVT"].astype(float).values[:ps]
    else:
        known   = tvt_inp[:ps].values
        valid   = ~np.isnan(known)
        pre_z   = z[:ps][valid]
        pre_tvt = known[valid]

    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.
    else:
        slope  = -1.0
        pre_r2 = 0.

    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    post_z  = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post  = len(post_z)
    z_end   = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z       = z_end - Z_anchor
    total_md      = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.
    post_dz_rate  = total_z / total_md
    phys_tvt_end  = slope * total_z
    pre_dz_md     = ((z[ps-1] - z[0]) / (md[ps-1] - md[0])) if ps > 1 and abs(md[ps-1] - md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md

    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev        = gr - gr_at_physics

    # ── xcorr features (per well, broadcast to all rows) ─────────────────────
    xf = compute_xcorr_features(gr, anchored_physics, tw_tvt, tw_gr, ps)
    best_off = xf["xcorr_best_offset"]
    gr_at_xcorr = np.interp(anchored_physics + best_off, tw_tvt, tw_gr,
                             left=tw_gr[0], right=tw_gr[-1])
    gr_dev_xcorr = gr - gr_at_xcorr

    # ── v8 features ──────────────────────────────────────────────────────────
    feat = {"gr": gr, "gr_diff": np.gradient(gr), "gr_diff2": np.gradient(np.gradient(gr))}
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values

    feat.update({
        "md": md, "z": z,
        "dz_dmd":     np.gradient(z, md),
        "dx_dmd":     np.gradient(x, md),
        "dy_dmd":     np.gradient(y, md),
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
        "post_ps_frac":          np.where(n_post > 0,
                                     np.maximum(0, np.arange(n) - ps) / n_post, 0.).astype(float),
        "pre_r2": np.full(n, pre_r2),
        # v8 pre-PS dz features
    })
    if ps > 2:
        pre_dz  = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt    = tvt_inp[:ps].values
        vtvt    = vtvt[~np.isnan(vtvt)]
        pdtvt   = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, (pdtvt[-min(50, len(pdtvt)):].std()
                                               if len(pdtvt) > 1 else 0.))
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    # ── new v17 features ─────────────────────────────────────────────────────
    feat["gr_at_xcorr"]    = gr_at_xcorr
    feat["gr_dev_xcorr"]   = gr_dev_xcorr
    for k, v in xf.items():
        feat[k] = np.full(n, float(v))

    df = pd.DataFrame(feat)
    df["well_id"]    = well_id
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"]    = tvt_true - anchored_physics
        df["target_tvt"]           = tvt_true
        df["anchored_physics_col"] = anchored_physics

    return df


# ── Load train / val splits ────────────────────────────────────────────────
train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"Train: {len(train_ids)} wells, Val: {len(val_ids)} wells")

tr_dfs, vl_dfs = [], []
for i, wid in enumerate(train_ids):
    try:    tr_dfs.append(build(*load_well(wid), wid, "train"))
    except Exception as e: print(f"  SKIP {wid}: {e}")
    if (i + 1) % 100 == 0: print(f"  train {i+1}/{len(train_ids)}")

for wid in val_ids:
    try:    vl_dfs.append(build(*load_well(wid), wid, "train"))
    except Exception as e: print(f"  SKIP {wid}: {e}")

tr_all = pd.concat(tr_dfs, ignore_index=True)
vl_all = pd.concat(vl_dfs, ignore_index=True)

EXCL  = {"well_id", "target_correction", "target_tvt", "is_post_ps", "anchored_physics_col"}
FCOLS = [c for c in tr_all.columns if c not in EXCL]
print(f"Features: {len(FCOLS)}")

tr_p = tr_all[tr_all["is_post_ps"] == 1]
vl_p = vl_all[vl_all["is_post_ps"] == 1]
X_tr = tr_p[FCOLS].values.astype(np.float32); y_tr = tr_p["target_correction"].values.astype(np.float32)
X_vl = vl_p[FCOLS].values.astype(np.float32); y_vl = vl_p["target_correction"].values.astype(np.float32)

# ── LightGBM training ──────────────────────────────────────────────────────
params = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 255, "learning_rate": 0.02,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 20, "lambda_l1": 0.05, "lambda_l2": 0.05,
    "verbose": -1, "seed": 42, "n_jobs": -1,
}
lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=FCOLS)
lgb_vl = lgb.Dataset(X_vl, label=y_vl, feature_name=FCOLS, reference=lgb_tr)

print("Training LGBM v17 (xcorr features)...")
callbacks = [lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)]
model = lgb.train(params, lgb_tr, num_boost_round=15000,
                  valid_sets=[lgb_vl], callbacks=callbacks)

print(f"Best iteration: {model.best_iteration}")
joblib.dump(model, MODELS_DIR / "lgbm_v17_xcorr.pkl")
print("Model saved → models/lgbm_v17_xcorr.pkl")

# ── Per-well val evaluation ────────────────────────────────────────────────
feat_cols = model.feature_name()
rmses = []
for wid in val_ids:
    hw, tw = load_well(wid)
    ps = get_ps(hw)
    tvt_true = hw["TVT"].astype(float).values[ps:]
    df = build(hw, tw, wid, "train")
    df_p = df[df["is_post_ps"] == 1]
    corr = model.predict(df_p[feat_cols].values.astype(np.float32))
    pred = gaussian_filter1d(df_p["anchored_physics_col"].values + corr, sigma=1.)
    rmses.append(float(np.sqrt(np.mean((pred - tvt_true) ** 2))))

arr = np.array(rmses)
print(f"\nVal RMSE: mean={arr.mean():.3f} ft  median={np.median(arr):.3f} ft  (N={len(arr)})")

# ── Feature importance (top 20) ────────────────────────────────────────────
imp = pd.DataFrame({"feat": feat_cols, "imp": model.feature_importance("gain")})
imp = imp.sort_values("imp", ascending=False)
imp.to_csv(FEAT_DIR / "lgbm_v17_importance.csv", index=False)
print("\nTop 20 features:")
print(imp.head(20).to_string(index=False))

# ── Generate test predictions ──────────────────────────────────────────────
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
for wid in TEST_WELLS:
    hw, tw = load_well(wid, "test")
    df = build(hw, tw, wid, "test")
    df_p = df[df["is_post_ps"] == 1]
    corr = model.predict(df_p[feat_cols].values.astype(np.float32))
    tvt_pred = gaussian_filter1d(df_p["anchored_physics_col"].values + corr, sigma=1.)
    empty = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    rows = hw.index[empty].tolist()
    id_map = {f"{wid}_{i}": v for i, v in zip(rows, tvt_pred)}
    mask = sub["id"].isin(set(id_map.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)
    print(f"[{wid}] TVT=[{tvt_pred.min():.1f},{tvt_pred.max():.1f}] range={tvt_pred.max()-tvt_pred.min():.1f}")

missing = sub["tvt"].isna().sum()
print(f"\nTest predictions: {len(sub)} rows, {missing} missing")
if missing == 0:
    sub.to_csv(SUBS_DIR / "lgbm_v17_xcorr.csv", index=False)
    print("Submission saved → submissions/lgbm_v17_xcorr.csv")
else:
    print(f"WARNING: {missing} missing predictions")
    sub.to_csv(SUBS_DIR / "lgbm_v17_xcorr_debug.csv", index=False)
