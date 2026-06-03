"""
LightGBM v23 — Spatial Neighbor 3D Correction Features (v8 hyperparams)
========================================================================

Same architecture as v22, but uses v8's proven hyperparameters:
  lr=0.02, num_leaves=255, lambda_l1/l2=0.05, early_stop=500, N_EST=20000

v22 used lr=0.05, num_leaves=127 → only 601 rounds, model undertrained.
With v8 hyperparams, LGBM has enough capacity to learn the interaction
between nbr_count_consistent_x and nbr_3d_corr (sign-mismatch guard).

Architecture
------------
  Base features: v8 build() function (46 features)
  + 6 new spatial neighbor features = 52 features total:
      nbr_alpha_x            — weighted avg formation dip (X) from K=10 nearest train wells
      nbr_alpha_y            — weighted avg formation dip (Y) from K=10 nearest train wells
      nbr_3d_corr            — alpha_x*(x_i-x_ps) + alpha_y*(y_i-y_ps) per row
      nbr_dist_min           — distance to nearest training neighbor (ft)
      nbr_r2_mean            — mean r2_xy of top-K neighbors (reliability signal)
      nbr_count_consistent_x — fraction of K neighbors with same-sign alpha_x

Target : target_correction = tvt_true − anchored_physics
Predict: tvt = anchored_physics + correction

Train: 658 wells / Val: 115 wells
Baseline: v8 val RMSE ≈ 11.983 ft
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

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
FEAT_DIR   = DATA_DIR / "features"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

K_NEIGHBORS       = 10
MIN_DIST_CAP      = 100.0   # ft — minimum distance cap for IDW
MIN_R2_QUALITY    = 0.3
SMOOTH_SIGMA      = 1.0     # Gaussian smoothing for test predictions

# ── LightGBM hyperparameters — v8's proven settings ───────────────────────────
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        255,
    "learning_rate":     0.02,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "lambda_l1":         0.05,
    "lambda_l2":         0.05,
    "verbose":           -1,
    "seed":              42,
    "n_jobs":            -1,
}
N_ESTIMATORS          = 20000
EARLY_STOPPING_ROUNDS = 500

print("=" * 70)
print("LGBM v23 — Spatial Neighbor 3D Correction (v8 hyperparams)")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# v8 feature builder (copied verbatim from lgbm_v8_train.py build())
# ══════════════════════════════════════════════════════════════════════════════

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values


def load_well(wid, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (pd.read_csv(d / f"{wid}__horizontal_well.csv"),
            pd.read_csv(d / f"{wid}__typewell.csv"))


def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def build(hw, tw, well_id, split="train"):
    """Build v8 feature dataframe for one well (46 base features)."""
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

    # Pre-PS linear fit: TVT ~ slope * Z + intercept
    if split == "train" and "TVT" in hw.columns:
        pre_z   = z[:ps]
        pre_tvt = hw["TVT"].astype(float).values[:ps]
    else:
        known  = tvt_inp[:ps].values
        valid  = ~np.isnan(known)
        pre_z  = z[:ps][valid]
        pre_tvt = known[valid]

    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.
    else:
        slope  = -1.0
        pre_r2 = 0.

    # Anchored physics: anchor_tvt at PS (avoids intercept bias)
    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    # Post-PS trajectory features
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

    # Typewell GR at physics TVT
    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev        = gr - gr_at_physics

    # GR rolling statistics
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
    })

    if ps > 2:
        pre_dz  = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt    = tvt_inp[:ps].values
        vtvt    = vtvt[~np.isnan(vtvt)]
        pdtvt   = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]    = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"]   = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]    = np.full(n, (pdtvt[-min(50, len(pdtvt)):].std()
                                                  if len(pdtvt) > 1 else 0.))
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    df = pd.DataFrame(feat)
    df["well_id"]    = well_id
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    # Store anchored_physics for prediction reconstruction
    df["anchored_physics_col"] = anchored_physics.astype(np.float32)

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"] = tvt_true - anchored_physics
        df["target_tvt"]        = tvt_true

    # Store X, Y for spatial feature computation
    df["_x"] = x
    df["_y"] = y

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load dip coefficients and build neighbor pool
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading well dip coefficients...")

dip_df     = pd.read_csv(FEAT_DIR / "well_dip_coeffs.csv")
train_ids  = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids    = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"  Loaded {len(dip_df)} wells  (r2_xy < 0.3: {(dip_df['r2_xy'] < MIN_R2_QUALITY).sum()})")
print(f"  Train: {len(train_ids)} wells | Val: {len(val_ids)} wells")

# Neighbor pool = training wells with good r2_xy (658-pool, matching test_well_nbr_stats.json)
train_pool = dip_df[
    dip_df["wid"].isin(train_ids) & (dip_df["r2_xy"] >= MIN_R2_QUALITY)
].copy().reset_index(drop=True)
print(f"  Neighbor pool: {len(train_pool)} training wells (r2_xy >= {MIN_R2_QUALITY})")

pool_cx   = train_pool["cx"].values
pool_cy   = train_pool["cy"].values
pool_ax   = train_pool["alpha_x"].values
pool_ay   = train_pool["alpha_y"].values
pool_r2   = train_pool["r2_xy"].values
pool_wids = train_pool["wid"].values

dip_lookup = dip_df.set_index("wid")[["cx", "cy"]].to_dict("index")


def compute_well_neighbor_features(wid: str, cx_well: float, cy_well: float) -> dict:
    """Find K nearest training neighbors (excl. self) and return weighted dip stats."""
    dist = np.sqrt((pool_cx - cx_well) ** 2 + (pool_cy - cy_well) ** 2)
    dist[pool_wids == wid] = np.inf

    k = min(K_NEIGHBORS, int(np.isfinite(dist).sum()))
    nearest_idx = np.argpartition(dist, k)[:k]
    nearest_idx = nearest_idx[np.argsort(dist[nearest_idx])]
    nearest_dist = dist[nearest_idx]

    w = 1.0 / np.maximum(nearest_dist, MIN_DIST_CAP)
    w /= w.sum()

    nbr_alpha_x = float(np.dot(w, pool_ax[nearest_idx]))
    nbr_alpha_y = float(np.dot(w, pool_ay[nearest_idx]))
    nbr_dist_min = float(nearest_dist[0])
    nbr_r2_mean  = float(np.mean(pool_r2[nearest_idx]))

    if nbr_alpha_x == 0.0:
        nbr_count_consistent_x = 0.5
    else:
        nbr_count_consistent_x = float(
            np.mean(np.sign(pool_ax[nearest_idx]) == np.sign(nbr_alpha_x))
        )

    return {
        "nbr_alpha_x":            nbr_alpha_x,
        "nbr_alpha_y":            nbr_alpha_y,
        "nbr_dist_min":           nbr_dist_min,
        "nbr_r2_mean":            nbr_r2_mean,
        "nbr_count_consistent_x": nbr_count_consistent_x,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Precompute per-well neighbor stats
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Precomputing spatial neighbor stats for all wells...")

all_wids = list(dict.fromkeys(train_ids + val_ids + TEST_WELLS))  # dedup preserving order
well_nbr_stats = {}
for i, wid in enumerate(all_wids):
    if wid in dip_lookup:
        cx = dip_lookup[wid]["cx"]
        cy = dip_lookup[wid]["cy"]
        well_nbr_stats[wid] = compute_well_neighbor_features(wid, cx, cy)
    else:
        well_nbr_stats[wid] = {
            "nbr_alpha_x": 0.0, "nbr_alpha_y": 0.0,
            "nbr_dist_min": 9999.0, "nbr_r2_mean": 0.0,
            "nbr_count_consistent_x": 0.5,
        }
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_wids)} wells processed", flush=True)

print(f"  Done: {len(well_nbr_stats)} wells")

# Diagnostics
hard_wells_diag = ["ba48188d", "00bbac68", "1b1eba53", "389ae58f", "81bf5923"]
print("\n  Neighbor stats for key wells:")
print(f"  {'Well':<12}  {'nbr_ax':>8}  {'nbr_ay':>8}  {'dist_min':>10}  {'r2_mean':>8}  {'consist':>8}")
for hw_ in hard_wells_diag:
    if hw_ in well_nbr_stats:
        s = well_nbr_stats[hw_]
        print(f"  {hw_:<12}  {s['nbr_alpha_x']:>8.4f}  {s['nbr_alpha_y']:>8.4f}  "
              f"{s['nbr_dist_min']:>10.1f}  {s['nbr_r2_mean']:>8.4f}  "
              f"{s['nbr_count_consistent_x']:>8.3f}")


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 spatial neighbor features to a well dataframe.
    Requires columns: well_id, ps_idx, _x (raw X), _y (raw Y).
    """
    wid = df["well_id"].iloc[0]
    ps  = int(df["ps_idx"].iloc[0])
    n   = len(df)

    stats  = well_nbr_stats.get(wid, {
        "nbr_alpha_x": 0.0, "nbr_alpha_y": 0.0,
        "nbr_dist_min": 9999.0, "nbr_r2_mean": 0.0,
        "nbr_count_consistent_x": 0.5,
    })
    nbr_ax = stats["nbr_alpha_x"]
    nbr_ay = stats["nbr_alpha_y"]

    x_arr = df["_x"].values
    y_arr = df["_y"].values
    x_ps  = x_arr[ps] if ps < n else x_arr[-1]
    y_ps  = y_arr[ps] if ps < n else y_arr[-1]
    dx    = x_arr - x_ps
    dy    = y_arr - y_ps

    df = df.copy()
    df["nbr_alpha_x"]            = np.float32(nbr_ax)
    df["nbr_alpha_y"]            = np.float32(nbr_ay)
    df["nbr_3d_corr"]            = (nbr_ax * dx + nbr_ay * dy).astype(np.float32)
    df["nbr_dist_min"]           = np.float32(stats["nbr_dist_min"])
    df["nbr_r2_mean"]            = np.float32(stats["nbr_r2_mean"])
    df["nbr_count_consistent_x"] = np.float32(stats["nbr_count_consistent_x"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build train / val feature matrices
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Building feature matrices (v8 + spatial)...")

SPATIAL_FEATS = [
    "nbr_alpha_x", "nbr_alpha_y", "nbr_3d_corr",
    "nbr_dist_min", "nbr_r2_mean", "nbr_count_consistent_x",
]

tr_dfs, vl_dfs = [], []
EXCL = {"well_id", "target_correction", "target_tvt", "is_post_ps",
        "anchored_physics_col", "_x", "_y"}

for i, wid in enumerate(train_ids):
    try:
        df = build(*load_well(wid), wid, "train")
        df = add_spatial_features(df)
        tr_dfs.append(df)
    except Exception as e:
        print(f"  SKIP train/{wid}: {e}")
    if (i + 1) % 100 == 0:
        print(f"  train {i+1}/{len(train_ids)}", flush=True)

print(f"  Train wells built: {len(tr_dfs)}")

for wid in val_ids:
    try:
        df = build(*load_well(wid), wid, "train")
        df = add_spatial_features(df)
        vl_dfs.append(df)
    except Exception as e:
        print(f"  SKIP val/{wid}: {e}")

print(f"  Val wells built:   {len(vl_dfs)}")

tr_all = pd.concat(tr_dfs, ignore_index=True)
vl_all = pd.concat(vl_dfs, ignore_index=True)

# Feature columns
FCOLS = [c for c in tr_all.columns if c not in EXCL]
# Ensure spatial features are at the end and present
for sf in SPATIAL_FEATS:
    if sf not in FCOLS:
        FCOLS.append(sf)
print(f"  Features: {len(FCOLS)}  ({len(FCOLS) - len(SPATIAL_FEATS)} base + {len(SPATIAL_FEATS)} spatial)")

# Filter to post-PS rows
tr_p = tr_all[tr_all["is_post_ps"] == 1]
vl_p = vl_all[vl_all["is_post_ps"] == 1]
print(f"  Post-PS rows — Train: {len(tr_p):,}  Val: {len(vl_p):,}")

# Target: correction = true_tvt - anchored_physics  (same as v8)
TARGET = "target_correction"

X_tr = tr_p[FCOLS].values.astype(np.float32)
y_tr = tr_p[TARGET].values.astype(np.float32)
X_vl = vl_p[FCOLS].values.astype(np.float32)
y_vl = vl_p[TARGET].values.astype(np.float32)

print(f"  X_tr: {X_tr.shape}  y_tr range: [{y_tr.min():.2f}, {y_tr.max():.2f}]  mean: {y_tr.mean():.4f}")
print(f"  X_vl: {X_vl.shape}  y_vl range: [{y_vl.min():.2f}, {y_vl.max():.2f}]  mean: {y_vl.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Train LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/5] Training LightGBM (n={N_ESTIMATORS}, early_stop={EARLY_STOPPING_ROUNDS})...")

lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=FCOLS, free_raw_data=False)
lgb_vl = lgb.Dataset(X_vl, label=y_vl, feature_name=FCOLS, reference=lgb_tr, free_raw_data=False)

model = lgb.train(
    LGBM_PARAMS,
    lgb_tr,
    num_boost_round=N_ESTIMATORS,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(500),
    ],
)

best_iter = model.best_iteration
best_rmse = model.best_score["valid_0"]["rmse"]
print(f"\n  Best iteration : {best_iter}")
print(f"  Best val RMSE  : {best_rmse:.4f} ft  (correction target)")

# Feature importances
fi = pd.DataFrame({
    "feature":    FCOLS,
    "importance": model.feature_importance("gain"),
}).sort_values("importance", ascending=False)
fi.to_csv(MODELS_DIR / "lgbm_v23_spatial_importance.csv", index=False)
print("\n  Top-20 feature importances:")
print(fi.head(20).to_string(index=False))

# Save model
model_meta = {
    "model":         model,
    "feature_cols":  FCOLS,
    "spatial_feats": SPATIAL_FEATS,
    "target":        TARGET,
    "best_iter":     best_iter,
    "best_rmse":     best_rmse,
}
joblib.dump(model_meta, MODELS_DIR / "lgbm_v23_spatial.pkl")
print(f"\n  Saved: models/lgbm_v23_spatial.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Validate + generate test predictions
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Validating on all 115 val wells...")

per_well_rmse = []
for wid in val_ids:
    wdf_p = vl_p[vl_p["well_id"] == wid]
    if len(wdf_p) == 0:
        continue
    corr_pred = model.predict(wdf_p[FCOLS].values.astype(np.float32))
    tvt_pred  = gaussian_filter1d(
        wdf_p["anchored_physics_col"].values + corr_pred, sigma=SMOOTH_SIGMA
    )
    tvt_true  = wdf_p["target_tvt"].values
    rmse_w    = float(np.sqrt(np.mean((tvt_pred[:len(tvt_true)] - tvt_true[:len(tvt_pred)]) ** 2)))
    per_well_rmse.append({"well_id": wid, "rmse": rmse_w, "n_rows": len(wdf_p)})

pw_df = pd.DataFrame(per_well_rmse).sort_values("rmse", ascending=False)

# Overall weighted RMSE
all_corr_pred = model.predict(X_vl)
all_tvt_pred  = vl_p["anchored_physics_col"].values + all_corr_pred
all_tvt_true  = vl_p["target_tvt"].values
overall_rmse  = float(np.sqrt(np.mean((all_tvt_pred - all_tvt_true) ** 2)))

print(f"\n  Overall val RMSE: {overall_rmse:.4f} ft   (v8 baseline: 11.983 ft)")
print(f"  Per-well — mean: {pw_df['rmse'].mean():.4f}  median: {pw_df['rmse'].median():.4f}")
print(f"\n  Worst 20 wells (sorted worst → best):")
print(f"  {'Well':<14}  {'RMSE':>8}  {'n_rows':>8}")
for _, row in pw_df.head(20).iterrows():
    print(f"  {row['well_id']:<14}  {row['rmse']:>8.3f}  {int(row['n_rows']):>8}")

print(f"\n  Best 10 wells:")
print(f"  {'Well':<14}  {'RMSE':>8}")
for _, row in pw_df.tail(10).iterrows():
    print(f"  {row['well_id']:<14}  {row['rmse']:>8.3f}")

# Hard wells spotlight
print("\n  Hard wells (baseline approx from v8 results):")
baselines = {"ba48188d": 51.6, "1b1eba53": 45.0, "389ae58f": 30.0, "81bf5923": 25.0}
for wid in ["ba48188d", "1b1eba53", "389ae58f", "81bf5923"]:
    match = pw_df[pw_df["well_id"] == wid]
    if len(match):
        r = float(match["rmse"].iloc[0])
        b = baselines.get(wid, "?")
        delta = (r - b) if isinstance(b, float) else 0.0
        tag = "BETTER" if delta < 0 else "worse"
        print(f"  {wid}:  v23={r:.3f} ft  baseline~{b} ft  Δ={delta:+.1f} ft [{tag}]")

# ── Test predictions ──────────────────────────────────────────────────────────
print("\n  Generating test predictions...")
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sub_ids_set = set(sub["id"].values)

for wid in TEST_WELLS:
    try:
        hw, tw = load_well(wid, "test")
        df   = build(hw, tw, wid, "test")
        df   = add_spatial_features(df)
        df_p = df[df["is_post_ps"] == 1]

        corr_pred = model.predict(df_p[FCOLS].values.astype(np.float32))
        tvt_pred  = gaussian_filter1d(
            df_p["anchored_physics_col"].values + corr_pred, sigma=SMOOTH_SIGMA
        )

        # submission row index = absolute row index in test CSV (0..N-1)
        ps = int(df["ps_idx"].iloc[0])
        empty_rows = hw.index[
            hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
        ].tolist()

        id_map = {f"{wid}_{i}": float(v) for i, v in zip(empty_rows, tvt_pred)
                  if f"{wid}_{i}" in sub_ids_set}

        mask = sub["id"].isin(set(id_map.keys()))
        sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)

        tvt_arr = np.array(list(id_map.values()))
        print(f"  [{wid}] {len(id_map)} rows  TVT=[{tvt_arr.min():.1f}, {tvt_arr.max():.1f}]  "
              f"range={tvt_arr.max()-tvt_arr.min():.1f} ft")

        s = well_nbr_stats.get(wid, {})
        nbr_c = df_p["nbr_3d_corr"].values
        print(f"    nbr_ax={s.get('nbr_alpha_x',0):.4f}  nbr_ay={s.get('nbr_alpha_y',0):.4f}  "
              f"nbr_3d_corr=[{nbr_c.min():.1f}, {nbr_c.max():.1f}]  "
              f"dist_min={s.get('nbr_dist_min',9999):.0f} ft")
    except Exception as e:
        print(f"  [{wid}] ERROR: {e}")

nan_count = sub["tvt"].isna().sum()
print(f"\n  Submission NaN: {nan_count}")
sub.to_csv(SUBS_DIR / "lgbm_v23_spatial.csv", index=False)
print(f"  Saved: submissions/lgbm_v23_spatial.csv")

print("\n" + "=" * 70)
print(f"  FINAL: val RMSE = {overall_rmse:.4f} ft")
print(f"  vs v8 baseline:    11.983 ft")
print(f"  Improvement:       {11.983 - overall_rmse:+.4f} ft")
print("=" * 70)
