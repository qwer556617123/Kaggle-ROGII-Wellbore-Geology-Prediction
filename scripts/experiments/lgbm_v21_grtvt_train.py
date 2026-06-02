"""
LightGBM v21 — v8 + GR-physics integration features

New features vs v8:
  tw_gr_gradient:    local dGR/dTVT of typewell at physics TVT (per row)
  gr_tvt_correction: gr_dev / tw_gr_gradient — direct GR-based TVT correction estimate
  gr_dev_running_mean: cumulative mean of gr_dev since PS (systematic drift indicator)
  x_disp:            cumulative X displacement from PS anchor (per row)
  y_disp:            cumulative Y displacement from PS anchor (per row)
  pre_resid_std:     std of pre-PS TVT residuals

Key insight:
  tw_gr_gradient = how fast the typewell GR changes with TVT at current physics position
  gr_tvt_correction = gr_dev / tw_gr_gradient = estimated TVT offset from GR signal alone
  - If physics is accurate (81bf5923): gr_dev ≈ 0 → gr_tvt_correction ≈ 0 (no overcorrection)
  - If dipping (ba48188d): gr_dev grows → gr_tvt_correction gives non-zero directional estimate
  - Where typewell is GR-flat (gradient ≈ 0): clipped to prevent instability

Target: correction = true_tvt - anchored_physics
Train: 658 wells / Val: 115 wells
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


def build(hw, tw, well_id, split="train"):
    """Build feature dataframe for one well (v21 features)."""
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
        pre_resid_std_val = float(np.std(pre_tvt - pre_tvt_pred.flatten()))
    else:
        slope  = -1.0
        pre_r2 = 0.
        pre_resid_std_val = 30.

    # Anchored physics
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

    # ── NEW v21: Typewell GR gradient at physics TVT (local dGR/dTVT) ────────
    # Computed via central difference on typewell at ±5 ft from physics TVT
    dtvt = 5.0
    tw_gr_plus  = np.interp(anchored_physics + dtvt, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    tw_gr_minus = np.interp(anchored_physics - dtvt, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    tw_gr_gradient = (tw_gr_plus - tw_gr_minus) / (2.0 * dtvt)  # API/ft

    # ── NEW v21: GR-based TVT correction estimate ─────────────────────────────
    # gr_tvt_correction = how much TVT correction is implied by GR mismatch
    # Clipped: where typewell is GR-flat (gradient near 0), correction is unreliable
    MIN_GRADIENT = 0.2  # API/ft — below this, GR gives unreliable TVT signal
    safe_gradient = np.where(
        np.abs(tw_gr_gradient) >= MIN_GRADIENT,
        tw_gr_gradient,
        np.sign(tw_gr_gradient + 1e-9) * MIN_GRADIENT
    )
    gr_tvt_correction = np.clip(gr_dev / safe_gradient, -300.0, 300.0)
    # Also track where gradient is informative (high gradient = GR-rich TVT region)
    tw_gr_gradient_mag = np.abs(tw_gr_gradient)

    # Feature dict — GR rolling stats
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
        # v21 new features
        "tw_gr_gradient":       tw_gr_gradient,
        "tw_gr_gradient_mag":   tw_gr_gradient_mag,
        "gr_tvt_correction":    gr_tvt_correction,
        "pre_resid_std":        np.full(n, pre_resid_std_val),
    })

    # Pre-PS slope/TVT increment stats
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

    # XY cumulative displacement from PS anchor (per row)
    x_anchor = x[max(0, ps - 1)]
    y_anchor = y[max(0, ps - 1)]
    feat["x_disp"] = x - x_anchor
    feat["y_disp"] = y - y_anchor

    # Well-level horizontal direction features
    post_x = x[ps:] if ps < n else x[:]
    post_y = y[ps:] if ps < n else y[:]
    post_x_range = float(post_x.max() - post_x.min()) if len(post_x) > 0 else 0.
    post_y_range = float(post_y.max() - post_y.min()) if len(post_y) > 0 else 0.
    post_x_rate  = float(post_x[-1] - post_x[0]) / total_md if total_md > 0 else 0.
    post_y_rate  = float(post_y[-1] - post_y[0]) / total_md if total_md > 0 else 0.
    feat["post_x_range"] = np.full(n, post_x_range)
    feat["post_y_range"] = np.full(n, post_y_range)
    feat["post_x_rate"]  = np.full(n, post_x_rate)
    feat["post_y_rate"]  = np.full(n, post_y_rate)

    # Running mean of GR deviation since PS (causal, bounded scale)
    gr_dev_cumsum = np.zeros(n)
    gr_dev_running_mean = np.zeros(n)
    for i in range(ps, n):
        gr_dev_cumsum[i] = gr_dev_cumsum[i - 1] + gr_dev[i]
        gr_dev_running_mean[i] = gr_dev_cumsum[i] / max(1, i - ps + 1)
    feat["gr_dev_running_mean"] = gr_dev_running_mean

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


# ── LightGBM training ─────────────────────────────────────────────────────
params = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 255, "learning_rate": 0.02,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 20, "lambda_l1": 0.05, "lambda_l2": 0.05,
    "verbose": -1, "seed": 42, "n_jobs": -1,
}
lgb_tr  = lgb.Dataset(X_tr, label=y_tr, feature_name=FCOLS)
lgb_vl  = lgb.Dataset(X_vl, label=y_vl, feature_name=FCOLS, reference=lgb_tr)
model   = lgb.train(params, lgb_tr, num_boost_round=15000,
                    valid_sets=[lgb_vl],
                    callbacks=[lgb.early_stopping(500, verbose=False),
                               lgb.log_evaluation(2000)])

print(f"Best iter: {model.best_iteration}  val RMSE={model.best_score['valid_0']['rmse']:.4f}")
joblib.dump(model, MODELS_DIR / "lgbm_v21_grtvt.pkl")

# Feature importance
fi = pd.DataFrame({"feature": FCOLS, "importance": model.feature_importance("gain")})
fi = fi.sort_values("importance", ascending=False)
fi.to_csv(MODELS_DIR / "lgbm_v21_importance.csv", index=False)
print(fi.head(20).to_string(index=False))

# Per-well val RMSE
rmses = []
well_rmse_list = []
for wid in val_ids:
    try:
        hw, tw = load_well(wid)
        ps = get_ps(hw)
        df   = build(hw, tw, wid, "train")
        df_p = df[df["is_post_ps"] == 1]
        corr     = model.predict(df_p[FCOLS].values.astype(np.float32))
        tvt_pred = gaussian_filter1d(df_p["anchored_physics_col"].values + corr, sigma=1.0)
        tvt_true = hw["TVT"].astype(float).values[ps:]
        r = np.sqrt(np.mean((tvt_pred[:len(tvt_true)] - tvt_true[:len(tvt_pred)]) ** 2))
        rmses.append(r)
        well_rmse_list.append((wid, r))
    except Exception as e:
        print(f"  SKIP {wid}: {e}")
rmses = np.array(rmses)
print(f"\nPer-well val RMSE: mean={rmses.mean():.3f}  median={np.median(rmses):.3f}")
well_rmse_list.sort(key=lambda t: -t[1])
print("Top 10 hardest:")
for wid, r in well_rmse_list[:10]:
    print(f"  {wid}  {r:.2f}")

# ── Generate submission ────────────────────────────────────────────────────
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
for wid in TEST_WELLS:
    hw, tw = load_well(wid, "test")
    ps = get_ps(hw)
    df   = build(hw, tw, wid, "test")
    df_p = df[df["is_post_ps"] == 1]
    corr     = model.predict(df_p[FCOLS].values.astype(np.float32))
    tvt_pred = gaussian_filter1d(df_p["physics_tvt"].values + corr, sigma=1.0)
    empty  = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    rows   = hw.index[empty].tolist()
    id_map = {f"{wid}_{i}": v for i, v in zip(rows, tvt_pred)}
    mask   = sub["id"].isin(set(id_map.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)
    print(f"[{wid}] TVT=[{tvt_pred.min():.1f}, {tvt_pred.max():.1f}] range={tvt_pred.max()-tvt_pred.min():.1f}")

sub.to_csv(SUBS_DIR / "lgbm_v21_grtvt.csv", index=False)
print(f"Saved lgbm_v21_grtvt.csv  NaN={sub.tvt.isna().sum()}")
