"""Quick per-well val RMSE evaluation for lgbm_v20 (no retraining)."""
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
FEAT_DIR  = DATA_DIR / "features"


def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values


def load_well(wid):
    return (pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv"),
            pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv"))


def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def build(hw, tw, well_id, split="train"):
    n   = len(hw)
    ps  = get_ps(hw)
    gr  = fill_arr(hw["GR"])
    md  = hw["MD"].astype(float).values
    z   = hw["Z"].astype(float).values
    x   = hw["X"].astype(float).values
    y   = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt     = tvt_inp.ffill().bfill().values
    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr   = fill_arr(tw["GR"])
    gr_s    = pd.Series(gr)

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
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.0
        pre_resid_std_val = float(np.std(pre_tvt - pre_tvt_pred.flatten()))
    else:
        slope = -1.0
        pre_r2 = 0.0
        pre_resid_std_val = 30.0

    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    post_z  = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post  = len(post_z)
    z_end   = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z   = z_end - Z_anchor
    total_md  = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.0
    post_dz_rate   = total_z / total_md
    phys_tvt_end   = slope * total_z
    pre_dz_md      = (
        (z[ps - 1] - z[0]) / (md[ps - 1] - md[0])
        if ps > 1 and abs(md[ps - 1] - md[0]) > 0.01 else -0.01
    )
    dz_rate_change = post_dz_rate - pre_dz_md

    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr - gr_at_physics

    feat = {"gr": gr, "gr_diff": np.gradient(gr), "gr_diff2": np.gradient(np.gradient(gr))}
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values

    feat.update({
        "md": md, "z": z,
        "dz_dmd":      np.gradient(z, md),
        "dx_dmd":      np.gradient(x, md),
        "dy_dmd":      np.gradient(y, md),
        "inclination": np.arctan2(
            np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
            np.abs(np.gradient(z)) + 1e-9
        ) * 180 / np.pi,
        "raw_dz":  np.diff(z,  prepend=z[0]),
        "raw_dmd": np.diff(md, prepend=md[0]),
    })

    feat.update({
        "last_known_tvt": lkt,
        "rows_since_ps":  np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps": np.maximum(0, ps - np.arange(n)).astype(float),
        "ps_idx":         float(ps),
        "tvt_z_slope":    np.full(n, slope),
        "physics_tvt":    anchored_physics,
        "physics_vs_lkt": anchored_physics - lkt,
        "physics_dtvt":   slope * np.diff(z, prepend=z[0]),
        "gr_at_physics":  gr_at_physics,
        "gr_dev_physics": gr_dev,
    })

    feat.update({
        "total_z_change_post": np.full(n, total_z),
        "physics_tvt_at_end":  np.full(n, phys_tvt_end),
        "post_dz_rate":        np.full(n, post_dz_rate),
        "dz_rate_change":      np.full(n, dz_rate_change),
        "n_post_ps":           np.full(n, float(n_post)),
        "post_ps_frac":        np.where(
            n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.0
        ).astype(float),
        "pre_r2": np.full(n, pre_r2),
    })

    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.0])
        vtvt   = tvt_inp[:ps].values
        vtvt   = vtvt[~np.isnan(vtvt)]
        pdtvt  = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.0])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.0)
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    x_anchor = x[max(0, ps - 1)]
    y_anchor = y[max(0, ps - 1)]
    feat["x_disp"] = x - x_anchor
    feat["y_disp"] = y - y_anchor

    gr_dev_cumsum = np.zeros(n)
    for i in range(ps, n):
        gr_dev_cumsum[i] = gr_dev_cumsum[i - 1] + gr_dev[i]
    feat["gr_dev_cumsum"] = gr_dev_cumsum

    feat["pre_resid_std"] = np.full(n, pre_resid_std_val)

    dtvt       = 5.0
    tw_gr_plus  = np.interp(anchored_physics + dtvt, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    tw_gr_minus = np.interp(anchored_physics - dtvt, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    tw_gr_grad  = (tw_gr_plus - tw_gr_minus) / (2.0 * dtvt)
    safe_grad   = np.where(np.abs(tw_gr_grad) >= 0.2, tw_gr_grad, np.sign(tw_gr_grad + 1e-9) * 0.2)
    feat["tw_gr_gradient"]     = tw_gr_grad
    feat["tw_gr_gradient_mag"] = np.abs(tw_gr_grad)
    feat["gr_tvt_correction"]  = np.clip(gr_dev / safe_grad, -300.0, 300.0)

    gr_dev_running_mean = np.zeros(n)
    for i in range(ps, n):
        gr_dev_running_mean[i] = gr_dev_cumsum[i] / max(1, i - ps + 1)
    feat["gr_dev_running_mean"] = gr_dev_running_mean

    post_x = x[ps:] if ps < n else x[:]
    post_y = y[ps:] if ps < n else y[:]
    feat["post_x_range"] = np.full(n, float(post_x.max() - post_x.min()) if len(post_x) else 0.0)
    feat["post_y_range"] = np.full(n, float(post_y.max() - post_y.min()) if len(post_y) else 0.0)
    feat["post_x_rate"]  = np.full(n, float(post_x[-1] - post_x[0]) / total_md if total_md > 0 else 0.0)
    feat["post_y_rate"]  = np.full(n, float(post_y[-1] - post_y[0]) / total_md if total_md > 0 else 0.0)

    df = pd.DataFrame(feat)
    df["well_id"]    = well_id
    df["row_idx"]    = np.arange(n, dtype=np.int32)
    df["is_post_ps"] = (np.arange(n) >= ps).astype(np.int8)
    df["anchored_physics_col"] = anchored_physics.astype(np.float32)

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"] = (tvt_true - anchored_physics).astype(np.float32)
        df["target_tvt"]        = tvt_true.astype(np.float32)

    return df


def add_derived_gr(df_p, PHYS="physics_tvt"):
    df_p = df_p.copy()
    for sigma in [0, 3, 7]:
        for radius in [15, 30, 50, 80]:
            src   = f"gr_val_tvt_s{sigma}_r{radius}"
            fname = f"gr_val_corr_s{sigma}_r{radius}"
            if src in df_p.columns and fname not in df_p.columns:
                df_p[fname] = df_p[src] - df_p[PHYS]
    for w in [200, 500, 1000]:
        src   = f"local_xcorr_tvt_{w}"
        fname = f"local_xcorr_corr_{w}"
        if src in df_p.columns and fname not in df_p.columns:
            df_p[fname] = df_p[src] - df_p[PHYS]
    if "xcorr_global_corrected_tvt" in df_p.columns and "xcorr_global_corr_to_phys" not in df_p.columns:
        df_p["xcorr_global_corr_to_phys"] = df_p["xcorr_global_corrected_tvt"] - df_p[PHYS]
    return df_p


# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model and GR val features...")
model = joblib.load(DATA_DIR / "models" / "lgbm_v20.pkl")
FCOLS = model.feature_name()
print(f"  best_iter={model.best_iteration}  features={len(FCOLS)}")

with open(FEAT_DIR / "gr_xcorr_features_val.pkl", "rb") as f:
    gr_val = pickle.load(f)

# GR cols to merge (exclude metadata)
EXCL_GR = {"well_id", "is_post_ps", "row_idx", "ps_idx", "last_known_tvt",
            "physics_tvt", "target_tvt", "target_tvt_delta", "target_correction"}
GR_FCOLS = [c for c in gr_val.columns if c not in EXCL_GR]

val_ids = pd.read_csv(FEAT_DIR / "val_ids.csv", header=None)[0].tolist()
print(f"  Val wells: {len(val_ids)}")

# ── Per-well RMSE ──────────────────────────────────────────────────────────────
rmses = []
well_rmse_list = []

for wid in val_ids:
    try:
        hw, tw  = load_well(wid)
        df_base = build(hw, tw, wid, "train")
        gr_w    = gr_val[gr_val["well_id"] == wid].copy()
        gr_sub  = gr_w[["well_id", "row_idx"] + GR_FCOLS]
        df_base = df_base.merge(gr_sub, on=["well_id", "row_idx"], how="left")
        df_p    = df_base[df_base["is_post_ps"] == 1].copy()
        df_p[GR_FCOLS] = df_p[GR_FCOLS].fillna(0.0)
        df_p    = add_derived_gr(df_p)

        corr     = model.predict(df_p[FCOLS].values.astype(np.float32))
        tvt_pred = gaussian_filter1d(df_p["anchored_physics_col"].values + corr, sigma=1.0)
        ps       = get_ps(hw)
        tvt_true = hw["TVT"].astype(float).values[ps:]
        n_min    = min(len(tvt_pred), len(tvt_true))
        r = float(np.sqrt(np.mean((tvt_pred[:n_min] - tvt_true[:n_min]) ** 2)))
        rmses.append(r)
        well_rmse_list.append((wid, r))
    except Exception as e:
        print(f"  SKIP {wid}: {e}")

rmses = np.array(rmses)
overall = rmses.mean()

print(f"\n{'='*55}")
print(f"  Overall Val RMSE  : {overall:.4f} ft")
print(f"  Median  Val RMSE  : {np.median(rmses):.4f} ft")
print(f"  Std     Val RMSE  : {rmses.std():.4f} ft")
print(f"  Min/Max Val RMSE  : {rmses.min():.3f} / {rmses.max():.3f} ft")
print(f"{'='*55}")
print(f"  v8  baseline      : 11.983 ft")
print(f"  v21 baseline      : 11.983 ft")
delta = overall - 11.983
print(f"  Delta vs baseline : {delta:+.3f} ft  ({'BETTER' if delta < 0 else 'WORSE'})")
print(f"{'='*55}")

well_rmse_list.sort(key=lambda t: -t[1])
print(f"\nPer-well RMSE (worst → best):")
print(f"  {'#':<4} {'well_id':<12} {'RMSE':>10}")
print("  " + "-"*28)
for rank, (wid, r) in enumerate(well_rmse_list, 1):
    flag = " ◄" if r > 30 else ""
    print(f"  {rank:<4} {wid:<12} {r:>10.3f}{flag}")
