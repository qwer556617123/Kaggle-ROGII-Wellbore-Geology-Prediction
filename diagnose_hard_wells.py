"""Diagnose hard wells: compare v8 vs v19 per-well RMSE on 115 val wells."""
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"


def fill_arr(series):
    return series.astype(float).ffill().bfill().fillna(0).values


def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def load_well(wid):
    return (pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv"),
            pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv"))


def build(hw, tw, well_id):
    """Build features (v19 = v8 + 4 horizontal features)."""
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

    pre_z   = z[:ps]
    pre_tvt = hw["TVT"].astype(float).values[:ps]

    if len(pre_z) > 5:
        reg  = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.
    else:
        slope = -1.0; pre_r2 = 0.

    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    post_z  = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post  = len(post_z)
    total_z  = (float(post_z[-1]) if n_post > 0 else float(z[-1])) - Z_anchor
    total_md = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.
    post_dz_rate   = total_z / total_md
    phys_tvt_end   = slope * total_z
    pre_dz_md      = ((z[ps-1]-z[0])/(md[ps-1]-md[0])) if ps > 1 and abs(md[ps-1]-md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md

    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])

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
        "inclination": np.arctan2(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
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
        "gr_dev_physics":  gr - gr_at_physics,
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
        vtvt   = tvt_inp[:ps].values; vtvt = vtvt[~np.isnan(vtvt)]
        pdtvt  = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.)
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    post_x = x[ps:] if ps < n else x[:]
    post_y = y[ps:] if ps < n else y[:]
    feat["post_x_range"] = np.full(n, float(post_x.max() - post_x.min()) if len(post_x) > 0 else 0.)
    feat["post_y_range"] = np.full(n, float(post_y.max() - post_y.min()) if len(post_y) > 0 else 0.)
    feat["post_x_rate"]  = np.full(n, float(post_x[-1] - post_x[0]) / total_md if total_md > 0 else 0.)
    feat["post_y_rate"]  = np.full(n, float(post_y[-1] - post_y[0]) / total_md if total_md > 0 else 0.)

    df = pd.DataFrame(feat)
    df["well_id"]          = well_id
    df["is_post_ps"]       = (np.arange(n) >= ps).astype(int)
    df["anchored_physics_col"] = anchored_physics
    return df


m8  = joblib.load(MODELS_DIR / "lgbm_v8.pkl")
m19 = joblib.load(MODELS_DIR / "lgbm_v19_horiz.pkl")
f8  = m8.feature_name()
f19 = m19.feature_name()

val_ids = pd.read_csv(FEAT_DIR / "val_ids.csv", header=None)[0].tolist()
rows = []
for wid in val_ids:
    try:
        hw, tw = load_well(wid)
        ps     = get_ps(hw)
        df     = build(hw, tw, wid)
        df_p   = df[df["is_post_ps"] == 1]
        tvt_true = hw["TVT"].astype(float).values[ps:]

        c8   = m8.predict(df_p[f8].values.astype(np.float32))
        p8   = gaussian_filter1d(df_p["anchored_physics_col"].values + c8, sigma=1.0)
        r8   = float(np.sqrt(np.mean((p8[:len(tvt_true)] - tvt_true[:len(p8)]) ** 2)))

        c19  = m19.predict(df_p[f19].values.astype(np.float32))
        p19  = gaussian_filter1d(df_p["anchored_physics_col"].values + c19, sigma=1.0)
        r19  = float(np.sqrt(np.mean((p19[:len(tvt_true)] - tvt_true[:len(p19)]) ** 2)))

        x  = hw["X"].astype(float).values
        y  = hw["Y"].astype(float).values
        px = x[ps:]; py = y[ps:]
        z  = hw["Z"].astype(float).values
        pz = z[ps:]
        md = hw["MD"].astype(float).values
        pm = md[ps:]
        total_md = float(pm[-1] - pm[0]) if len(pm) > 1 else 1.

        rows.append({
            "wid": wid, "v8": round(r8, 2), "v19": round(r19, 2),
            "delta": round(r19 - r8, 2),
            "post_x_rate": round(float(px[-1] - px[0]) / total_md, 4) if len(px) > 0 else 0.,
            "post_y_rate": round(float(py[-1] - py[0]) / total_md, 4) if len(py) > 0 else 0.,
            "post_x_range": round(float(px.max() - px.min()), 1) if len(px) > 0 else 0.,
            "post_y_range": round(float(py.max() - py.min()), 1) if len(py) > 0 else 0.,
            "post_dz_rate": round(float(pz[-1] - pz[0]) / total_md, 4) if len(pz) > 0 else 0.,
        })
    except Exception as e:
        print(f"FAIL {wid}: {e}")

df_res = pd.DataFrame(rows)
print(f"\nTotal wells: {len(df_res)}")
print(f"v8  mean={df_res['v8'].mean():.3f}  median={df_res['v8'].median():.3f}")
print(f"v19 mean={df_res['v19'].mean():.3f}  median={df_res['v19'].median():.3f}")
print(f"\nv19 better (delta<0): {(df_res['delta']<0).sum()} wells")
print(f"v19 worse  (delta>0): {(df_res['delta']>0).sum()} wells")
print(f"v19 worse by >2 ft:   {(df_res['delta']>2).sum()} wells")
print(f"v19 better by >2 ft:  {(df_res['delta']<-2).sum()} wells")

print("\n=== TOP 20 hardest wells (by v8 RMSE) ===")
top20 = df_res.sort_values("v8", ascending=False).head(20)
print(top20.to_string(index=False))

print("\n=== Wells where v19 helps most (delta < -5) ===")
helped = df_res[df_res["delta"] < -5].sort_values("delta")
print(helped.to_string(index=False) if len(helped) > 0 else "  None")

print("\n=== Wells where v19 hurts most (delta > 5) ===")
hurt = df_res[df_res["delta"] > 5].sort_values("delta", ascending=False)
print(hurt.to_string(index=False) if len(hurt) > 0 else "  None")
