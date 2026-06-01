import numpy as np, pandas as pd, joblib, re
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
np.random.seed(42)

MODEL_PATH = "/kaggle/input/datasets/qwer556617123/rogii-lgbm-model/lgbm_final_reg.pkl"
TEST_DIR = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction/test")

model = joblib.load(MODEL_PATH)
FCOLS = model.feature_name()
print("Model loaded, features:", len(FCOLS))

# Find all test wells from horizontal_well files
hw_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
well_ids = [f.name.replace("__horizontal_well.csv", "") for f in hw_files]
print("Test wells:", well_ids)

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def build_test(hw, tw):
    n = len(hw); ps = get_ps(hw)
    gr = fill_arr(hw["GR"]); md = hw["MD"].astype(float).values
    z = hw["Z"].astype(float).values; x = hw["X"].astype(float).values; y = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().values
    tw_tvt = tw["TVT"].astype(float).values; tw_gr = fill_arr(tw["GR"])
    gr_s = pd.Series(gr)
    known = tvt_inp[:ps].values; valid = ~np.isnan(known)
    pre_z = z[:ps][valid]; pre_tvt = known[valid]
    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1,1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_r2 = float(1 - np.var(pre_tvt - reg.predict(pre_z.reshape(-1,1))) / (np.var(pre_tvt) + 1e-9))
    else:
        slope = -1.0; pre_r2 = 0.
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)]); Z_anchor = z[max(0, ps-1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)
    post_z = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post = len(post_z)
    z_end = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z = z_end - Z_anchor
    total_md = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.
    post_dz_rate = total_z / total_md; phys_tvt_end = slope * total_z
    pre_dz_md = ((z[ps-1] - z[0]) / (md[ps-1] - md[0])) if ps > 1 and abs(md[ps-1] - md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md
    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr - gr_at_physics
    feat = {"gr": gr, "gr_diff": np.gradient(gr), "gr_diff2": np.gradient(np.gradient(gr))}
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat["gr_mean_" + str(w)] = roll.mean().values
        feat["gr_std_" + str(w)] = roll.std().fillna(0).values
        feat["gr_range_" + str(w)] = (roll.max() - roll.min()).values
    feat.update({
        "md": md, "z": z, "dz_dmd": np.gradient(z, md),
        "dx_dmd": np.gradient(x, md), "dy_dmd": np.gradient(y, md),
        "inclination": np.arctan2(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2), np.abs(np.gradient(z)) + 1e-9) * 180 / np.pi,
        "raw_dz": np.diff(z, prepend=z[0]), "raw_dmd": np.diff(md, prepend=md[0]),
        "last_known_tvt": lkt,
        "rows_since_ps": np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps": np.maximum(0, ps - np.arange(n)).astype(float),
        "ps_idx": float(ps), "tvt_z_slope": np.full(n, slope),
        "physics_tvt": anchored_physics, "physics_vs_lkt": anchored_physics - lkt,
        "physics_dtvt": slope * np.diff(z, prepend=z[0]),
        "gr_at_physics": gr_at_physics, "gr_dev_physics": gr_dev,
        "total_z_change_post": np.full(n, total_z),
        "physics_tvt_at_end": np.full(n, phys_tvt_end),
        "post_dz_rate": np.full(n, post_dz_rate), "dz_rate_change": np.full(n, dz_rate_change),
        "n_post_ps": np.full(n, float(n_post)),
        "post_ps_frac": np.where(n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.).astype(float),
        "pre_r2": np.full(n, pre_r2)
    })
    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt = tvt_inp[:ps].values; vtvt = vtvt[~np.isnan(vtvt)]
        pdtvt = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"] = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.)
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)
    df = pd.DataFrame(feat)
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    df["anchored_physics_col"] = anchored_physics
    return df, ps

rows = []
for well_id in well_ids:
    hw = pd.read_csv(TEST_DIR / (well_id + "__horizontal_well.csv"))
    tw = pd.read_csv(TEST_DIR / (well_id + "__typewell.csv"))
    df, ps = build_test(hw, tw)
    df_post = df[df["is_post_ps"] == 1].copy()
    X = df_post[FCOLS].values.astype(np.float32)
    corr = model.predict(X)
    tvt_pred = gaussian_filter1d(df_post["anchored_physics_col"].values + corr, sigma=1.0)
    post_indices = df_post.index.tolist()
    for i, idx in enumerate(post_indices):
        rows.append({"id": well_id + "_" + str(idx), "tvt": round(float(tvt_pred[i]), 4)})
    print("  " + well_id + ": " + str(len(post_indices)) + " rows, TVT [" + str(round(tvt_pred[0],1)) + ", " + str(round(tvt_pred[-1],1)) + "]")

sub = pd.DataFrame(rows)
print("Total rows: " + str(len(sub)))
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Done! First rows:")
print(sub.head().to_string())
# Verify format
assert list(sub.columns) == ["id", "tvt"], "Wrong columns: " + str(sub.columns.tolist())
assert len(sub) == 14151, "Wrong row count: " + str(len(sub))
print("Format check passed!")