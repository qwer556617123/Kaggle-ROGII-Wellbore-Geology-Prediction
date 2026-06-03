"""
ROGII ranked LGBM candidate notebook.

This notebook script is designed for competition notebook submission. Kaggle
will rerun it with the hidden test set substituted into the competition input.

Default candidate: v13-like LGBM with anchor blend beta=0.75.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression


warnings.filterwarnings("ignore")
np.random.seed(42)

BETA = float(os.getenv("ROGII_BLEND_BETA", "0.75"))
NUM_ROUNDS = int(os.getenv("ROGII_NUM_ROUNDS", "1254"))
SMOOTH_SIGMA = float(os.getenv("ROGII_SMOOTH_SIGMA", "1.0"))

COMP_CANDIDATES = [
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("."),
]
COMP_DIR = next(path for path in COMP_CANDIDATES if (path / "train").exists() and (path / "test").exists())
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = COMP_DIR / "test"
SAMPLE_PATH = COMP_DIR / "sample_submission.csv"
OUT_PATH = Path("/kaggle/working/submission.csv")
if not OUT_PATH.parent.exists():
    OUT_PATH = Path("submission.csv")

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 255,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.3,
    "lambda_l2": 0.3,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

EXCLUDE = {"well_id", "target_correction", "target_tvt", "is_post_ps", "anchored_physics_col"}


def fill_arr(values) -> np.ndarray:
    return pd.Series(values).ffill().bfill().astype(float).to_numpy()


def get_ps(hw: pd.DataFrame) -> int:
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def load_well(well_id: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = TRAIN_DIR if split == "train" else TEST_DIR
    return (
        pd.read_csv(root / f"{well_id}__horizontal_well.csv"),
        pd.read_csv(root / f"{well_id}__typewell.csv"),
    )


def build_features(hw: pd.DataFrame, tw: pd.DataFrame, well_id: str, split: str) -> pd.DataFrame:
    n = len(hw)
    ps = get_ps(hw)
    gr = fill_arr(hw["GR"])
    md = hw["MD"].astype(float).to_numpy()
    z = hw["Z"].astype(float).to_numpy()
    x = hw["X"].astype(float).to_numpy()
    y = hw["Y"].astype(float).to_numpy()
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().to_numpy()
    tw_tvt = tw["TVT"].astype(float).to_numpy()
    tw_gr = fill_arr(tw["GR"])
    gr_s = pd.Series(gr)

    if split == "train" and "TVT" in hw.columns:
        pre_z = z[:ps]
        pre_tvt = hw["TVT"].astype(float).to_numpy()[:ps]
    else:
        known = tvt_inp.iloc[:ps].to_numpy()
        valid = ~np.isnan(known)
        pre_z = z[:ps][valid]
        pre_tvt = known[valid]

    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
        pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.0
    else:
        slope = -1.0
        pre_r2 = 0.0

    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    z_anchor = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - z_anchor)

    post_z = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post = len(post_z)
    z_end = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z = z_end - z_anchor
    total_md = float(post_md[-1] - post_md[0]) if n_post > 1 else 1.0
    post_dz_rate = total_z / total_md
    phys_tvt_end = slope * total_z
    pre_dz_md = (z[ps - 1] - z[0]) / (md[ps - 1] - md[0]) if ps > 1 and abs(md[ps - 1] - md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md

    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr - gr_at_physics

    feat = {
        "gr": gr,
        "gr_diff": np.gradient(gr),
        "gr_diff2": np.gradient(np.gradient(gr)),
    }
    for window in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(window, center=True, min_periods=1)
        feat[f"gr_mean_{window}"] = roll.mean().to_numpy()
        feat[f"gr_std_{window}"] = roll.std().fillna(0).to_numpy()
        feat[f"gr_range_{window}"] = (roll.max() - roll.min()).to_numpy()

    feat.update(
        {
            "md": md,
            "z": z,
            "dz_dmd": np.gradient(z, md),
            "dx_dmd": np.gradient(x, md),
            "dy_dmd": np.gradient(y, md),
            "inclination": np.arctan2(
                np.sqrt(np.gradient(x) ** 2 + np.gradient(y) ** 2),
                np.abs(np.gradient(z)) + 1e-9,
            )
            * 180
            / np.pi,
            "raw_dz": np.diff(z, prepend=z[0]),
            "raw_dmd": np.diff(md, prepend=md[0]),
            "last_known_tvt": lkt,
            "rows_since_ps": np.maximum(0, np.arange(n) - ps).astype(float),
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
            "post_ps_frac": np.where(n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.0).astype(float),
            "pre_r2": np.full(n, pre_r2),
        }
    )

    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.0])
        vtvt = tvt_inp.iloc[:ps].to_numpy()
        vtvt = vtvt[~np.isnan(vtvt)]
        pdtvt = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.0])
        feat["pre_ps_dz_slope"] = np.full(n, pre_dz[-min(50, len(pre_dz)) :].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)) :].mean())
        feat["pre_ps_dtvt_std"] = np.full(n, pdtvt[-min(50, len(pdtvt)) :].std() if len(pdtvt) > 1 else 0.0)
    else:
        for key in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[key] = np.zeros(n)

    df = pd.DataFrame(feat)
    df["well_id"] = well_id
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    df["anchored_physics_col"] = anchored_physics
    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).to_numpy()
        df["target_correction"] = tvt_true - anchored_physics
        df["target_tvt"] = tvt_true
    return df


def smooth_by_well(frame: pd.DataFrame, values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values
    out = np.empty_like(values, dtype=float)
    tmp = pd.DataFrame({"well_id": frame["well_id"].to_numpy(), "pred": values})
    for _well_id, idx in tmp.groupby("well_id", sort=False).indices.items():
        out[idx] = gaussian_filter1d(values[idx], sigma=sigma)
    return out


def submission_ids(test: pd.DataFrame) -> pd.Series:
    rows = test["ps_idx"].astype(int) + test["rows_since_ps"].round().astype(int)
    return test["well_id"].astype(str) + "_" + rows.astype(str)


print(f"Competition directory: {COMP_DIR}")
print(f"Candidate beta={BETA}, rounds={NUM_ROUNDS}, smooth_sigma={SMOOTH_SIGMA}")

train_ids = sorted(p.name.replace("__horizontal_well.csv", "") for p in TRAIN_DIR.glob("*__horizontal_well.csv"))
train_frames = []
for i, well_id in enumerate(train_ids, start=1):
    try:
        train_frames.append(build_features(*load_well(well_id, "train"), well_id, "train"))
    except Exception as exc:
        print(f"SKIP train/{well_id}: {exc}")
    if i % 150 == 0:
        print(f"built train wells {i}/{len(train_ids)}")

train_all = pd.concat(train_frames, ignore_index=True)
train_post = train_all[train_all["is_post_ps"] == 1].copy()
feature_cols = [col for col in train_post.columns if col not in EXCLUDE and pd.api.types.is_numeric_dtype(train_post[col])]

print(f"Training rows={len(train_post):,}, wells={train_post['well_id'].nunique()}, features={len(feature_cols)}")
dataset = lgb.Dataset(
    train_post[feature_cols].to_numpy(dtype=np.float32),
    label=train_post["target_correction"].to_numpy(dtype=np.float32),
    feature_name=feature_cols,
)
model = lgb.train(
    LGBM_PARAMS,
    dataset,
    num_boost_round=NUM_ROUNDS,
    callbacks=[lgb.log_evaluation(250)],
)

test_ids = sorted(p.name.replace("__horizontal_well.csv", "") for p in TEST_DIR.glob("*__horizontal_well.csv"))
test_frames = []
for well_id in test_ids:
    test_frames.append(build_features(*load_well(well_id, "test"), well_id, "test"))
test_all = pd.concat(test_frames, ignore_index=True)
test_post = test_all[test_all["is_post_ps"] == 1].copy()

corr = model.predict(test_post[feature_cols].to_numpy(dtype=np.float32), num_iteration=NUM_ROUNDS)
lgbm_tvt = test_post["anchored_physics_col"].to_numpy(dtype=float) + corr
lgbm_tvt = smooth_by_well(test_post, lgbm_tvt, SMOOTH_SIGMA)
anchor = test_post["last_known_tvt"].to_numpy(dtype=float)
pred = anchor + BETA * (lgbm_tvt - anchor)

pred_map = pd.Series(np.round(pred, 6), index=submission_ids(test_post)).to_dict()
sample = pd.read_csv(SAMPLE_PATH)
missing = set(sample["id"]) - set(pred_map)
extra = set(pred_map) - set(sample["id"])
if missing or extra:
    raise RuntimeError(f"Submission id mismatch: missing={len(missing)} extra={len(extra)}")

sample["tvt"] = sample["id"].map(pred_map)
if sample["tvt"].isna().any():
    raise RuntimeError("NaN predictions in submission.")

sample.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH} rows={len(sample)}")
print(sample.groupby(sample["id"].str[:8])["tvt"].agg(["count", "min", "max", "mean"]).to_string())
