"""
LightGBM v20 — GR-Typewell Correlation + Physics Features
==========================================================

Architecture
------------
  Base features  (v21 physics + trajectory, ~58 cols):
    - Anchored-physics TVT, pre-PS linear slope, Z/MD trajectory stats
    - XY cumulative displacement from PS anchor  (v20 addition)
    - Running GR deviation cumsum / mean since PS (v20/v21 addition)
    - TW GR gradient → GR-based TVT correction   (v21 addition)
    - Per-row and per-well GR rolling stats, pre-PS residuals

  New GR features (v3 typewell matching, 82 cols):
    - GR calibration: gr_calib_a, gr_calib_b, gr_calib_r2
    - GR value matching: gr_val_tvt_sX_rY, gr_val_delta_sX_rY, gr_val_err_sX_rY
      (σ ∈ {0,3,7}, radius ∈ {15,30,50,80 ft})  — 36 cols
    - Global xcorr: xcorr_global_best_delta, _peak_corr, _corr_at_0, _corr_std,
                    _corr_rel, _second_delta, xcorr_global_corrected_tvt   — 7 cols
    - Local sliding-window xcorr (W ∈ {200,500,1000}):
        local_xcorr_tvt_W, _delta_W, _score_W                               — 9 cols
    - GR gradient: gr_grad_hw, tw_grad_at_lkt, tw_grad_at_val_tvt,
                   gr_grad_sign_match, gr_grad_sign_val, gr_grad_ratio_lkt,
                   tw_curv_at_lkt                                            — 7 cols
    - Formation boundary: gr_boundary_strength, rows_to_prev_bound,
                          rows_to_next_bound, tw_local_range_at_lkt         — 4 cols
    - Pre-PS xcorr: pre_ps_xcorr_best_delta, _peak_corr, _corr_at_0        — 3 cols
    - TW GR at estimates: gr_at_lkt, gr_at_xcorr_global, gr_at_local_xcorr,
                          gr_at_val_match                                    — 4 cols
    - GR deviations: gr_dev_lkt, gr_dev_xcorr_global, gr_dev_local_xcorr,
                     gr_dev_val_match                                        — 4 cols
    - Agreement metrics: val_vs_local_xcorr, val_vs_global_xcorr,
                         xcorr_vs_lkt                                        — 3 cols

Merge key: (well_id, row_idx)   [row_idx = 0..n-1 per well]

Target:  target_correction = true_tvt − anchored_physics
Predict: tvt = anchored_physics + model(correction)

Train: 658 wells / Val: 115 wells
Baseline: v8 / v21 val RMSE ≈ 11.983 ft
Goal:    < 11 ft
"""
import numpy as np
import pandas as pd
import pickle
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
PRED_DIR   = DATA_DIR / "predictions"
FEAT_DIR   = DATA_DIR / "features"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values


def load_well(wid, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (pd.read_csv(d / f"{wid}__horizontal_well.csv"),
            pd.read_csv(d / f"{wid}__typewell.csv"))


def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


# ── Base feature builder (v21 with row_idx for merge key) ────────────────────

def build(hw, tw, well_id, split="train"):
    """
    Build v21-style base feature dataframe for one well.
    Adds 'row_idx' column (0..n-1) used to merge GR xcorr features.
    """
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

    # ── Pre-PS linear fit: TVT ~ slope * Z ──────────────────────────────────
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
        pre_r2 = float(
            1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)
        ) if np.var(pre_tvt) > 0 else 1.0
        pre_resid_std_val = float(np.std(pre_tvt - pre_tvt_pred.flatten()))
    else:
        slope = -1.0
        pre_r2 = 0.0
        pre_resid_std_val = 30.0

    # ── Anchored physics TVT ─────────────────────────────────────────────────
    anchor_tvt       = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor         = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    # ── Post-PS trajectory summary ───────────────────────────────────────────
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

    # ── Typewell GR at physics TVT (per row) ────────────────────────────────
    gr_at_physics = np.interp(
        anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1]
    )
    gr_dev = gr - gr_at_physics

    # ── GR rolling statistics ────────────────────────────────────────────────
    feat = {
        "gr":       gr,
        "gr_diff":  np.gradient(gr),
        "gr_diff2": np.gradient(np.gradient(gr)),
    }
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values

    # ── Trajectory features ──────────────────────────────────────────────────
    feat.update({
        "md":          md,
        "z":           z,
        "dz_dmd":      np.gradient(z, md),
        "dx_dmd":      np.gradient(x, md),
        "dy_dmd":      np.gradient(y, md),
        "inclination": np.arctan2(
            np.sqrt(np.gradient(x) ** 2 + np.gradient(y) ** 2),
            np.abs(np.gradient(z)) + 1e-9
        ) * 180 / np.pi,
        "raw_dz":      np.diff(z,  prepend=z[0]),
        "raw_dmd":     np.diff(md, prepend=md[0]),
    })

    # ── TVT-related features ─────────────────────────────────────────────────
    feat.update({
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
    })

    # ── Post-PS summary scalars ──────────────────────────────────────────────
    feat.update({
        "total_z_change_post": np.full(n, total_z),
        "physics_tvt_at_end":  np.full(n, phys_tvt_end),
        "post_dz_rate":        np.full(n, post_dz_rate),
        "dz_rate_change":      np.full(n, dz_rate_change),
        "n_post_ps":           np.full(n, float(n_post)),
        "post_ps_frac":        np.where(
            n_post > 0,
            np.maximum(0, np.arange(n) - ps) / n_post, 0.0
        ).astype(float),
        "pre_r2":              np.full(n, pre_r2),
    })

    # ── Pre-PS slope/dTVT statistics ────────────────────────────────────────
    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.0])
        vtvt   = tvt_inp[:ps].values
        vtvt   = vtvt[~np.isnan(vtvt)]
        pdtvt  = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.0])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(
            n, pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.0
        )
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    # ── v20: XY cumulative displacement from PS anchor ───────────────────────
    x_anchor = x[max(0, ps - 1)]
    y_anchor = y[max(0, ps - 1)]
    feat["x_disp"] = x - x_anchor
    feat["y_disp"] = y - y_anchor

    # ── v20: Running GR deviation cumsum since PS ────────────────────────────
    gr_dev_cumsum = np.zeros(n)
    for i in range(ps, n):
        gr_dev_cumsum[i] = gr_dev_cumsum[i - 1] + gr_dev[i]
    feat["gr_dev_cumsum"] = gr_dev_cumsum

    # ── v20: Pre-PS residual std ─────────────────────────────────────────────
    feat["pre_resid_std"] = np.full(n, pre_resid_std_val)

    # ── v21: TW GR gradient → per-row TVT correction estimate ───────────────
    dtvt         = 5.0
    tw_gr_plus   = np.interp(anchored_physics + dtvt, tw_tvt, tw_gr,
                             left=tw_gr[0], right=tw_gr[-1])
    tw_gr_minus  = np.interp(anchored_physics - dtvt, tw_tvt, tw_gr,
                             left=tw_gr[0], right=tw_gr[-1])
    tw_gr_grad   = (tw_gr_plus - tw_gr_minus) / (2.0 * dtvt)  # API/ft
    safe_grad    = np.where(
        np.abs(tw_gr_grad) >= 0.2,
        tw_gr_grad,
        np.sign(tw_gr_grad + 1e-9) * 0.2
    )
    feat["tw_gr_gradient"]     = tw_gr_grad
    feat["tw_gr_gradient_mag"] = np.abs(tw_gr_grad)
    feat["gr_tvt_correction"]  = np.clip(gr_dev / safe_grad, -300.0, 300.0)

    # ── v21: Running mean of GR deviation since PS ───────────────────────────
    gr_dev_running_mean = np.zeros(n)
    for i in range(ps, n):
        gr_dev_running_mean[i] = gr_dev_cumsum[i] / max(1, i - ps + 1)
    feat["gr_dev_running_mean"] = gr_dev_running_mean

    # ── XY post range/rate scalars ───────────────────────────────────────────
    post_x = x[ps:] if ps < n else x[:]
    post_y = y[ps:] if ps < n else y[:]
    feat["post_x_range"] = np.full(n, float(post_x.max() - post_x.min()) if len(post_x) else 0.0)
    feat["post_y_range"] = np.full(n, float(post_y.max() - post_y.min()) if len(post_y) else 0.0)
    feat["post_x_rate"]  = np.full(n, float(post_x[-1] - post_x[0]) / total_md if total_md > 0 else 0.0)
    feat["post_y_rate"]  = np.full(n, float(post_y[-1] - post_y[0]) / total_md if total_md > 0 else 0.0)

    df = pd.DataFrame(feat)
    df["well_id"]    = well_id
    df["row_idx"]    = np.arange(n, dtype=np.int32)  # key for GR merge
    df["is_post_ps"] = (np.arange(n) >= ps).astype(np.int8)
    # Always store physics col so test prediction can use tvt = physics + correction
    df["anchored_physics_col"] = anchored_physics.astype(np.float32)

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"]    = (tvt_true - anchored_physics).astype(np.float32)
        df["target_tvt_delta"]     = (tvt_true - lkt).astype(np.float32)
        df["target_tvt"]           = tvt_true.astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load pre-computed GR xcorr features
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LGBM v20 — GR-Typewell Correlation Model")
print("=" * 60)
print("\n[1/5] Loading GR xcorr features (pkl)...")

with open(FEAT_DIR / "gr_xcorr_features_train.pkl", "rb") as f:
    gr_train = pickle.load(f)
with open(FEAT_DIR / "gr_xcorr_features_val.pkl", "rb") as f:
    gr_val = pickle.load(f)
with open(FEAT_DIR / "gr_xcorr_features_test.pkl", "rb") as f:
    gr_test = pickle.load(f)

print(f"  Train GR: {gr_train.shape}  ({gr_train['well_id'].nunique()} wells)")
print(f"  Val   GR: {gr_val.shape}  ({gr_val['well_id'].nunique()} wells)")
print(f"  Test  GR: {gr_test.shape}  ({gr_test['well_id'].nunique()} wells)")

# Columns to merge from GR pkl (drop metadata & targets that overlap with base)
GR_META_DROP = {
    "well_id", "is_post_ps", "row_idx", "ps_idx",
    "last_known_tvt", "physics_tvt",
    "target_tvt", "target_tvt_delta", "target_correction",
}
GR_FCOLS = [c for c in gr_train.columns if c not in GR_META_DROP]
print(f"  GR feature columns to merge: {len(GR_FCOLS)}")


def merge_gr_features(base_df: pd.DataFrame, gr_df: pd.DataFrame,
                      gr_cols: list) -> pd.DataFrame:
    """Left-join GR features into base_df on (well_id, row_idx)."""
    gr_sub = gr_df[["well_id", "row_idx"] + gr_cols]
    return base_df.merge(gr_sub, on=["well_id", "row_idx"], how="left")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build base features for train / val wells
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Building base features (v21 physics + trajectory)...")

train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"  Train: {len(train_ids)} wells | Val: {len(val_ids)} wells")

tr_dfs, vl_dfs = [], []

for i, wid in enumerate(train_ids):
    try:
        tr_dfs.append(build(*load_well(wid), wid, "train"))
    except Exception as e:
        print(f"  SKIP train/{wid}: {e}")
    if (i + 1) % 100 == 0:
        print(f"    train {i+1}/{len(train_ids)}", flush=True)

print(f"  Train wells built: {len(tr_dfs)}")

for wid in val_ids:
    try:
        vl_dfs.append(build(*load_well(wid), wid, "train"))
    except Exception as e:
        print(f"  SKIP val/{wid}: {e}")

print(f"  Val wells built:   {len(vl_dfs)}")

tr_all = pd.concat(tr_dfs, ignore_index=True)
vl_all = pd.concat(vl_dfs, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Merge GR xcorr features & feature selection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Merging GR xcorr features...")

tr_all = merge_gr_features(tr_all, gr_train, GR_FCOLS)
vl_all = merge_gr_features(vl_all, gr_val,   GR_FCOLS)
print(f"  Train merged: {tr_all.shape}")
print(f"  Val   merged: {vl_all.shape}")

# Filter to post-PS rows for training
tr_p = tr_all[tr_all["is_post_ps"] == 1].copy()
vl_p = vl_all[vl_all["is_post_ps"] == 1].copy()
print(f"  Post-PS rows — Train: {len(tr_p):,}  Val: {len(vl_p):,}")

# NaN audit on GR features after merge
nan_rates_gr = tr_p[GR_FCOLS].isnull().mean()
high_nan_gr  = nan_rates_gr[nan_rates_gr > 0.10]
if len(high_nan_gr):
    print(f"  WARNING: dropping {len(high_nan_gr)} GR features with >10% NaN:")
    for c in high_nan_gr.index:
        print(f"    {c}: {high_nan_gr[c]:.1%}")
    GR_FCOLS = [c for c in GR_FCOLS if c not in high_nan_gr.index]
else:
    print(f"  NaN audit passed — all {len(GR_FCOLS)} GR features clean")

# Fill any remaining NaN with 0 (edge cases in short wells)
tr_p[GR_FCOLS] = tr_p[GR_FCOLS].fillna(0.0)
vl_p[GR_FCOLS] = vl_p[GR_FCOLS].fillna(0.0)

# ── Derived "GR correction to physics" features ──────────────────────────────
# These express each GR-based TVT estimate as a correction over physics_tvt
# (same units as target_correction), making them directly learnable in the
# first few boosting rounds without needing a multi-step tree interaction.
DERIVED_GR_FEATS = []
PHYS = "physics_tvt"

# Best GR-value match variants: gr_val_tvt_sX_rY − physics_tvt
for sigma in [0, 3, 7]:
    for radius in [15, 30, 50, 80]:
        src = f"gr_val_tvt_s{sigma}_r{radius}"
        if src in tr_p.columns:
            fname = f"gr_val_corr_s{sigma}_r{radius}"
            tr_p[fname] = tr_p[src] - tr_p[PHYS]
            vl_p[fname] = vl_p[src] - vl_p[PHYS]
            DERIVED_GR_FEATS.append(fname)

# Local xcorr variants: local_xcorr_tvt_W − physics_tvt
for w in [200, 500, 1000]:
    src = f"local_xcorr_tvt_{w}"
    if src in tr_p.columns:
        fname = f"local_xcorr_corr_{w}"
        tr_p[fname] = tr_p[src] - tr_p[PHYS]
        vl_p[fname] = vl_p[src] - vl_p[PHYS]
        DERIVED_GR_FEATS.append(fname)

# Global xcorr corrected TVT − physics_tvt  (= xcorr_global_best_delta, but cleaner)
if "xcorr_global_corrected_tvt" in tr_p.columns:
    fname = "xcorr_global_corr_to_phys"
    tr_p[fname] = tr_p["xcorr_global_corrected_tvt"] - tr_p[PHYS]
    vl_p[fname] = vl_p["xcorr_global_corrected_tvt"] - vl_p[PHYS]
    DERIVED_GR_FEATS.append(fname)

print(f"  Added {len(DERIVED_GR_FEATS)} derived GR-correction-to-physics features")

# Build final feature column list
EXCL_COLS = {
    "well_id", "row_idx", "is_post_ps",
    "target_correction", "target_tvt_delta", "target_tvt",
    "anchored_physics_col",
}
FCOLS = [c for c in tr_p.columns if c not in EXCL_COLS]
base_cnt = len(FCOLS) - len(GR_FCOLS) - len(DERIVED_GR_FEATS)
print(f"\n  Total features: {len(FCOLS)}")
print(f"    Base (v21) features             : {base_cnt}")
print(f"    GR xcorr  features (pkl)        : {len(GR_FCOLS)}")
print(f"    Derived GR-corr-to-phys features: {len(DERIVED_GR_FEATS)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train LightGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Training LightGBM v20...")

# ── TARGET CHOICE: tvt − physics (correction target) ─────────────────────────
# Rationale: physics_tvt is already a calibrated trajectory estimate so
# target_correction = tvt − physics_tvt has much lower variance (~5-8 ft std)
# than target_tvt_delta = tvt − lkt (~15 ft std).  Lower-variance target means
# LightGBM RMSE decreases faster and more stably, enabling early stopping to
# fire at a genuinely good stopping point (not too early).
#
# Run #1 failure: target=correction + patience=150 → best at round 111 → GR features
# never discovered.  Fix: patience=400, which gives 400 more rounds AFTER physics
# plateau to let GR features exploit their residual signal.
TARGET_COL = "target_correction"

X_tr = tr_p[FCOLS].values.astype(np.float32)
y_tr = tr_p[TARGET_COL].values.astype(np.float32)
X_vl = vl_p[FCOLS].values.astype(np.float32)
y_vl = vl_p[TARGET_COL].values.astype(np.float32)

print(f"  X_tr: {X_tr.shape}  X_vl: {X_vl.shape}")
print(f"  Target: '{TARGET_COL}'  "
      f"train mean={y_tr.mean():.2f}  std={y_tr.std():.2f}")

params = {
    # ── Core objective ──────────────────────────────────────────────────────
    "objective":         "regression",
    "metric":            "rmse",
    # ── Tree structure ──────────────────────────────────────────────────────
    # num_leaves=127: fast enough (~30 min for 3.2M rows, 155 features).
    # Higher than needed for pure physics (63 leaves) to capture GR interactions.
    "num_leaves":        127,
    # ── Learning rate ───────────────────────────────────────────────────────
    # lr=0.05 with patience=400: ~1500 effective rounds.
    # At lr=0.05 the physics signal is captured in ~150 rounds;
    # patience=400 gives 400 additional rounds for GR features to contribute
    # to the residual.  Much faster than lr=0.02 (which would need 5000+ rounds).
    "learning_rate":     0.05,
    # ── Regularisation ─────────────────────────────────────────────────────
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,    # small leaves → picks up signal from hard wells
    "lambda_l1":         0.05,
    "lambda_l2":         0.05,
    # ── Misc ────────────────────────────────────────────────────────────────
    "verbose":           -1,
    "seed":              42,
    "n_jobs":            -1,
}

lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=FCOLS)
lgb_vl = lgb.Dataset(X_vl, label=y_vl, feature_name=FCOLS, reference=lgb_tr)

model = lgb.train(
    params, lgb_tr,
    num_boost_round=10_000,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(600, verbose=False),
        lgb.log_evaluation(1000),
    ],
)

best_iter = model.best_iteration
best_rmse = model.best_score["valid_0"]["rmse"]
print(f"\n  Best iteration: {best_iter}  |  Val RMSE on '{TARGET_COL}': {best_rmse:.4f} ft")

# Save model
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "lgbm_v20.pkl")
print(f"  Saved: models/lgbm_v20.pkl")

# Feature importance
fi = pd.DataFrame({
    "feature":    FCOLS,
    "importance": model.feature_importance("gain"),
})
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
fi.to_csv(MODELS_DIR / "lgbm_v20_importance.csv", index=False)

print("\n  Top 20 features by importance (gain):")
print(f"  {'rank':<5} {'feature':<40} {'importance':>14}")
print("  " + "-" * 62)
for rank, (_, row) in enumerate(fi.head(20).iterrows(), 1):
    is_new = "★" if any(
        row["feature"].startswith(p) for p in [
            "gr_val_", "gr_calib_", "local_xcorr_", "xcorr_global_",
            "gr_grad_", "gr_boundary_", "gr_dev_lkt", "gr_dev_xcorr",
            "gr_dev_local", "gr_dev_val", "val_vs_", "xcorr_vs_",
            "gr_at_lkt", "gr_at_xcorr", "gr_at_local", "gr_at_val",
            "pre_ps_xcorr", "tw_local_range",
        ]
    ) else " "
    print(f"  {rank:<5} {is_new}{row['feature']:<39} {row['importance']:>14,.0f}")

# All GR-derived feature names (pkl + computed) for importance summary
ALL_GR_NAMES = set(GR_FCOLS) | set(DERIVED_GR_FEATS)
gr_fi    = fi[fi["feature"].isin(ALL_GR_NAMES)]
base_fi  = fi[~fi["feature"].isin(ALL_GR_NAMES)]
gr_total   = gr_fi["importance"].sum()
base_total = base_fi["importance"].sum()
total      = gr_total + base_total
print(f"\n  GR (xcorr + derived) features total importance: {gr_total/total:.1%}  "
      f"(base {base_total/total:.1%})")
print(f"  Top 5 GR features:")
for _, row in gr_fi.head(5).iterrows():
    print(f"    {row['feature']:<45} {row['importance']:>14,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper — add derived GR features to a per-well post-PS dataframe
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_gr(df_p: pd.DataFrame) -> pd.DataFrame:
    """Add derived GR-correction-to-physics features to a post-PS dataframe."""
    df_p = df_p.copy()
    PHYS = "physics_tvt"
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Per-well validation RMSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Computing per-well validation RMSE...")

rmses = []
well_rmse_list = []

for wid in val_ids:
    try:
        hw, tw  = load_well(wid)
        ps      = get_ps(hw)
        df_base = build(hw, tw, wid, "train")
        df_base = merge_gr_features(df_base, gr_val, GR_FCOLS)
        df_p    = df_base[df_base["is_post_ps"] == 1].copy()
        df_p[GR_FCOLS] = df_p[GR_FCOLS].fillna(0.0)
        df_p    = add_derived_gr(df_p)

        corr     = model.predict(df_p[FCOLS].values.astype(np.float32))
        # Predict: tvt = physics_tvt + correction
        tvt_pred = gaussian_filter1d(
            df_p["anchored_physics_col"].values + corr, sigma=1.0
        )
        tvt_true = hw["TVT"].astype(float).values[ps:]
        n_min    = min(len(tvt_pred), len(tvt_true))
        r = float(np.sqrt(np.mean((tvt_pred[:n_min] - tvt_true[:n_min]) ** 2)))
        rmses.append(r)
        well_rmse_list.append((wid, r))
    except Exception as e:
        print(f"  SKIP val/{wid}: {e}")

rmses = np.array(rmses)
overall_rmse = rmses.mean()

print(f"\n{'='*55}")
print(f"  Overall Val RMSE  : {overall_rmse:.4f} ft")
print(f"  Median  Val RMSE  : {np.median(rmses):.4f} ft")
print(f"  Std     Val RMSE  : {rmses.std():.4f} ft")
print(f"  Min/Max Val RMSE  : {rmses.min():.3f} / {rmses.max():.3f} ft")
print(f"{'='*55}")
print(f"  v8 baseline       : 11.983 ft")
delta = overall_rmse - 11.983
print(f"  Delta vs baseline : {delta:+.3f} ft  "
      f"({'BETTER ✓' if delta < 0 else 'WORSE ✗'})")
print(f"{'='*55}")

# Per-well breakdown (sorted worst → best)
well_rmse_list.sort(key=lambda t: -t[1])
print(f"\nPer-well Val RMSE (sorted worst → best):")
print(f"  {'#':<4} {'well_id':<12} {'RMSE (ft)':>10}")
print("  " + "-" * 30)
for rank, (wid, r) in enumerate(well_rmse_list, 1):
    flag = " ◄" if r > 30 else ""
    print(f"  {rank:<4} {wid:<12} {r:>10.3f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate test predictions + submission
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating test predictions...")

sub        = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_preds = []

for wid in TEST_WELLS:
    hw, tw  = load_well(wid, "test")
    ps      = get_ps(hw)
    df_base = build(hw, tw, wid, "test")

    # Merge GR test features for this specific well
    gr_w    = gr_test[gr_test["well_id"] == wid].copy()
    df_base = merge_gr_features(df_base, gr_w, GR_FCOLS)
    df_p    = df_base[df_base["is_post_ps"] == 1].copy()
    df_p[GR_FCOLS] = df_p[GR_FCOLS].fillna(0.0)
    df_p    = add_derived_gr(df_p)

    corr     = model.predict(df_p[FCOLS].values.astype(np.float32))
    # Predict: tvt = physics_tvt + correction
    tvt_pred = gaussian_filter1d(df_p["anchored_physics_col"].values + corr, sigma=1.0)

    # Map to submission IDs  (empty TVT_input rows = prediction targets)
    empty = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    rows  = hw.index[empty].tolist()

    id_map = {f"{wid}_{i}": float(v) for i, v in zip(rows, tvt_pred)}
    for id_str, tvt_val in id_map.items():
        test_preds.append({"id": id_str, "tvt": tvt_val})

    mask = sub["id"].isin(set(id_map.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)

    print(f"  [{wid}] PS={ps}  post-PS rows={len(rows)}"
          f"  TVT=[{tvt_pred.min():.1f}, {tvt_pred.max():.1f}]"
          f"  range={tvt_pred.max()-tvt_pred.min():.1f} ft")

# Save submission
SUBS_DIR.mkdir(exist_ok=True)
sub.to_csv(SUBS_DIR / "lgbm_v20.csv", index=False)
nan_count = sub["tvt"].isna().sum()
print(f"\n  Saved: submissions/lgbm_v20.csv  "
      f"(rows={len(sub)}, NaN={nan_count})")

# Save raw test predictions
PRED_DIR.mkdir(exist_ok=True)
pred_df = pd.DataFrame(test_preds)
pred_df.to_csv(PRED_DIR / "lgbm_v20_test.csv", index=False)
print(f"  Saved: predictions/lgbm_v20_test.csv  (rows={len(pred_df)})")

print("\n" + "=" * 60)
print("lgbm_v20 training complete.")
print(f"  Val RMSE: {overall_rmse:.4f} ft  (best iter: {best_iter})")
print(f"  Submission: submissions/lgbm_v20.csv")
print("=" * 60)
