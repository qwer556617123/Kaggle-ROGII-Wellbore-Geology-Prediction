"""
LightGBM V2 — ROGII Wellbore Geology Prediction
================================================
Key improvements over v1:
1. Target: one-step dTVT = TVT[i] - TVT[i-1]  (small, stable deltas)
2. Feature: gr_match_tvt — GR cross-correlation match to typewell TVT
3. Feature: tw_gr_at_true_tvt — typewell GR at TRUE current TVT (train)
4. Inference: autoregressive with TVT feedback per step
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
MODELS_DIR.mkdir(exist_ok=True); SUBS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "lgbm_v2"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── Data loaders ───────────────────────────────────────────────────────────
def load_well(well_id, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(d / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{well_id}__typewell.csv")
    return hw, tw

def get_ps_index(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def fill_arr(arr):
    s = pd.Series(arr)
    return s.ffill().bfill().astype(float).values

# ── GR cross-correlation TVT matching ─────────────────────────────────────
def compute_gr_match_tvt(hw_gr, tw_gr, tw_tvt, last_known_tvt,
                          template_half=20, search_half=150):
    """
    For each row i, find the typewell TVT with highest cross-correlation
    to the local horizontal GR window hw_gr[i-W:i+W+1].
    Uses last_known_tvt as search center.
    Returns: gr_match_tvt array (length n)
    """
    n = len(hw_gr)
    tw_n = len(tw_gr)
    match_tvt = np.full(n, np.nan)
    W = template_half
    S = search_half

    hw_gr = fill_arr(hw_gr)
    tw_gr_f = fill_arr(tw_gr)
    # Smooth for more robust correlation
    hw_smooth = gaussian_filter1d(hw_gr, sigma=2.0)
    tw_smooth = gaussian_filter1d(tw_gr_f, sigma=1.0)

    for i in range(n):
        # Template: local GR window
        t_lo, t_hi = max(0, i - W), min(n, i + W + 1)
        template = hw_smooth[t_lo:t_hi].copy()
        template -= template.mean()
        if template.std() < 1e-3:
            match_tvt[i] = last_known_tvt[i]
            continue

        # Typewell search window centered on last known TVT
        tw_center = np.searchsorted(tw_tvt, last_known_tvt[i])
        tw_lo = max(0, tw_center - S)
        tw_hi = min(tw_n, tw_center + S + len(template))
        tw_seg = tw_smooth[tw_lo:tw_hi]

        if len(tw_seg) < len(template):
            match_tvt[i] = last_known_tvt[i]
            continue

        # Normalised cross-correlation
        corr = correlate(tw_seg, template, mode='valid')
        if len(corr) == 0:
            match_tvt[i] = last_known_tvt[i]
            continue

        best_j = int(np.argmax(corr))
        # Map back to typewell index
        best_tw_idx = tw_lo + best_j + W
        best_tw_idx = np.clip(best_tw_idx, 0, tw_n - 1)
        match_tvt[i] = float(tw_tvt[best_tw_idx])

    # Fill any remaining NaN
    nans = np.isnan(match_tvt)
    if nans.any():
        x = np.arange(n)
        match_tvt[nans] = np.interp(x[nans], x[~nans], match_tvt[~nans])

    return match_tvt

# ── Feature builder (per well) ─────────────────────────────────────────────
def build_features_v2(well_id, split="train"):
    hw, tw = load_well(well_id, split)
    ps_idx = get_ps_index(hw)
    n = len(hw)

    gr_raw = fill_arr(hw["GR"])
    gr_s   = pd.Series(gr_raw)
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    tw_gr_arr = pd.Series(tw_gr).ffill().bfill().values

    md  = hw["MD"].astype(float).values
    z   = hw["Z"].astype(float).values
    x   = hw["X"].astype(float).values
    y   = hw["Y"].astype(float).values

    # Last known TVT (forward-filled from TVT_input)
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = tvt_inp.ffill().values
    # Back-fill first values if PS is at row 0 (edge case)
    if np.isnan(last_known_tvt[0]):
        last_known_tvt = pd.Series(last_known_tvt).bfill().values

    feat = {"well_id": [well_id] * n}

    # GR features
    feat["gr"] = gr_raw
    feat["gr_diff"]  = np.gradient(gr_raw)
    feat["gr_diff2"] = np.gradient(np.gradient(gr_raw))
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values

    # Depth / trajectory
    feat["md"]          = md
    feat["z"]           = z
    feat["dz_dmd"]      = np.gradient(z, md)
    feat["dx_dmd"]      = np.gradient(x, md)
    feat["dy_dmd"]      = np.gradient(y, md)
    feat["inclination"] = np.arctan2(
        np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
        np.abs(np.gradient(z)) + 1e-9
    ) * 180 / np.pi

    feat["last_known_tvt"] = last_known_tvt
    feat["rows_since_ps"]  = np.maximum(0, np.arange(n) - ps_idx).astype(float)
    feat["rows_before_ps"] = np.maximum(0, ps_idx - np.arange(n)).astype(float)
    feat["ps_idx"]         = float(ps_idx)

    # TVT trend (from pre-PS data)
    tvt_trend = np.gradient(last_known_tvt, md)
    feat["tvt_trend"] = tvt_trend

    # Typewell GR at last_known_tvt (static anchor feature)
    tw_idx_anchor = np.searchsorted(tw_tvt, last_known_tvt).clip(0, len(tw_tvt)-1)
    feat["tw_gr_at_anchor"]   = tw_gr_arr[tw_idx_anchor]
    feat["tw_tvt_at_anchor"]  = tw_tvt[tw_idx_anchor]
    feat["gr_anchor_diff"]    = gr_raw - tw_gr_arr[tw_idx_anchor]

    # ── GR cross-correlation TVT match (KEY FEATURE) ──
    # Stride=5 for speed, interpolate between
    gr_match = compute_gr_match_tvt(
        gr_raw, tw_gr_arr, tw_tvt, last_known_tvt,
        template_half=20, search_half=150
    )
    feat["gr_match_tvt"] = gr_match

    # Typewell GR at GR-matched TVT
    tw_match_idx = np.searchsorted(tw_tvt, gr_match).clip(0, len(tw_tvt)-1)
    feat["tw_gr_at_match"]  = tw_gr_arr[tw_match_idx]
    feat["gr_match_diff"]   = gr_raw - tw_gr_arr[tw_match_idx]
    feat["match_vs_anchor"] = gr_match - last_known_tvt  # How far GR match is from anchor

    # Typewell GR gradient at matched position
    tw_gr_grad = np.gradient(tw_gr_arr, tw_tvt)
    feat["tw_gr_grad_at_match"] = tw_gr_grad[tw_match_idx]

    df = pd.DataFrame(feat)

    # ── Targets (train only) ──
    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_tvt"]      = tvt_true
        # One-step delta: TVT[i] - TVT[i-1]
        df["target_dtvt_1"]   = np.concatenate([[0], np.diff(tvt_true)])
        # Delta from last known TVT (for context)
        df["target_dtvt_anc"] = tvt_true - last_known_tvt
        df["is_post_ps"]      = (np.arange(n) >= ps_idx).astype(int)

        # tw_gr at TRUE TVT (critical for training! not possible at test time)
        tw_idx_true = np.searchsorted(tw_tvt, tvt_true).clip(0, len(tw_tvt)-1)
        df["tw_gr_at_true_tvt"] = tw_gr_arr[tw_idx_true]
        df["tw_tvt_true"]       = tw_tvt[tw_idx_true]
        df["gr_true_diff"]      = gr_raw - tw_gr_arr[tw_idx_true]

    return df

# ── Load well ID lists ──────────────────────────────────────────────────────
print("=" * 60)
print("LightGBM V2 — ROGII Wellbore Geology Prediction")
print("=" * 60)

train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"Train wells: {len(train_ids)}, Val wells: {len(val_ids)}")

# ── Build features for train wells ─────────────────────────────────────────
print("\nBuilding train features (with GR-match)...")
train_dfs = []
for i, wid in enumerate(train_ids):
    try:
        df = build_features_v2(wid, "train")
        train_dfs.append(df)
        if (i+1) % 100 == 0 or i+1 == len(train_ids):
            print(f"  {i+1}/{len(train_ids)}")
    except Exception as e:
        print(f"  SKIP {wid}: {e}")
train_all = pd.concat(train_dfs, ignore_index=True)
print(f"Train shape: {train_all.shape}")

print("\nBuilding val features...")
val_dfs = []
for i, wid in enumerate(val_ids):
    try:
        val_dfs.append(build_features_v2(wid, "train"))
        if (i+1) % 50 == 0 or i+1 == len(val_ids):
            print(f"  {i+1}/{len(val_ids)}")
    except Exception as e:
        print(f"  SKIP {wid}: {e}")
val_all = pd.concat(val_dfs, ignore_index=True)
print(f"Val shape: {val_all.shape}")

# ── Prepare training matrices ───────────────────────────────────────────────
EXCLUDE = {"well_id", "target_tvt", "target_dtvt_1", "target_dtvt_anc",
           "is_post_ps", "tw_gr_at_true_tvt", "tw_tvt_true", "gr_true_diff"}

# Use tw_gr_at_true_tvt as a training feature (not available at test time,
# but gr_match_tvt provides a similar proxy at test time)
TRAIN_EXTRA = {"tw_gr_at_true_tvt", "gr_true_diff"}
FEATURE_COLS_TRAIN = [c for c in train_all.columns
                      if c not in (EXCLUDE - TRAIN_EXTRA)]
FEATURE_COLS_TEST  = [c for c in train_all.columns
                      if c not in EXCLUDE]  # exclude true TVT features

# Train only on post-PS rows; target = one-step dTVT
train_post = train_all[train_all["is_post_ps"] == 1].copy()
val_post   = val_all[val_all["is_post_ps"] == 1].copy()
print(f"\nPost-PS: train={len(train_post)}, val={len(val_post)}")
print(f"Train features: {len(FEATURE_COLS_TRAIN)}, Test features: {len(FEATURE_COLS_TEST)}")

X_tr = train_post[FEATURE_COLS_TRAIN].values.astype(np.float32)
y_tr = train_post["target_dtvt_1"].values.astype(np.float32)
X_vl = val_post[FEATURE_COLS_TRAIN].values.astype(np.float32)
y_vl = val_post["target_dtvt_1"].values.astype(np.float32)

# ── Train LightGBM ─────────────────────────────────────────────────────────
params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples":20,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "seed":             42,
    "n_jobs":           -1,
}
lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS_TRAIN)
lgb_vl = lgb.Dataset(X_vl, label=y_vl, reference=lgb_tr)

print("\nTraining LightGBM v2...")
model = lgb.train(
    params, lgb_tr, num_boost_round=3000,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(150, verbose=True),
        lgb.log_evaluation(100),
    ],
)

joblib.dump(model, MODELS_DIR / f"{MODEL_NAME}.pkl")
print(f"Model saved. Best iteration: {model.best_iteration}")

# Feature importance
imp = pd.DataFrame({
    "feature": FEATURE_COLS_TRAIN,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
imp.to_csv(MODELS_DIR / f"{MODEL_NAME}_importance.csv", index=False)
print("Top-15 features:")
print(imp.head(15).to_string(index=False))

# ── Validation: integrate one-step deltas per well ─────────────────────────
print("\n" + "=" * 55)
print("Validation: per-well RMSE (integrating one-step predictions)")
print("=" * 55)

val_all["pred_dtvt_1"] = model.predict(
    val_all[FEATURE_COLS_TRAIN].values.astype(np.float32)
)

well_rmse = []
for wid, grp in val_all.groupby("well_id"):
    grp = grp.sort_index()
    ps = int(grp["ps_idx"].iloc[0])
    post = grp[grp["is_post_ps"] == 1]
    if len(post) == 0: continue

    # Integrate: start from last known TVT, accumulate predicted dTVT
    start_tvt = float(post["last_known_tvt"].iloc[0])
    pred_dtvts = post["pred_dtvt_1"].values
    pred_tvt = start_tvt + np.cumsum(pred_dtvts)

    truth = post["target_tvt"].values
    rmse = np.sqrt(np.mean((truth - pred_tvt) ** 2))
    well_rmse.append((wid, rmse, len(post)))

well_rmse.sort(key=lambda x: -x[1])
for wid, rmse, n in well_rmse:
    print(f"  {wid}  RMSE={rmse:7.3f}  n={n}")

overall_rmse = np.sqrt(np.mean([
    (t - p) ** 2
    for _, grp in val_all.groupby("well_id")
    for t, p in zip(
        grp[grp["is_post_ps"]==1]["target_tvt"].values,
        (float(grp[grp["is_post_ps"]==1]["last_known_tvt"].iloc[0]) +
         np.cumsum(model.predict(grp[grp["is_post_ps"]==1][FEATURE_COLS_TRAIN].values.astype(np.float32))))
        if (grp["is_post_ps"]==1).any() else []
    )
]))
print(f"\n  *** Overall Val RMSE: {overall_rmse:.4f} ft ***")

# ── Build test features & autoregressive inference ─────────────────────────
print("\n" + "=" * 55)
print("Building test features & autoregressive inference...")
print("=" * 55)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

for well_id in TEST_WELLS:
    print(f"\n[{well_id}]")
    hw, tw = load_well(well_id, "test")
    ps_idx = get_ps_index(hw)

    # Build test features (without true TVT features)
    df = build_features_v2(well_id, "test")
    df_post = df[df["rows_before_ps"] == 0].copy().reset_index(drop=True)
    print(f"  Post-PS rows: {len(df_post)}")

    # Get starting TVT
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    start_tvt = float(tvt_inp.ffill().iloc[ps_idx - 1])
    print(f"  Start TVT: {start_tvt:.2f} ft")

    # Autoregressive inference
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    tw_gr_arr = pd.Series(tw_gr).ffill().bfill().values

    pred_tvts = []
    current_tvt = start_tvt
    n_post = len(df_post)

    # Batch-predict all rows at once (approximate, not truly autoregressive)
    # For true autoregressive, we would need to update tw_gr_at_true_tvt per step
    # Here we use gr_match_tvt as the substitute (already computed)
    # Add tw_gr_at_true_tvt as gr_match feature (best we can do without true TVT)
    df_post_feat = df_post.copy()
    # Fill tw_gr_at_true_tvt with tw_gr_at_match (proxy)
    df_post_feat["tw_gr_at_true_tvt"] = df_post_feat["tw_gr_at_match"]
    df_post_feat["gr_true_diff"]      = df_post_feat["gr_match_diff"]

    X_post = df_post_feat[FEATURE_COLS_TRAIN].values.astype(np.float32)
    pred_dtvts = model.predict(X_post)

    # Integrate
    pred_tvt_arr = start_tvt + np.cumsum(pred_dtvts)
    pred_tvt_arr = gaussian_filter1d(pred_tvt_arr.astype(float), sigma=1.5)

    print(f"  Predicted TVT: [{pred_tvt_arr.min():.1f}, {pred_tvt_arr.max():.1f}]")

    # Build submission rows
    empty_mask   = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    row_indices  = hw.index[empty_mask].tolist()
    assert len(row_indices) == len(pred_tvt_arr), \
        f"Length mismatch: {len(row_indices)} vs {len(pred_tvt_arr)}"

    id_to_tvt = {f"{well_id}_{i}": v for i, v in zip(row_indices, pred_tvt_arr)}
    mask = sub["id"].isin(set(id_to_tvt.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_to_tvt)

print(f"\nNaN in submission: {sub['tvt'].isna().sum()}")
sub_path = SUBS_DIR / f"{MODEL_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Rows: {len(sub)} | TVT: [{sub['tvt'].min():.2f}, {sub['tvt'].max():.2f}]")
print(f"\n*** Val RMSE: {overall_rmse:.4f} ft ***")
