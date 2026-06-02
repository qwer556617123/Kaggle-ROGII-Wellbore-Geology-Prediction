"""
LightGBM V3 — ROGII Wellbore Geology Prediction
================================================
Key improvements over v2:
1. GR matching uses TRUE TVT as search center during training (not PS anchor)
2. Search radius increased to ±300 typewell steps = ±150 ft
3. Autoregressive inference: predicted TVT feeds back into GR matching per step
4. No train/test feature mismatch (only use GR-match-based features in model)
5. Added pre-PS Z-slope feature (well trajectory baseline)
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

DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
MODELS_DIR.mkdir(exist_ok=True); SUBS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "lgbm_v3"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

def fill_arr(arr):
    return pd.Series(arr).ffill().bfill().astype(float).values

def load_well(well_id, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(d / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{well_id}__typewell.csv")
    return hw, tw

def get_ps_index(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

# ── GR cross-correlation TVT matching ──────────────────────────────────────
def compute_gr_match_tvt(hw_gr, tw_gr_smooth, tw_tvt, center_tvt,
                          template_half=20, search_half=300):
    """
    For each row i:
      - Takes horizontal GR window of size 2*template_half+1
      - Searches typewell GR ± search_half around center_tvt[i]
      - Returns typewell TVT with max cross-correlation
    center_tvt: array of per-row search center TVT values (e.g., true TVT or predicted TVT)
    """
    n = len(hw_gr)
    tw_n = len(tw_gr_smooth)
    match_tvt = np.full(n, np.nan)
    W = template_half
    S = search_half

    hw_smooth = gaussian_filter1d(fill_arr(hw_gr), sigma=2.0)

    for i in range(n):
        t_lo, t_hi = max(0, i - W), min(n, i + W + 1)
        template = hw_smooth[t_lo:t_hi].copy()
        template -= template.mean()
        if len(template) < 5 or template.std() < 0.5:
            match_tvt[i] = center_tvt[i]
            continue

        tw_center = int(np.searchsorted(tw_tvt, center_tvt[i]))
        tw_lo = max(0, tw_center - S)
        tw_hi = min(tw_n, tw_center + S + len(template))
        tw_seg = tw_gr_smooth[tw_lo:tw_hi]

        if len(tw_seg) < len(template):
            match_tvt[i] = center_tvt[i]
            continue

        corr = correlate(tw_seg, template, mode='valid')
        best_j = int(np.argmax(corr))
        best_tw_idx = np.clip(tw_lo + best_j + W, 0, tw_n - 1)
        match_tvt[i] = float(tw_tvt[best_tw_idx])

    nans = np.isnan(match_tvt)
    if nans.any() and (~nans).any():
        x = np.arange(n)
        match_tvt[nans] = np.interp(x[nans], x[~nans], match_tvt[~nans])
    return match_tvt


def build_core_features(hw, tw, well_id):
    """
    Build features that don't depend on TVT (GR stats, trajectory).
    Returns: dict of arrays, scalar ps_idx, arrays of md/z/gr/tw metadata
    """
    n = len(hw)
    ps_idx = get_ps_index(hw)

    gr_raw = fill_arr(hw["GR"])
    md = hw["MD"].astype(float).values
    z  = hw["Z"].astype(float).values
    x  = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values

    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = tvt_inp.ffill().bfill().values

    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr   = fill_arr(tw["GR"])
    tw_gr_s = gaussian_filter1d(tw_gr, sigma=1.0)
    tw_gr_grad = np.gradient(tw_gr, tw_tvt)

    feat = {}
    feat["well_id"] = [well_id] * n

    # GR features
    gr_s = pd.Series(gr_raw)
    feat["gr"] = gr_raw
    feat["gr_diff"]  = np.gradient(gr_raw)
    feat["gr_diff2"] = np.gradient(np.gradient(gr_raw))
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values

    # Trajectory
    feat["md"]         = md
    feat["z"]          = z
    feat["dz_dmd"]     = np.gradient(z, md)
    feat["dx_dmd"]     = np.gradient(x, md)
    feat["dy_dmd"]     = np.gradient(y, md)
    feat["inclination"] = np.arctan2(
        np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
        np.abs(np.gradient(z)) + 1e-9
    ) * 180 / np.pi

    feat["last_known_tvt"] = last_known_tvt
    feat["rows_since_ps"]  = np.maximum(0, np.arange(n) - ps_idx).astype(float)
    feat["rows_before_ps"] = np.maximum(0, ps_idx - np.arange(n)).astype(float)
    feat["ps_idx"]         = float(ps_idx)

    # Pre-PS trajectory: avg dZ/dMD over last 50 pre-PS rows (constant across well)
    if ps_idx > 0:
        pre_dz  = np.gradient(z[:ps_idx], md[:ps_idx]) if ps_idx > 1 else np.array([0.0])
        pre_tvt = tvt_inp[:ps_idx].values
        pre_dtvt = np.diff(pre_tvt[~np.isnan(pre_tvt)]) if np.sum(~np.isnan(pre_tvt)) > 1 else np.array([0.0])
        feat["pre_ps_dz_slope"]  = float(pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = float(pre_dtvt[-min(50, len(pre_dtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = float(pre_dtvt[-min(50, len(pre_dtvt)):].std()) if len(pre_dtvt) > 1 else 0.0
    else:
        feat["pre_ps_dz_slope"]  = 0.0
        feat["pre_ps_dtvt_mean"] = 0.0
        feat["pre_ps_dtvt_std"]  = 0.0
    feat["pre_ps_dz_slope"]  = np.full(n, feat["pre_ps_dz_slope"])
    feat["pre_ps_dtvt_mean"] = np.full(n, feat["pre_ps_dtvt_mean"])
    feat["pre_ps_dtvt_std"]  = np.full(n, feat["pre_ps_dtvt_std"])

    return feat, ps_idx, md, z, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt


def add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad,
                                 center_tvt, feat_prefix=""):
    """
    Compute typewell-related features given per-row TVT estimates (center_tvt).
    Adds to feat dict in-place.
    """
    n = len(gr_raw)

    # GR cross-correlation match
    gr_match = compute_gr_match_tvt(
        gr_raw, tw_gr_s, tw_tvt, center_tvt,
        template_half=20, search_half=300
    )
    feat[f"{feat_prefix}gr_match_tvt"] = gr_match

    # Typewell features at matched position
    tw_match_idx = np.searchsorted(tw_tvt, gr_match).clip(0, len(tw_tvt) - 1)
    feat[f"{feat_prefix}tw_gr_at_match"]  = tw_gr_s[tw_match_idx]
    feat[f"{feat_prefix}gr_match_diff"]   = gr_raw - tw_gr_s[tw_match_idx]
    feat[f"{feat_prefix}match_vs_anchor"] = gr_match - center_tvt
    feat[f"{feat_prefix}tw_gr_grad_match"] = tw_gr_grad[tw_match_idx]

    # GR match delta (rate of TVT position change from GR)
    feat[f"{feat_prefix}gr_match_dtvt"] = np.gradient(gr_match)


# ── Build features for a well ───────────────────────────────────────────────
def build_features_v3(well_id, split="train"):
    hw, tw = load_well(well_id, split)
    n = len(hw)

    feat, ps_idx, md, z, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt = \
        build_core_features(hw, tw, well_id)

    # For training: use true TVT as GR matching center (better signal)
    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        center_tvt = tvt_true.copy()  # Perfect center for training
        # Add TVT-dependent features with true TVT center
        add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, center_tvt, "")
        # Also add PS-anchor-based features (what test would start with)
        add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt, "anc_")
        # Targets
        feat["target_tvt"]    = tvt_true
        feat["target_dtvt_1"] = np.concatenate([[0.0], np.diff(tvt_true)])
        feat["is_post_ps"]    = (np.arange(n) >= ps_idx).astype(int)
    else:
        # Test: use last known TVT as initial center
        center_tvt = last_known_tvt.copy()
        add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, center_tvt, "")
        add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, center_tvt, "anc_")
        feat["is_post_ps"] = (np.arange(n) >= ps_idx).astype(int)

    return pd.DataFrame(feat), ps_idx, tw_tvt, tw_gr_s, tw_gr_grad, gr_raw


# ── Main: build train/val features ─────────────────────────────────────────
print("=" * 60)
print("LightGBM V3 — ROGII Wellbore Geology Prediction")
print("=" * 60)

train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"Train wells: {len(train_ids)}, Val wells: {len(val_ids)}")

print("\nBuilding train features (true-TVT-centered GR matching)...")
train_dfs = []
for i, wid in enumerate(train_ids):
    try:
        df, *_ = build_features_v3(wid, "train")
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
        df, *_ = build_features_v3(wid, "train")
        val_dfs.append(df)
        if (i+1) % 50 == 0 or i+1 == len(val_ids):
            print(f"  {i+1}/{len(val_ids)}")
    except Exception as e:
        print(f"  SKIP {wid}: {e}")
val_all = pd.concat(val_dfs, ignore_index=True)
print(f"Val shape: {val_all.shape}")

# ── Prepare training matrices ───────────────────────────────────────────────
EXCLUDE = {"well_id", "target_tvt", "target_dtvt_1", "is_post_ps"}
FEATURE_COLS = [c for c in train_all.columns if c not in EXCLUDE]

train_post = train_all[train_all["is_post_ps"] == 1].copy()
val_post   = val_all[val_all["is_post_ps"] == 1].copy()
print(f"\nPost-PS: train={len(train_post)}, val={len(val_post)}")
print(f"Features: {len(FEATURE_COLS)}")

X_tr = train_post[FEATURE_COLS].values.astype(np.float32)
y_tr = train_post["target_dtvt_1"].values.astype(np.float32)
X_vl = val_post[FEATURE_COLS].values.astype(np.float32)
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
lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
lgb_vl = lgb.Dataset(X_vl, label=y_vl, reference=lgb_tr)

print("\nTraining LightGBM v3...")
model = lgb.train(
    params, lgb_tr, num_boost_round=3000,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(200, verbose=True),
        lgb.log_evaluation(100),
    ],
)

joblib.dump(model, MODELS_DIR / f"{MODEL_NAME}.pkl")
print(f"Model saved. Best iteration: {model.best_iteration}")

imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
imp.to_csv(MODELS_DIR / f"{MODEL_NAME}_importance.csv", index=False)
print("Top-15 features:")
print(imp.head(15).to_string(index=False))

# ── Validation: integrate predictions per well ─────────────────────────────
# For validation, features were built with TRUE TVT center (oracle)
# We evaluate by integrating one-step dTVT predictions
print("\n" + "=" * 55)
print("Validation (oracle features, integrate dTVT)...")
print("=" * 55)

val_all["pred_dtvt_1"] = model.predict(
    val_all[FEATURE_COLS].values.astype(np.float32)
)

well_rmse = []
for wid, grp in val_all.groupby("well_id"):
    grp = grp.sort_index()
    post = grp[grp["is_post_ps"] == 1]
    if len(post) == 0: continue
    start_tvt = float(post["last_known_tvt"].iloc[0])
    pred_tvt  = start_tvt + np.cumsum(post["pred_dtvt_1"].values)
    truth     = post["target_tvt"].values
    rmse = np.sqrt(np.mean((truth - pred_tvt) ** 2))
    well_rmse.append((wid, rmse, len(post)))

well_rmse.sort(key=lambda x: -x[1])
for wid, rmse, n in well_rmse:
    print(f"  {wid}  RMSE={rmse:7.3f}  n={n}")

oracle_rmse = np.sqrt(np.mean([
    (truth[j] - tvt_pred) ** 2
    for _, grp in val_all.groupby("well_id")
    for j, tvt_pred in enumerate(
        (float(grp[grp["is_post_ps"]==1]["last_known_tvt"].iloc[0]) +
         np.cumsum(model.predict(grp[grp["is_post_ps"]==1][FEATURE_COLS].values.astype(np.float32))))
        if (grp["is_post_ps"]==1).any() else []
    )
    for truth in [grp[grp["is_post_ps"]==1]["target_tvt"].values]
]))
print(f"\n  *** Oracle Val RMSE: {oracle_rmse:.4f} ft ***")

# ── Test: autoregressive inference with TVT feedback ──────────────────────
print("\n" + "=" * 55)
print("Test: autoregressive inference...")
print("=" * 55)

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

for well_id in TEST_WELLS:
    print(f"\n[{well_id}]")
    hw, tw = load_well(well_id, "test")
    ps_idx = get_ps_index(hw)
    n = len(hw)
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    tw_gr_s = gaussian_filter1d(tw_gr, sigma=1.0)
    tw_gr_grad = np.gradient(tw_gr, tw_tvt)

    # Build base features (without TVT-dependent ones)
    feat, ps_idx_out, md, z, gr_raw, tw_tvt_out, tw_gr_s_out, tw_gr_grad_out, last_known_tvt = \
        build_core_features(hw, tw, well_id)

    # Add is_post_ps (not added by build_core_features)
    feat["is_post_ps"] = (np.arange(n) >= ps_idx_out).astype(int)

    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    start_tvt = float(tvt_inp.ffill().iloc[max(0, ps_idx - 1)])
    print(f"  Start TVT: {start_tvt:.2f} ft")

    # ── 2-pass autoregressive refinement ──
    # Pass 1: use PS anchor as center for all rows
    add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt, "")
    add_tvt_dependent_features(feat, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt, "anc_")

    df = pd.DataFrame(feat)
    df_post = df[df["is_post_ps"] == 1].copy().reset_index(drop=True)
    X_post = df_post[FEATURE_COLS].values.astype(np.float32)
    pred_dtvts_pass1 = model.predict(X_post)
    pred_tvt_pass1 = start_tvt + np.cumsum(pred_dtvts_pass1)

    # Pass 2: use pass-1 predictions as center for GR matching
    center_tvt_pass2 = last_known_tvt.copy()
    post_indices = np.where(df["is_post_ps"].values == 1)[0]
    for k, idx in enumerate(post_indices):
        center_tvt_pass2[idx] = pred_tvt_pass1[k]

    feat2 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in feat.items()
             if k not in ["gr_match_tvt", "tw_gr_at_match", "gr_match_diff",
                          "match_vs_anchor", "tw_gr_grad_match", "gr_match_dtvt",
                          "anc_gr_match_tvt", "anc_tw_gr_at_match", "anc_gr_match_diff",
                          "anc_match_vs_anchor", "anc_tw_gr_grad_match", "anc_gr_match_dtvt"]}
    add_tvt_dependent_features(feat2, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, center_tvt_pass2, "")
    add_tvt_dependent_features(feat2, gr_raw, tw_tvt, tw_gr_s, tw_gr_grad, last_known_tvt, "anc_")

    df2 = pd.DataFrame(feat2)
    df2_post = df2[df2["is_post_ps"] == 1].copy().reset_index(drop=True)
    X_post2 = df2_post[FEATURE_COLS].values.astype(np.float32)
    pred_dtvts_pass2 = model.predict(X_post2)
    pred_tvt_pass2 = start_tvt + np.cumsum(pred_dtvts_pass2)
    pred_tvt_final = gaussian_filter1d(pred_tvt_pass2.astype(float), sigma=1.0)

    print(f"  Pass1 TVT: [{pred_tvt_pass1.min():.1f}, {pred_tvt_pass1.max():.1f}]")
    print(f"  Pass2 TVT: [{pred_tvt_final.min():.1f}, {pred_tvt_final.max():.1f}]")

    # Map predictions to submission IDs
    empty_mask  = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    row_indices = hw.index[empty_mask].tolist()
    assert len(row_indices) == len(pred_tvt_final)
    id_to_tvt = {f"{well_id}_{i}": v for i, v in zip(row_indices, pred_tvt_final)}
    mask = sub["id"].isin(set(id_to_tvt.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_to_tvt)

print(f"\nNaN in submission: {sub['tvt'].isna().sum()}")
sub_path = SUBS_DIR / f"{MODEL_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Rows: {len(sub)} | TVT: [{sub['tvt'].min():.2f}, {sub['tvt'].max():.2f}]")
print(f"\n*** Oracle Val RMSE: {oracle_rmse:.4f} ft ***")
print("Note: true test RMSE will be higher (oracle uses true TVT for GR matching)")
