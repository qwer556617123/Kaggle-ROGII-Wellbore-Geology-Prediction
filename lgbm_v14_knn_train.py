"""
lgbm_v14_knn_train.py
---------------------
Train LightGBM v14 with kNN correction transfer features.
Validates against held-out 115-well split (val_ids.csv).
Compares to current best: row-weighted RMSE = 15.08 ft, per-well mean = 11.95 ft.

Target: target_correction  (= true_TVT - anchored_physics_col)
Final TVT = gaussian_filter1d(anchored_physics_col + predicted_correction, sigma=1.0)
"""

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
FEAT_DIR   = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"

MODELS_DIR.mkdir(exist_ok=True)
SUBS_DIR.mkdir(exist_ok=True)

# ── Load v2 features ─────────────────────────────────────────────────────────
print("=" * 65)
print("Loading v2 features (train_features_v2.parquet)...")
train_all = pd.read_parquet(FEAT_DIR / "train_features_v2.parquet")
print(f"  Loaded: {train_all.shape[0]:,} rows × {train_all.shape[1]} cols")

# ── Load saved split IDs ──────────────────────────────────────────────────────
val_ids       = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
train_ids_list = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()

train_df = train_all[train_all["well_id"].isin(set(train_ids_list))].copy()
val_df   = train_all[train_all["well_id"].isin(set(val_ids))].copy()

print(f"  Train wells : {train_df['well_id'].nunique():>4}  "
      f"({train_df.shape[0]:,} rows)")
print(f"  Val   wells : {val_df['well_id'].nunique():>4}  "
      f"({val_df.shape[0]:,} rows)")

# ── Feature columns ───────────────────────────────────────────────────────────
EXCL = {
    "well_id", "target_correction", "is_post_ps",
    "anchored_physics_col",
    # legacy names that might appear in older splits
    "target_dtvt", "target_tvt",
}
FCOLS = [c for c in train_all.columns if c not in EXCL]
print(f"\nFeature columns: {len(FCOLS)}")

knn_feats = [c for c in FCOLS if "knn" in c]
print(f"kNN features ({len(knn_feats)}): {knn_feats}")

new_tw_feats = [c for c in FCOLS if c in {
    "tw_local_gr_range", "gr_tw_at_physics", "gr_dev_from_tw",
    "tw_gradient_at_pos", "tw_curvature_at_pos"
}]
print(f"New typewell features ({len(new_tw_feats)}): {new_tw_feats}")

# ── Filter to post-PS rows only ───────────────────────────────────────────────
tr_p  = train_df[train_df["is_post_ps"] == 1]
val_p = val_df[val_df["is_post_ps"] == 1]

X_tr  = tr_p[FCOLS].values.astype(np.float32)
y_tr  = tr_p["target_correction"].values.astype(np.float32)
X_val = val_p[FCOLS].values.astype(np.float32)
y_val = val_p["target_correction"].values.astype(np.float32)

print(f"\nPost-PS train rows : {len(X_tr):,}")
print(f"Post-PS val   rows : {len(X_val):,}")

# ── LightGBM datasets ─────────────────────────────────────────────────────────
lgb_tr  = lgb.Dataset(X_tr,  label=y_tr,  feature_name=FCOLS)
lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FCOLS, reference=lgb_tr)

# ── Model params (v13_reg baseline + kNN features) ────────────────────────────
params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       255,
    "learning_rate":    0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq":     5,
    "min_child_samples": 50,
    "lambda_l1":        0.3,
    "lambda_l2":        0.3,
    "verbose":          -1,
    "seed":             42,
    "n_jobs":           -1,
}

print("\n" + "=" * 65)
print("Training LightGBM v14 (kNN features)...")
print("=" * 65)

callbacks = [
    lgb.log_evaluation(500),
    lgb.early_stopping(200, verbose=True),
]

model = lgb.train(
    params,
    lgb_tr,
    num_boost_round=20000,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)

lgb_raw_rmse = model.best_score["valid_0"]["rmse"]
print(f"\nBest round : {model.best_iteration}")
print(f"LGB raw val RMSE (no smoothing): {lgb_raw_rmse:.4f} ft")

# ── Per-well validation with Gaussian smoothing ───────────────────────────────
print("\n" + "=" * 65)
print("Computing per-well val RMSE (gaussian_filter1d σ=1.0)...")
print("=" * 65)

well_rmses = []
for wid in val_ids:
    wdf   = val_df[val_df["well_id"] == wid]
    wdf_p = wdf[wdf["is_post_ps"] == 1]
    if len(wdf_p) == 0:
        continue

    corr_pred = model.predict(wdf_p[FCOLS].values.astype(np.float32))
    phys      = wdf_p["anchored_physics_col"].values
    tvt_pred  = gaussian_filter1d(phys + corr_pred, sigma=1.0)
    tvt_true  = wdf_p["target_correction"].values + phys   # = true TVT

    rmse = np.sqrt(np.mean((tvt_pred - tvt_true) ** 2))
    well_rmses.append((wid, rmse, len(wdf_p)))

# Sort worst → best
well_rmses.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Well ID':<14} {'RMSE (ft)':>10} {'N rows':>8}")
print("-" * 36)
for wid, rmse, n in well_rmses:
    print(f"{wid:<14} {rmse:>10.3f} {n:>8}")

rmse_arr = np.array([r for _, r, _ in well_rmses])
n_arr    = np.array([n for _, _, n in well_rmses])
row_weights = n_arr / n_arr.sum()

per_well_mean   = rmse_arr.mean()
per_well_median = np.median(rmse_arr)
per_well_wt     = (rmse_arr * row_weights).sum()   # row-count-weighted mean

# Row-level RMSE (all post-PS val rows, with smoothing applied per-well concatenated)
all_pred_parts = []
all_true_parts = []
for wid in val_ids:
    wdf   = val_df[val_df["well_id"] == wid]
    wdf_p = wdf[wdf["is_post_ps"] == 1]
    if len(wdf_p) == 0:
        continue
    corr_pred = model.predict(wdf_p[FCOLS].values.astype(np.float32))
    phys      = wdf_p["anchored_physics_col"].values
    tvt_pred  = gaussian_filter1d(phys + corr_pred, sigma=1.0)
    tvt_true  = wdf_p["target_correction"].values + phys
    all_pred_parts.append(tvt_pred)
    all_true_parts.append(tvt_true)

all_pred = np.concatenate(all_pred_parts)
all_true = np.concatenate(all_true_parts)
row_weighted_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)
print(f"  Row-weighted RMSE (w/ smoothing) : {row_weighted_rmse:.4f} ft")
print(f"  Per-well mean RMSE               : {per_well_mean:.4f} ft")
print(f"  Per-well median RMSE             : {per_well_median:.4f} ft")
print(f"  Per-well wt-mean RMSE            : {per_well_wt:.4f} ft")
print(f"  Best round                       : {model.best_iteration}")
print()
print("  ── Baseline comparison ──")
print(f"  Current best row-weighted RMSE   : 15.08 ft")
print(f"  Current best per-well mean RMSE  : 11.95 ft")
improved = row_weighted_rmse < 15.08
print(f"\n  → Improved over current best?    : {'YES ✓' if improved else 'NO ✗'}")

# ── Feature importances ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TOP 20 FEATURES BY GAIN")
print("=" * 65)

imp_df = pd.DataFrame({
    "feature": FCOLS,
    "gain":    model.feature_importance("gain"),
    "split":   model.feature_importance("split"),
})
imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
imp_df["gain_pct"] = 100.0 * imp_df["gain"] / imp_df["gain"].sum()

print(f"\n{'Rank':<5} {'Feature':<30} {'Gain':>12} {'Gain%':>7} {'Split':>8}")
print("-" * 65)
for i, row in imp_df.head(20).iterrows():
    marker = " ◄ kNN" if "knn" in row["feature"] else ""
    print(f"{i+1:<5} {row['feature']:<30} {row['gain']:>12,.0f} "
          f"{row['gain_pct']:>6.2f}% {row['split']:>8}{marker}")

# Save importance CSV
imp_path = MODELS_DIR / "lgbm_v14_knn_importance.csv"
imp_df.to_csv(imp_path, index=False)
print(f"\nFeature importance saved → {imp_path}")

# ── Save model ─────────────────────────────────────────────────────────────────
model_path = MODELS_DIR / "lgbm_v14_knn.pkl"
joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

# ── Save best_iteration for final training ────────────────────────────────────
meta = {
    "best_iteration":      model.best_iteration,
    "row_weighted_rmse":   row_weighted_rmse,
    "per_well_mean_rmse":  per_well_mean,
    "per_well_median_rmse": per_well_median,
    "improved":            improved,
    "fcols":               FCOLS,
}
joblib.dump(meta, MODELS_DIR / "lgbm_v14_knn_meta.pkl")
print(f"Meta saved  → {MODELS_DIR / 'lgbm_v14_knn_meta.pkl'}")

print("\n" + "=" * 65)
if improved:
    print("✓ Val RMSE IMPROVED — ready to run lgbm_v14_knn_final_train.py")
else:
    print("✗ Val RMSE did NOT improve — do NOT run final training")
    print("  Possible causes:")
    print("  • kNN features may have train/val leakage")
    print("  • kNN features computed from correlated neighbour pool")
    print("  • Consider ablation: disable kNN features, re-train")
print("=" * 65)
