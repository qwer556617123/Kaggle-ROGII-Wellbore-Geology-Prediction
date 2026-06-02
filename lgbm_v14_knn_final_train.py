"""
lgbm_v14_knn_final_train.py
----------------------------
Final training on ALL 773 wells using best_iteration from val run.
Generates test predictions and saves submission.

Prerequisites:
    Run lgbm_v14_knn_train.py first.
    models/lgbm_v14_knn_meta.pkl must exist and improved == True.

Final TVT = gaussian_filter1d(physics_tvt + predicted_correction, sigma=1.0)
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
TEST_DIR   = DATA_DIR / "test"

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── Guard: only run if val run showed improvement ─────────────────────────────
meta_path = MODELS_DIR / "lgbm_v14_knn_meta.pkl"
if not meta_path.exists():
    raise FileNotFoundError(
        "lgbm_v14_knn_meta.pkl not found. "
        "Run lgbm_v14_knn_train.py first."
    )

meta = joblib.load(meta_path)
print("=" * 65)
print("Loaded meta from val run:")
print(f"  best_iteration     : {meta['best_iteration']}")
print(f"  row_weighted_rmse  : {meta['row_weighted_rmse']:.4f} ft")
print(f"  per_well_mean_rmse : {meta['per_well_mean_rmse']:.4f} ft")
print(f"  improved           : {meta['improved']}")

if not meta["improved"]:
    print("\n✗ Val run did NOT improve over baseline (15.08 ft).")
    print("  Aborting final training — check for leakage in kNN features.")
    raise SystemExit(1)

best_iter = meta["best_iteration"]
FCOLS     = meta["fcols"]
print(f"\n✓ Val improved — training final model with {best_iter} rounds.")
print("=" * 65)

# ── Load ALL training data (773 wells) ────────────────────────────────────────
print("\nLoading v2 features (all 773 wells)...")
train_all = pd.read_parquet(FEAT_DIR / "train_features_v2.parquet")
print(f"  Loaded: {train_all.shape[0]:,} rows × {train_all.shape[1]} cols")
print(f"  Wells  : {train_all['well_id'].nunique()}")

# Filter to post-PS rows only
tr_p = train_all[train_all["is_post_ps"] == 1]
X_all = tr_p[FCOLS].values.astype(np.float32)
y_all = tr_p["target_correction"].values.astype(np.float32)
print(f"  Post-PS rows: {len(X_all):,}")

# ── Build LGB dataset (no validation — fixed num_boost_round) ─────────────────
lgb_all = lgb.Dataset(X_all, label=y_all, feature_name=FCOLS)

params = {
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        255,
    "learning_rate":     0.02,
    "feature_fraction":  0.7,
    "bagging_fraction":  0.7,
    "bagging_freq":      5,
    "min_child_samples": 50,
    "lambda_l1":         0.3,
    "lambda_l2":         0.3,
    "verbose":           -1,
    "seed":              42,
    "n_jobs":            -1,
}

print(f"\nTraining final model for {best_iter} rounds on all 773 wells...")
model_final = lgb.train(
    params,
    lgb_all,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(500)],
)
print("Training complete.")

# ── Save final model ──────────────────────────────────────────────────────────
final_model_path = MODELS_DIR / "lgbm_v14_knn_final.pkl"
joblib.dump(model_final, final_model_path)
print(f"Final model saved → {final_model_path}")

# ── Load test features ────────────────────────────────────────────────────────
print("\nLoading test_features_v2.parquet...")
test_df = pd.read_parquet(FEAT_DIR / "test_features_v2.parquet")
print(f"  Loaded: {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")
print(f"  Test wells: {test_df['well_id'].unique().tolist()}")

# Verify FCOLS all present in test
missing_in_test = [c for c in FCOLS if c not in test_df.columns]
if missing_in_test:
    print(f"  WARNING: {len(missing_in_test)} feature(s) missing in test → {missing_in_test}")
    print("  Dropping missing features from FCOLS...")
    FCOLS_TEST = [c for c in FCOLS if c in test_df.columns]
else:
    FCOLS_TEST = FCOLS
    print(f"  All {len(FCOLS)} features present in test ✓")

# ── Load sample submission ────────────────────────────────────────────────────
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"\nSample submission: {sub.shape[0]:,} rows")

# ── Generate test predictions ─────────────────────────────────────────────────
print("\nGenerating test predictions...")

for wid in TEST_WELLS:
    print(f"  Well {wid}...")

    # test features rows for this well
    wdf_p = test_df[test_df["well_id"] == wid].copy()
    print(f"    Feature rows: {len(wdf_p)}")

    # Predict correction
    corr = model_final.predict(wdf_p[FCOLS_TEST].values.astype(np.float32))

    # Use physics_tvt as baseline (= anchored_physics_col, same computation)
    phys = wdf_p["physics_tvt"].values
    tvt_pred = gaussian_filter1d(phys + corr, sigma=1.0)

    # ── Map predictions to submission row IDs ─────────────────────────────────
    # Submission IDs: {well_id}_{row_index_in_original_csv}
    # Load original test CSV to get exact post-PS row indices
    hw = pd.read_csv(TEST_DIR / f"{wid}__horizontal_well.csv")
    post_ps_mask = (
        hw["TVT_input"].isna()
        | (hw["TVT_input"].astype(str).str.strip() == "")
    )
    post_ps_rows = hw.index[post_ps_mask].tolist()

    print(f"    Original CSV post-PS rows: {len(post_ps_rows)}")
    print(f"    TVT predictions          : {len(tvt_pred)}")

    if len(post_ps_rows) != len(tvt_pred):
        print(f"    ⚠ MISMATCH — aligning to min({len(post_ps_rows)}, {len(tvt_pred)})")
        n = min(len(post_ps_rows), len(tvt_pred))
        post_ps_rows = post_ps_rows[:n]
        tvt_pred     = tvt_pred[:n]

    id_map = {f"{wid}_{i}": v for i, v in zip(post_ps_rows, tvt_pred)}
    mask = sub["id"].isin(set(id_map.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)
    print(f"    Filled {mask.sum()} submission rows")

    # Sanity check: range of predicted TVT
    print(f"    TVT pred range: [{tvt_pred.min():.1f}, {tvt_pred.max():.1f}] ft")

# ── Verify no NaN in submission ───────────────────────────────────────────────
nan_count = sub["tvt"].isna().sum()
if nan_count > 0:
    print(f"\n⚠ WARNING: {nan_count} NaN values in submission — filling with 0")
    sub["tvt"] = sub["tvt"].fillna(0.0)
else:
    print(f"\n✓ No NaN values in submission")

# ── Save submission ───────────────────────────────────────────────────────────
sub_path = SUBS_DIR / "lgbm_v14_knn_final.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved → {sub_path}")
print(f"  Rows: {len(sub):,}")
print(f"  TVT range: [{sub['tvt'].min():.1f}, {sub['tvt'].max():.1f}] ft")
print(f"  TVT mean : {sub['tvt'].mean():.2f} ft")

print("\n" + "=" * 65)
print("DONE — lgbm_v14_knn_final submission ready.")
print(f"  Model   : {final_model_path}")
print(f"  Submit  : {sub_path}")
print("=" * 65)
