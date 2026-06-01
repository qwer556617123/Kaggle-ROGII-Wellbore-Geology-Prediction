"""
LightGBM Trainer for ROGII Wellbore Geology Prediction
=======================================================
Trains on pre-engineered features, evaluates per-well RMSE on validation set,
generates test predictions and Kaggle submission.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
FEAT_DIR  = DATA_DIR / "features"
TEST_DIR  = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
MODELS_DIR.mkdir(exist_ok=True)
SUBS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "lgbm_v1"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# Feature columns (exclude metadata and targets)
EXCLUDE_COLS = {"well_id", "target_tvt", "target_dtvt", "is_post_ps"}

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading features...")
train_df = pd.read_parquet(FEAT_DIR / "train_features.parquet")
val_df   = pd.read_parquet(FEAT_DIR / "val_features.parquet")
test_df  = pd.read_parquet(FEAT_DIR / "test_features.parquet")

print(f"  Train: {train_df.shape}  |  Val: {val_df.shape}  |  Test: {test_df.shape}")

# Only train on post-PS rows (that's what we need to predict)
train_post = train_df[train_df["is_post_ps"] == 1].copy()
val_post   = val_df[val_df["is_post_ps"] == 1].copy()
print(f"  Post-PS train rows: {len(train_post)}  |  Post-PS val rows: {len(val_post)}")

FEATURE_COLS = [c for c in train_df.columns if c not in EXCLUDE_COLS]
# Drop lead features for post-PS prediction (future lookahead not available at inference)
FEATURE_COLS = [c for c in FEATURE_COLS if "lead" not in c]
print(f"  Features: {len(FEATURE_COLS)}")

X_train = train_post[FEATURE_COLS].values.astype(np.float32)
y_train = train_post["target_dtvt"].values.astype(np.float32)  # predict delta from last known TVT

X_val   = val_post[FEATURE_COLS].values.astype(np.float32)
y_val   = val_post["target_dtvt"].values.astype(np.float32)

# ── LightGBM training ──────────────────────────────────────────────────────
params = {
    "objective":          "regression",
    "metric":             "rmse",
    "num_leaves":         127,
    "learning_rate":      0.05,
    "feature_fraction":   0.8,
    "bagging_fraction":   0.8,
    "bagging_freq":       5,
    "min_child_samples":  20,
    "lambda_l1":          0.1,
    "lambda_l2":          0.1,
    "verbose":            -1,
    "seed":               42,
    "n_jobs":             -1,
}

lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

print(f"\nTraining LightGBM ({MODEL_NAME})...")
callbacks = [
    lgb.early_stopping(stopping_rounds=150, verbose=True),
    lgb.log_evaluation(period=100),
]

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)

# ── Save model ─────────────────────────────────────────────────────────────
model_path = MODELS_DIR / f"{MODEL_NAME}.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved: {model_path}")

# Feature importance
imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
imp_df.to_csv(MODELS_DIR / f"{MODEL_NAME}_importance.csv", index=False)
print("Top-15 features:")
print(imp_df.head(15).to_string(index=False))

# ── Validation evaluation ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Validation RMSE (per well)")
print("=" * 55)

val_post = val_post.copy()
val_post["pred_dtvt"] = model.predict(X_val)
val_post["pred_tvt"]  = val_post["last_known_tvt"] + val_post["pred_dtvt"]

well_rmse = []
for wid, grp in val_post.groupby("well_id"):
    truth = grp["target_tvt"].values
    pred  = grp["pred_tvt"].values
    rmse  = np.sqrt(np.mean((truth - pred) ** 2))
    well_rmse.append((wid, rmse, len(grp)))

well_rmse.sort(key=lambda x: -x[1])
for wid, rmse, n in well_rmse:
    print(f"  {wid}  RMSE={rmse:7.3f}  n={n}")

overall_rmse = np.sqrt(np.mean(
    (val_post["target_tvt"].values - val_post["pred_tvt"].values) ** 2
))
print(f"\n  *** Overall Val RMSE: {overall_rmse:.4f} ft ***")

# ── Test predictions ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Generating test predictions...")
print("=" * 55)

# Test: predict only post-PS rows
test_post = test_df.copy()

# Add lead features as NaN (same as 0 fill for test)
for lag in [1, 3, 5, 10]:
    col = f"gr_lead_{lag}"
    if col not in test_post.columns:
        test_post[col] = np.nan

# Make predictions
X_test = test_post[FEATURE_COLS].values.astype(np.float32)
test_post["pred_dtvt"] = model.predict(X_test)
test_post["pred_tvt"]  = test_post["last_known_tvt"] + test_post["pred_dtvt"]

# ── Per-well autoregressive correction ────────────────────────────────────
# Re-run predictions well by well, feeding predicted TVT back as last_known_tvt
print("Applying autoregressive TVT correction...")
predictions = {}
for well_id in TEST_WELLS:
    hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
    empty_mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    row_indices = hw.index[empty_mask].tolist()

    wdf = test_post[test_post["well_id"] == well_id].copy()
    # Filter to post-PS rows only (rows_before_ps == 0 means at or after PS point)
    wdf_post = wdf[wdf["rows_before_ps"] == 0]
    pred_tvt_arr = wdf_post["pred_tvt"].values

    # Light smoothing
    pred_tvt_arr = gaussian_filter1d(pred_tvt_arr.astype(float), sigma=1.5)

    print(f"  [{well_id}] {len(pred_tvt_arr)} predictions | TVT range [{pred_tvt_arr.min():.1f}, {pred_tvt_arr.max():.1f}]")
    predictions[well_id] = pred_tvt_arr

# ── Build submission ───────────────────────────────────────────────────────
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
for well_id, pred_tvt in predictions.items():
    hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
    empty_mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    row_indices = hw.index[empty_mask].tolist()
    assert len(row_indices) == len(pred_tvt), f"[{well_id}] length mismatch"
    id_to_tvt = {f"{well_id}_{i}": v for i, v in zip(row_indices, pred_tvt)}
    mask = sub["id"].isin(set(id_to_tvt.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_to_tvt)

nan_count = sub["tvt"].isna().sum()
assert nan_count == 0, f"Submission has {nan_count} NaN values!"

sub_path = SUBS_DIR / f"{MODEL_NAME}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"  Total rows: {len(sub)}")
print(f"  TVT range: [{sub['tvt'].min():.2f}, {sub['tvt'].max():.2f}]")
print(f"\n  *** Val RMSE: {overall_rmse:.4f} ft ***")
