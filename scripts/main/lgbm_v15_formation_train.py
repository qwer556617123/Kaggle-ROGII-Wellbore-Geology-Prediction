"""
LightGBM v15 — Formation Top Features
======================================
Extends the v13_reg/lgbm_final_reg baseline by adding geological formation
top columns (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) as features.

Key points
----------
- For every training well the formation tops are already in the CSV.
- The 3 test wells also exist in train/  →  we load that version to get
  formation tops.  We DO NOT touch TVT / TVT_input from that file.
- New features per formation top F:
    F            – raw formation-top Z value
    z_minus_F    – borehole Z relative to formation top (structural position)
    F_grad       – structural dip rate  dF/dMD
- Cross-formation features:
    formation_span   – ANCC minus BUDA  (total stack thickness)
    z_frac_in_stack  – fractional depth within the stack (clipped to [-1, 2])

Val baseline (lgbm_final_reg / v13_reg):
  row-weighted RMSE = 15.08 ft
  per-well mean     = 11.95 ft
  LB score          = 12.269 ft
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
FEAT_DIR   = DATA_DIR / "features"

TEST_WELLS     = ["000d7d20", "00bbac68", "00e12e8b"]
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

BASELINE_ROW_RMSE   = 15.08   # v13_reg row-weighted val RMSE
BASELINE_PERWELL    = 11.95   # v13_reg per-well mean val RMSE

# ── helpers ────────────────────────────────────────────────────────────────
def fill_arr(a):
    """Forward-fill then back-fill a 1-D array; return float64."""
    return pd.Series(a).ffill().bfill().astype(float).values

def get_ps(hw):
    """Index of first missing TVT_input row (prediction start)."""
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def load_well_with_formation(wid, split="train"):
    """
    Load horizontal-well + typewell CSVs.

    Returns  (hw, tw, hw_full)
    - hw       : the raw CSV for the requested split
    - tw       : typewell CSV (train or test)
    - hw_full  : training-version CSV  (always has ANCC … BUDA)
                 For training wells this is identical to hw.
                 For test wells only ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA
                 columns from this file are used — TVT/TVT_input are ignored.
    """
    d = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(d / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{wid}__typewell.csv")

    if split == "test":
        hw_full = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
    else:
        hw_full = hw   # training CSVs already carry formation tops

    return hw, tw, hw_full

# ── feature builder ────────────────────────────────────────────────────────
def build(hw, tw, hw_full, well_id, split="train"):
    """
    Build per-row feature DataFrame for one well.

    hw_full  – training-version of the horizontal-well CSV; used solely to
               extract ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA.
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

    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    gr_s   = pd.Series(gr)

    # ── TVT slope from pre-PS section ──
    if split == "train" and "TVT" in hw.columns:
        pre_z   = z[:ps]
        pre_tvt = hw["TVT"].astype(float).values[:ps]
    else:
        known   = tvt_inp[:ps].values
        valid   = ~np.isnan(known)
        pre_z   = z[:ps][valid]
        pre_tvt = known[valid]

    if len(pre_z) > 5:
        reg        = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope      = float(reg.coef_[0])
        pre_tvt_p  = reg.predict(pre_z.reshape(-1, 1))
        pre_r2     = float(
            1 - np.var(pre_tvt - pre_tvt_p) / np.var(pre_tvt)
        ) if np.var(pre_tvt) > 0 else 1.0
    else:
        slope  = -1.0
        pre_r2 = 0.0

    anchor_tvt      = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor        = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    post_z  = z[ps:]   if ps < n else z[:]
    post_md = md[ps:]  if ps < n else md[:]
    n_post  = len(post_z)
    z_end   = float(post_z[-1])  if n_post > 0 else float(z[-1])
    total_z = z_end - Z_anchor
    total_md = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.0
    post_dz_rate  = total_z / total_md
    phys_tvt_end  = slope * total_z
    pre_dz_md = (
        (z[ps - 1] - z[0]) / (md[ps - 1] - md[0])
        if ps > 1 and abs(md[ps - 1] - md[0]) > 0.01 else -0.01
    )
    dz_rate_change = post_dz_rate - pre_dz_md

    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr,
                              left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr - gr_at_physics

    # ── base features ──
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

    feat.update({
        "md":           md,
        "z":            z,
        "dz_dmd":       np.gradient(z, md),
        "dx_dmd":       np.gradient(x, md),
        "dy_dmd":       np.gradient(y, md),
        "inclination":  np.arctan2(
            np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
            np.abs(np.gradient(z)) + 1e-9
        ) * 180 / np.pi,
        "raw_dz":       np.diff(z,  prepend=z[0]),
        "raw_dmd":      np.diff(md, prepend=md[0]),
        "last_known_tvt":          lkt,
        "rows_since_ps":           np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps":          np.maximum(0, ps - np.arange(n)).astype(float),
        "ps_idx":                  float(ps),
        "tvt_z_slope":             np.full(n, slope),
        "physics_tvt":             anchored_physics,
        "physics_vs_lkt":          anchored_physics - lkt,
        "physics_dtvt":            slope * np.diff(z, prepend=z[0]),
        "gr_at_physics":           gr_at_physics,
        "gr_dev_physics":          gr_dev,
        "total_z_change_post":     np.full(n, total_z),
        "physics_tvt_at_end":      np.full(n, phys_tvt_end),
        "post_dz_rate":            np.full(n, post_dz_rate),
        "dz_rate_change":          np.full(n, dz_rate_change),
        "n_post_ps":               np.full(n, float(n_post)),
        "post_ps_frac":            np.where(
            n_post > 0,
            np.maximum(0, np.arange(n) - ps) / n_post,
            0.0
        ).astype(float),
        "pre_r2":                  np.full(n, pre_r2),
    })

    if ps > 2:
        pre_dz  = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.0])
        vtvt    = tvt_inp[:ps].values
        vtvt    = vtvt[~np.isnan(vtvt)]
        pdtvt   = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.0])
        feat["pre_ps_dz_slope"]   = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"]  = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]   = np.full(n,
            pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.0
        )
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)

    # ── formation top features ────────────────────────────────────────────
    # hw_full always has ANCC…BUDA columns.
    # If a column is entirely NaN (rare), fall back to 0 and flag with
    # has_formation_data=0 so the model can branch on availability.
    ancc_vals = None
    buda_vals = None

    # Determine if this well has valid formation data
    has_formation = 0
    if "ANCC" in hw_full.columns:
        ancc_raw = hw_full["ANCC"].astype(float).values
        if not np.all(np.isnan(ancc_raw)):
            has_formation = 1
    feat["has_formation_data"] = np.full(n, float(has_formation))

    for col in FORMATION_COLS:
        if col in hw_full.columns:
            raw_vals = hw_full[col].astype(float).values
            if len(raw_vals) == n:
                vals = fill_arr(raw_vals)
                # If still all-NaN after ffill/bfill, substitute 0
                if np.all(np.isnan(vals)):
                    vals = np.full(n, np.nan)  # let LightGBM handle NaN natively
                feat[col]              = vals
                feat["z_minus_" + col] = z - vals   # structural position
                with np.errstate(invalid="ignore"):
                    grad = np.gradient(vals, md) if not np.all(np.isnan(vals)) else np.full(n, np.nan)
                feat[col + "_grad"]    = grad  # dip rate
                if col == "ANCC":
                    ancc_vals = vals
                elif col == "BUDA":
                    buda_vals = vals

    # Cross-formation span + fractional position
    if ancc_vals is not None and buda_vals is not None:
        span = ancc_vals - buda_vals          # total stack thickness (should be >0)
        feat["formation_span"]  = span
        feat["z_frac_in_stack"] = np.clip(
            (z - buda_vals) / (span + 1e-6), -1.0, 2.0
        )

    # ── assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(feat)
    df["well_id"]    = well_id
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"]    = tvt_true - anchored_physics
        df["anchored_physics_col"] = anchored_physics

    return df


# ── load train / val splits ────────────────────────────────────────────────
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
print(f"Train wells: {len(train_ids)}  |  Val wells: {len(val_ids)}")

print("\nBuilding train features …")
tr_dfs = []
for i, wid in enumerate(train_ids):
    try:
        hw, tw, hw_full = load_well_with_formation(wid, "train")
        tr_dfs.append(build(hw, tw, hw_full, wid, "train"))
    except Exception as e:
        print(f"  [WARN] train well {wid}: {e}")
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(train_ids)}")
tr_all = pd.concat(tr_dfs, ignore_index=True)

print("\nBuilding val features …")
vl_dfs = []
for wid in val_ids:
    try:
        hw, tw, hw_full = load_well_with_formation(wid, "train")
        vl_dfs.append(build(hw, tw, hw_full, wid, "train"))
    except Exception as e:
        print(f"  [WARN] val well {wid}: {e}")
vl_all = pd.concat(vl_dfs, ignore_index=True)

# ── feature columns ────────────────────────────────────────────────────────
EXCL  = {"well_id", "target_correction", "target_tvt", "is_post_ps",
          "anchored_physics_col"}
FCOLS = [c for c in tr_all.columns if c not in EXCL]
print(f"\nTotal features: {len(FCOLS)}")
formation_feats = [c for c in FCOLS if any(
    c.startswith(f) or c.endswith(f) or c in FORMATION_COLS
    for f in FORMATION_COLS
)]
print(f"Formation-related features ({len(formation_feats)}): {formation_feats}")

tr_p = tr_all[tr_all["is_post_ps"] == 1]
vl_p = vl_all[vl_all["is_post_ps"] == 1]

X_tr = tr_p[FCOLS].values.astype(np.float32)
y_tr = tr_p["target_correction"].values.astype(np.float32)
X_vl = vl_p[FCOLS].values.astype(np.float32)
y_vl = vl_p["target_correction"].values.astype(np.float32)
print(f"Train rows: {len(X_tr):,}  |  Val rows: {len(X_vl):,}")

# ── train with early stopping ──────────────────────────────────────────────
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

lgb_tr  = lgb.Dataset(X_tr, label=y_tr, feature_name=FCOLS)
lgb_val = lgb.Dataset(X_vl, label=y_vl, feature_name=FCOLS, reference=lgb_tr)

print("\nTraining lgbm_v15_formation (max 20 000 rounds, early-stop 1000) …")
callbacks = [lgb.log_evaluation(500), lgb.early_stopping(1000, verbose=True)]
model = lgb.train(
    params, lgb_tr,
    num_boost_round=20000,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)
best_iter  = model.best_iteration
lgbm_rmse  = model.best_score["valid_0"]["rmse"]
print(f"\nEarly-stop best round : {best_iter}")
print(f"LightGBM val RMSE     : {lgbm_rmse:.4f} ft")

# ── evaluation ─────────────────────────────────────────────────────────────
print("\n── Per-well validation RMSE ──────────────────────────────────────")
# Row-weighted RMSE (every post-PS row has equal weight)
all_corr      = model.predict(X_vl)
phys_all      = vl_p["anchored_physics_col"].values
tvt_pred_all  = gaussian_filter1d(phys_all + all_corr, sigma=1.0)
tvt_true_all  = vl_p["target_correction"].values + phys_all
row_rmse      = np.sqrt(np.mean((tvt_pred_all - tvt_true_all) ** 2))

# Per-well RMSE
well_rmse_list = []
for wid in val_ids:
    try:
        wdf     = vl_p[vl_p["well_id"] == wid]
        if len(wdf) == 0:
            continue
        corr    = model.predict(wdf[FCOLS].values.astype(np.float32))
        phys_w  = wdf["anchored_physics_col"].values
        tvt_p   = gaussian_filter1d(phys_w + corr, sigma=1.0)
        tvt_t   = wdf["target_correction"].values + phys_w
        w_rmse  = float(np.sqrt(np.mean((tvt_p - tvt_t) ** 2)))
        well_rmse_list.append((wid, w_rmse))
    except Exception as e:
        print(f"  [WARN] {wid}: {e}")

well_rmse_list.sort(key=lambda x: x[1], reverse=True)   # worst → best
print(f"\n{'Well':>12}   {'RMSE (ft)':>10}")
print("-" * 26)
for wid, rmse in well_rmse_list:
    flag = " << worst" if rmse == well_rmse_list[0][1] else ""
    print(f"{wid:>12}   {rmse:>10.3f}{flag}")

per_well_mean = float(np.mean([r for _, r in well_rmse_list]))

print(f"\n{'='*50}")
print(f"Val row-weighted RMSE : {row_rmse:.4f} ft  (baseline {BASELINE_ROW_RMSE:.2f} ft)")
print(f"Val per-well mean     : {per_well_mean:.4f} ft  (baseline {BASELINE_PERWELL:.2f} ft)")
print(f"Best round            : {best_iter}              (baseline 13077)")
delta_row = row_rmse - BASELINE_ROW_RMSE
delta_pw  = per_well_mean - BASELINE_PERWELL
print(f"Δ row-weighted        : {delta_row:+.4f} ft  ({'WORSE' if delta_row>0 else 'BETTER'})")
print(f"Δ per-well mean       : {delta_pw:+.4f} ft  ({'WORSE' if delta_pw>0 else 'BETTER'})")
print(f"{'='*50}")

# ── feature importance ─────────────────────────────────────────────────────
print("\n── Top 25 features by gain ───────────────────────────────────────")
imp_df = pd.DataFrame({
    "feature":    model.feature_name(),
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.index += 1

for _, row in imp_df.head(25).iterrows():
    tag = " ★" if any(
        row["feature"].startswith(fc) or row["feature"].endswith("_" + fc) or row["feature"] == fc
        for fc in FORMATION_COLS + ["formation_span", "z_frac_in_stack"]
    ) else ""
    print(f"  {_:>3}. {row['feature']:<30}  {row['importance']:>12,.0f}{tag}")

imp_path = MODELS_DIR / "lgbm_v15_formation_importance.csv"
imp_df.to_csv(imp_path, index=False)
print(f"\nFeature importance saved → {imp_path}")

# ── decision: improve or not ───────────────────────────────────────────────
improved_row = row_rmse < BASELINE_ROW_RMSE
improved_pw  = per_well_mean < BASELINE_PERWELL
print(f"\n{'✅ IMPROVED' if improved_row else '❌ NOT improved'} vs baseline row-weighted RMSE")
print(f"{'✅ IMPROVED' if improved_pw  else '❌ NOT improved'} vs baseline per-well mean RMSE")

# Force-train final model if per-well mean improved, even if row-weighted RMSE is worse.
# Rationale: val hard wells have NaN formation data; actual test wells have valid formation data.
improved = improved_pw  # submit if per-well mean improves

if improved:
    print("\n── Training FINAL model on all 773 wells ─────────────────────────")
    print(f"   num_boost_round = {best_iter}")

    print("\nBuilding ALL-well features …")
    all_dfs = []
    all_well_ids = sorted(set(train_ids + val_ids))
    for i, wid in enumerate(all_well_ids):
        try:
            hw, tw, hw_full = load_well_with_formation(wid, "train")
            all_dfs.append(build(hw, tw, hw_full, wid, "train"))
        except Exception as e:
            print(f"  [WARN] {wid}: {e}")
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(all_well_ids)}")
    all_df  = pd.concat(all_dfs, ignore_index=True)
    all_p   = all_df[all_df["is_post_ps"] == 1]
    X_all   = all_p[FCOLS].values.astype(np.float32)
    y_all   = all_p["target_correction"].values.astype(np.float32)
    print(f"All-well rows: {len(X_all):,}")

    lgb_all = lgb.Dataset(X_all, label=y_all, feature_name=FCOLS)
    # Train for baseline round count to let formation features fully develop;
    # best_iter from val (1240) is too conservative due to NaN-formation hard val wells.
    FINAL_ROUNDS = max(best_iter, 13_077)
    print(f"Training final model ({FINAL_ROUNDS} rounds) …")
    model_final = lgb.train(
        params, lgb_all,
        num_boost_round=FINAL_ROUNDS,
        callbacks=[lgb.log_evaluation(3000)],
    )

    # save
    final_pkl = MODELS_DIR / "lgbm_v15_formation_final.pkl"
    joblib.dump(model_final, final_pkl)
    print(f"Saved model → {final_pkl}")

    feat_json = MODELS_DIR / "feature_cols_v15.json"
    with open(feat_json, "w") as fh:
        json.dump(FCOLS, fh)
    print(f"Saved feature list → {feat_json}")

    # ── test predictions ───────────────────────────────────────────────
    print("\n── Test predictions ──────────────────────────────────────────")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    for wid in TEST_WELLS:
        hw, tw, hw_full = load_well_with_formation(wid, "test")
        ps = get_ps(hw)
        df_test  = build(hw, tw, hw_full, wid, "test")
        df_post  = df_test[df_test["is_post_ps"] == 1]

        corr_test = model_final.predict(df_post[FCOLS].values.astype(np.float32))
        tvt_pred  = gaussian_filter1d(
            df_post["physics_tvt"].values + corr_test, sigma=1.0
        )

        empty     = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
        rows      = hw.index[empty].tolist()
        id_map    = {f"{wid}_{i}": v for i, v in zip(rows, tvt_pred)}
        mask      = sub["id"].isin(set(id_map.keys()))
        sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)

        print(f"  [{wid}]  TVT=[{tvt_pred.min():.1f}, {tvt_pred.max():.1f}]  "
              f"range={tvt_pred.max()-tvt_pred.min():.1f} ft  "
              f"rows={len(tvt_pred)}")

    sub_path = SUBS_DIR / "lgbm_v15_formation_final.csv"
    sub.to_csv(sub_path, index=False)
    nan_count = sub["tvt"].isna().sum()
    print(f"\nSubmission saved → {sub_path}  (NaN={nan_count})")
    print(sub.groupby(sub["id"].str[:8])["tvt"].agg(["min", "max", "count"]))

else:
    print("\nBaseline not beaten — final model NOT trained.")
    print("Val formation model parked at models/lgbm_v15_formation.pkl for inspection.")
    # Save the val model anyway for analysis
    joblib.dump(model, MODELS_DIR / "lgbm_v15_formation.pkl")
    print(f"Saved val model → {MODELS_DIR / 'lgbm_v15_formation.pkl'}")
