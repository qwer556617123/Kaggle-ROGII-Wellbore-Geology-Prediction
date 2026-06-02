"""
Feature Engineering v2 — kNN Correction Transfer + Extended Typewell Features
==============================================================================
Adds kNN-based correction transfer features from similar training wells.

Outputs:
  features/train_features_v2.parquet   (all 773 training wells, post-PS rows)
  features/test_features_v2.parquet    (3 test wells, post-PS rows)
  features/well_signatures.parquet     (per-well signature vectors)
  features/knn_correction_profiles.parquet (per-well correction profiles)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
FEAT_DIR  = DATA_DIR / "features"
FEAT_DIR.mkdir(exist_ok=True)

TEST_WELLS    = ["000d7d20", "00bbac68", "00e12e8b"]
K_NEIGHBORS   = 10
N_BINS        = 20
BIN_FRACS     = np.linspace(0.0, 1.0, N_BINS)

# ══════════════════════════════════════════════════════════════════════════════
# EXACT build() function copied from lgbm_final_reg_train.py
# ══════════════════════════════════════════════════════════════════════════════

def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values

def load_well(wid, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (pd.read_csv(d / f"{wid}__horizontal_well.csv"),
            pd.read_csv(d / f"{wid}__typewell.csv"))

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def build(hw, tw, well_id, split="train"):
    n = len(hw); ps = get_ps(hw)
    gr = fill_arr(hw["GR"]); md = hw["MD"].astype(float).values
    z  = hw["Z"].astype(float).values; x = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt = tvt_inp.ffill().bfill().values
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])
    gr_s   = pd.Series(gr)
    if split == "train" and "TVT" in hw.columns:
        pre_z   = z[:ps]; pre_tvt = hw["TVT"].astype(float).values[:ps]
    else:
        known  = tvt_inp[:ps].values; valid = ~np.isnan(known)
        pre_z  = z[:ps][valid]; pre_tvt = known[valid]
    if len(pre_z) > 5:
        reg = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0]); pre_tvt_pred = reg.predict(pre_z.reshape(-1, 1))
        pre_r2 = float(1 - np.var(pre_tvt - pre_tvt_pred) / np.var(pre_tvt)) if np.var(pre_tvt) > 0 else 1.
    else:
        slope = -1.0; pre_r2 = 0.
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps - 1)]); Z_anchor = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)
    post_z  = z[ps:] if ps < n else z[:]
    post_md = md[ps:] if ps < n else md[:]
    n_post  = len(post_z); z_end = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z = z_end - Z_anchor
    total_md = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.
    post_dz_rate = total_z / total_md; phys_tvt_end = slope * total_z
    pre_dz_md = ((z[ps - 1] - z[0]) / (md[ps - 1] - md[0])) if ps > 1 and abs(md[ps - 1] - md[0]) > 0.01 else -0.01
    dz_rate_change = post_dz_rate - pre_dz_md
    gr_at_physics = np.interp(anchored_physics, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr - gr_at_physics
    feat = {"gr": gr, "gr_diff": np.gradient(gr), "gr_diff2": np.gradient(np.gradient(gr))}
    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_range_{w}"] = (roll.max() - roll.min()).values
    feat.update({
        "md": md, "z": z, "dz_dmd": np.gradient(z, md),
        "dx_dmd": np.gradient(x, md), "dy_dmd": np.gradient(y, md),
        "inclination": np.arctan2(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
                                  np.abs(np.gradient(z)) + 1e-9) * 180 / np.pi,
        "raw_dz": np.diff(z, prepend=z[0]), "raw_dmd": np.diff(md, prepend=md[0]),
        "last_known_tvt": lkt, "rows_since_ps": np.maximum(0, np.arange(n) - ps).astype(float),
        "rows_before_ps": np.maximum(0, ps - np.arange(n)).astype(float), "ps_idx": float(ps),
        "tvt_z_slope": np.full(n, slope), "physics_tvt": anchored_physics,
        "physics_vs_lkt": anchored_physics - lkt,
        "physics_dtvt": slope * np.diff(z, prepend=z[0]),
        "gr_at_physics": gr_at_physics, "gr_dev_physics": gr_dev,
        "total_z_change_post": np.full(n, total_z), "physics_tvt_at_end": np.full(n, phys_tvt_end),
        "post_dz_rate": np.full(n, post_dz_rate), "dz_rate_change": np.full(n, dz_rate_change),
        "n_post_ps": np.full(n, float(n_post)),
        "post_ps_frac": np.where(n_post > 0, np.maximum(0, np.arange(n) - ps) / n_post, 0.).astype(float),
        "pre_r2": np.full(n, pre_r2),
    })
    if ps > 2:
        pre_dz = np.gradient(z[:ps], md[:ps]) if ps > 1 else np.array([0.])
        vtvt = tvt_inp[:ps].values; vtvt = vtvt[~np.isnan(vtvt)]
        pdtvt = np.diff(vtvt) if len(vtvt) > 1 else np.array([0.])
        feat["pre_ps_dz_slope"]  = np.full(n, pre_dz[-min(50, len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"] = np.full(n, pdtvt[-min(50, len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]  = np.full(n, (pdtvt[-min(50, len(pdtvt)):].std() if len(pdtvt) > 1 else 0.))
    else:
        for k in ["pre_ps_dz_slope", "pre_ps_dtvt_mean", "pre_ps_dtvt_std"]:
            feat[k] = np.zeros(n)
    df = pd.DataFrame(feat); df["well_id"] = well_id
    df["is_post_ps"] = (np.arange(n) >= ps).astype(int)
    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_correction"]    = tvt_true - anchored_physics
        df["anchored_physics_col"] = anchored_physics
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Extended Typewell GR Features
# ══════════════════════════════════════════════════════════════════════════════

def add_typewell_extended_features(df: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Add typewell gradient + additional GR-typewell alignment features."""
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_arr(tw["GR"])

    # Gradient of typewell GR w.r.t. TVT
    tw_gr_grad = np.gradient(tw_gr, tw_tvt)

    # Physics TVT (estimated position along typewell)
    phys_tvt = df["physics_tvt"].values

    # Vectorized: typewell GR gradient at physics TVT position
    tw_grad_at_pos = np.interp(phys_tvt, tw_tvt, tw_gr_grad,
                               left=tw_gr_grad[0], right=tw_gr_grad[-1])
    df = df.copy()
    df["tw_gradient_at_pos"] = tw_grad_at_pos

    # gr_tw_at_physics — alias for clarity (same as gr_at_physics in build())
    # already present as gr_at_physics; add named alias
    df["gr_tw_at_physics"] = df["gr_at_physics"].values

    # gr_dev_from_tw — alias for gr_dev_physics
    df["gr_dev_from_tw"] = df["gr_dev_physics"].values

    # Typewell GR second derivative (curvature) at estimated position
    tw_gr_grad2 = np.gradient(tw_gr_grad, tw_tvt)
    df["tw_curvature_at_pos"] = np.interp(phys_tvt, tw_tvt, tw_gr_grad2,
                                           left=tw_gr_grad2[0], right=tw_gr_grad2[-1])

    # Local typewell GR range ±25 samples around estimated position
    tw_idx = np.searchsorted(tw_tvt, phys_tvt).clip(0, len(tw_tvt) - 1)
    tw_local_range = np.array([
        float(np.ptp(tw_gr[max(0, i - 25):min(len(tw_gr), i + 25)]))
        for i in tw_idx
    ])
    df["tw_local_gr_range"] = tw_local_range

    return df

# ══════════════════════════════════════════════════════════════════════════════
# Well Signature
# ══════════════════════════════════════════════════════════════════════════════

def compute_well_signature(df_well: pd.DataFrame, is_test: bool = False) -> dict:
    """
    Compute a feature vector to identify similar wells.
    Uses only pre-PS features plus physics/trajectory features available at PS.
    """
    pre = df_well[df_well["is_post_ps"] == 0]

    # Scalar constants from build() (same for all rows)
    tvt_z_slope      = float(df_well["tvt_z_slope"].iloc[0])
    pre_r2           = float(df_well["pre_r2"].iloc[0])
    pre_ps_dz_slope  = float(df_well["pre_ps_dz_slope"].iloc[0])
    pre_ps_dtvt_mean = float(df_well["pre_ps_dtvt_mean"].iloc[0])

    # Anchor TVT at PS
    anchor_tvt = float(pre["last_known_tvt"].iloc[-1]) if len(pre) > 0 else 0.0

    # Pre-PS GR statistics (last 50 rows before PS)
    pre_gr = pre["gr"].values[-50:]
    if len(pre_gr) > 0:
        valid_gr = pre_gr[~np.isnan(pre_gr)]
    else:
        valid_gr = np.array([0.0])
    if len(valid_gr) == 0:
        valid_gr = np.array([0.0])

    gr_mean_pre = float(np.mean(valid_gr))
    gr_std_pre  = float(np.std(valid_gr))
    pcts = np.nanpercentile(valid_gr, [10, 25, 50, 75, 90])
    gr_p10, gr_p25, gr_p50, gr_p75, gr_p90 = (float(p) for p in pcts)

    # Inclination near PS (last 20 rows)
    pre_incl = pre["inclination"].values[-20:]
    incl_near_ps = float(np.nanmean(pre_incl)) if len(pre_incl) > 0 else 0.0

    # Total Z change post-PS (training: actual; test: 0 — not available)
    if not is_test:
        total_z_post = float(df_well["total_z_change_post"].iloc[0])
    else:
        total_z_post = 0.0

    return {
        "anchor_tvt":        anchor_tvt,
        "tvt_z_slope":       tvt_z_slope,
        "pre_r2":            pre_r2,
        "pre_ps_dz_slope":   pre_ps_dz_slope,
        "pre_ps_dtvt_mean":  pre_ps_dtvt_mean,
        "gr_mean_pre":       gr_mean_pre,
        "gr_std_pre":        gr_std_pre,
        "gr_p10_pre":        gr_p10,
        "gr_p25_pre":        gr_p25,
        "gr_p50_pre":        gr_p50,
        "gr_p75_pre":        gr_p75,
        "gr_p90_pre":        gr_p90,
        "inclination_near_ps": incl_near_ps,
        "total_z_post":      total_z_post,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Correction Profile
# ══════════════════════════════════════════════════════════════════════════════

def get_correction_profile(df_well: pd.DataFrame, n_bins: int = N_BINS) -> np.ndarray:
    """
    For a training well, interpolate target_correction at n_bins equally-spaced
    fractions of post-PS progress (0.0 … 1.0).
    Returns array of shape (n_bins,); NaN-filled if insufficient data.
    """
    post = df_well[df_well["is_post_ps"] == 1]
    if len(post) < 3 or "target_correction" not in post.columns:
        return np.full(n_bins, np.nan)

    fracs = post["post_ps_frac"].values
    corrs = post["target_correction"].values

    # Remove NaN corrections
    valid = ~np.isnan(corrs) & ~np.isnan(fracs)
    if valid.sum() < 2:
        return np.full(n_bins, np.nan)

    fracs_v = fracs[valid]
    corrs_v = corrs[valid]

    # Sort by fraction
    order = np.argsort(fracs_v)
    fracs_v, corrs_v = fracs_v[order], corrs_v[order]

    return np.interp(BIN_FRACS, fracs_v, corrs_v)

# ══════════════════════════════════════════════════════════════════════════════
# kNN Feature Addition
# ══════════════════════════════════════════════════════════════════════════════

def add_knn_features(
    df_well: pd.DataFrame,
    neighbor_indices: np.ndarray,   # (K,) indices into train_well_ids
    neighbor_distances: np.ndarray, # (K,) L2 distances
    profiles_matrix: np.ndarray,    # (n_train_wells, N_BINS)
    train_signatures_df: pd.DataFrame,  # (n_train_wells, sig_cols)
) -> pd.DataFrame:
    """
    Append kNN correction transfer features to df_well.
    Efficient: vectorised per-neighbor interpolation.
    """
    df = df_well.copy()
    n_rows = len(df)
    fracs  = df["post_ps_frac"].values  # (n_rows,)

    K = len(neighbor_indices)

    # Gather neighbor correction profiles: (K, N_BINS)
    nb_profiles = profiles_matrix[neighbor_indices]  # (K, N_BINS)

    # For each neighbor, interpolate correction at every row's frac → (K, n_rows)
    interp_corrs = np.full((K, n_rows), np.nan)
    for k in range(K):
        profile = nb_profiles[k]
        valid_mask = ~np.isnan(profile)
        if valid_mask.sum() >= 2:
            interp_corrs[k] = np.interp(
                fracs, BIN_FRACS[valid_mask], profile[valid_mask]
            )

    # n_valid per row
    n_valid = (~np.isnan(interp_corrs)).sum(axis=0).astype(float)  # (n_rows,)

    # Aggregate statistics (nanfunctions to handle any remaining NaNs)
    knn_mean  = np.nanmean(interp_corrs, axis=0)
    knn_std   = np.nanstd(interp_corrs, axis=0)
    knn_med   = np.nanmedian(interp_corrs, axis=0)
    knn_min   = np.nanmin(interp_corrs, axis=0)
    knn_max   = np.nanmax(interp_corrs, axis=0)

    # Distance-weighted mean: weight = 1 / (dist + eps)
    weights = 1.0 / (neighbor_distances + 1e-9)  # (K,)
    # Expand for broadcasting: (K, 1) * (K, n_rows) → weighted
    w_expanded = weights[:, np.newaxis] * (~np.isnan(interp_corrs)).astype(float)
    w_sum = w_expanded.sum(axis=0)  # (n_rows,)
    interp_no_nan = np.where(np.isnan(interp_corrs), 0.0, interp_corrs)
    knn_wt_mean = np.where(w_sum > 0,
                            (w_expanded * interp_no_nan).sum(axis=0) / w_sum,
                            knn_mean)

    # Handle edge case: all NaN → fill with 0
    knn_mean  = np.nan_to_num(knn_mean,  nan=0.0)
    knn_std   = np.nan_to_num(knn_std,   nan=0.0)
    knn_med   = np.nan_to_num(knn_med,   nan=0.0)
    knn_min   = np.nan_to_num(knn_min,   nan=0.0)
    knn_max   = np.nan_to_num(knn_max,   nan=0.0)
    knn_wt_mean = np.nan_to_num(knn_wt_mean, nan=0.0)

    df["knn_mean_corr"]    = knn_mean
    df["knn_std_corr"]     = knn_std
    df["knn_med_corr"]     = knn_med
    df["knn_min_corr"]     = knn_min
    df["knn_max_corr"]     = knn_max
    df["knn_wt_mean_corr"] = knn_wt_mean
    df["knn_n_valid"]      = n_valid

    # Neighbor slope statistics
    nb_slopes = train_signatures_df.iloc[neighbor_indices]["tvt_z_slope"].values
    knn_neighbor_slope = float(np.mean(nb_slopes))
    this_slope = float(df["tvt_z_slope"].iloc[0])

    df["knn_neighbor_slope"] = knn_neighbor_slope
    df["knn_slope_diff"]     = this_slope - knn_neighbor_slope

    return df

# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Feature Engineering v2 — kNN Correction Transfer")
    print("=" * 70)

    # ── 1. Discover all training wells ────────────────────────────────────
    all_well_dirs = list(TRAIN_DIR.glob("*__horizontal_well.csv"))
    all_train_ids = sorted([p.name.split("__")[0] for p in all_well_dirs])
    total_train   = len(all_train_ids)
    print(f"\nTraining wells found: {total_train}")

    # ── 2. Build all training well DataFrames ─────────────────────────────
    print("\n[Step 1/6] Building training well DataFrames with build()...")
    train_dfs = {}   # {well_id: df}
    failed    = []
    for i, wid in enumerate(all_train_ids):
        print(f"  Processing {wid} ({i+1}/{total_train})", end="\r")
        try:
            hw, tw = load_well(wid, "train")
            df = build(hw, tw, wid, "train")
            df = add_typewell_extended_features(df, tw)
            train_dfs[wid] = df
        except Exception as e:
            print(f"\n  WARNING: skipping {wid} — {e}")
            failed.append(wid)
    print(f"\n  Built {len(train_dfs)} training wells  ({len(failed)} failed)")

    # ── 3. Build test well DataFrames ─────────────────────────────────────
    print("\n[Step 2/6] Building test well DataFrames...")
    test_dfs = {}
    for i, wid in enumerate(TEST_WELLS):
        print(f"  Processing {wid} ({i+1}/{len(TEST_WELLS)})")
        try:
            hw, tw = load_well(wid, "test")
            df = build(hw, tw, wid, "test")
            df = add_typewell_extended_features(df, tw)
            test_dfs[wid] = df
        except Exception as e:
            print(f"  WARNING: skipping test well {wid} — {e}")

    # ── 4. Compute well signatures ─────────────────────────────────────────
    print("\n[Step 3/6] Computing well signatures...")
    train_well_ids   = list(train_dfs.keys())   # ordered list
    n_train_wells    = len(train_well_ids)

    train_signatures = []
    for wid in train_well_ids:
        sig = compute_well_signature(train_dfs[wid], is_test=False)
        sig["well_id"] = wid
        train_signatures.append(sig)
    train_sig_df = pd.DataFrame(train_signatures).set_index("well_id")

    test_signatures = []
    for wid in TEST_WELLS:
        if wid not in test_dfs:
            continue
        sig = compute_well_signature(test_dfs[wid], is_test=True)
        sig["well_id"] = wid
        test_signatures.append(sig)
    test_sig_df = pd.DataFrame(test_signatures).set_index("well_id") if test_signatures else pd.DataFrame()

    # Combine and save signatures
    all_sig_df = pd.concat([train_sig_df, test_sig_df])
    all_sig_df.reset_index().to_parquet(FEAT_DIR / "well_signatures.parquet", index=False)
    print(f"  Saved well_signatures.parquet  shape={all_sig_df.shape}")

    # ── 5. Fit kNN on training signatures ─────────────────────────────────
    print("\n[Step 4/6] Fitting kNN on training signatures (K=10)...")
    sig_cols = [c for c in train_sig_df.columns]
    X_train_sig = train_sig_df[sig_cols].values.astype(np.float32)

    # Handle any NaN in signatures (rare edge wells)
    X_train_sig = np.nan_to_num(X_train_sig, nan=0.0)

    scaler = StandardScaler()
    X_train_sig_scaled = scaler.fit_transform(X_train_sig)

    # Fit on training wells — query K+1 to exclude self for training wells
    knn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="euclidean", n_jobs=-1)
    knn_model.fit(X_train_sig_scaled)

    # Query training wells (exclude self: skip index 0 which is itself)
    train_distances_raw, train_neighbor_raw = knn_model.kneighbors(X_train_sig_scaled)
    # train_neighbor_raw[:, 0] is the well itself (distance ≈ 0)
    # Take columns 1..K+1 to get K true neighbors
    train_distances_knn  = train_distances_raw[:, 1:K_NEIGHBORS+1]   # (n_train, K)
    train_neighbors_knn  = train_neighbor_raw[:, 1:K_NEIGHBORS+1]    # (n_train, K)

    # Query test wells
    if len(test_signatures) > 0:
        test_well_ids_valid = [s["well_id"] for s in test_signatures]
        X_test_sig = test_sig_df.loc[test_well_ids_valid, sig_cols].values.astype(np.float32)
        X_test_sig = np.nan_to_num(X_test_sig, nan=0.0)
        X_test_sig_scaled = scaler.transform(X_test_sig)
        test_distances_raw, test_neighbors_raw = knn_model.kneighbors(X_test_sig_scaled)
        test_distances_knn = test_distances_raw[:, :K_NEIGHBORS]    # (n_test, K)
        test_neighbors_knn = test_neighbors_raw[:, :K_NEIGHBORS]    # (n_test, K)
    else:
        test_well_ids_valid = []

    print(f"  kNN model fitted on {n_train_wells} wells, K={K_NEIGHBORS}")

    # ── 6. Build correction profiles for all training wells ───────────────
    print("\n[Step 5/6] Building correction profiles for training wells...")
    profiles_list = []
    for wid in train_well_ids:
        profile = get_correction_profile(train_dfs[wid], N_BINS)
        profiles_list.append(profile)
    profiles_matrix = np.stack(profiles_list, axis=0)  # (n_train_wells, N_BINS)

    # Save correction profiles
    prof_df = pd.DataFrame(
        profiles_matrix,
        columns=[f"bin_{i:02d}" for i in range(N_BINS)]
    )
    prof_df.insert(0, "well_id", train_well_ids)
    prof_df.to_parquet(FEAT_DIR / "knn_correction_profiles.parquet", index=False)
    valid_profiles = (~np.isnan(profiles_matrix).all(axis=1)).sum()
    print(f"  Saved knn_correction_profiles.parquet  "
          f"({valid_profiles}/{n_train_wells} wells have valid profiles)")

    # ── 7. Add kNN features to training wells ─────────────────────────────
    print("\n[Step 6/6] Adding kNN features to all wells...")

    print("  [Train wells]")
    train_post_ps_dfs = []
    for i, wid in enumerate(train_well_ids):
        print(f"    {wid} ({i+1}/{n_train_wells})", end="\r")
        nb_indices = train_neighbors_knn[i]    # (K,)
        nb_dists   = train_distances_knn[i]    # (K,)

        df_knn = add_knn_features(
            train_dfs[wid], nb_indices, nb_dists,
            profiles_matrix, train_sig_df
        )
        # Keep post-PS rows only for output
        post = df_knn[df_knn["is_post_ps"] == 1].copy()
        train_post_ps_dfs.append(post)

    train_out = pd.concat(train_post_ps_dfs, ignore_index=True)
    print(f"\n  Training feature matrix: {train_out.shape}")

    print("  [Test wells]")
    test_post_ps_dfs = []
    for i, wid in enumerate(test_well_ids_valid):
        print(f"    {wid} ({i+1}/{len(test_well_ids_valid)})")
        nb_indices = test_neighbors_knn[i]
        nb_dists   = test_distances_knn[i]

        df_knn = add_knn_features(
            test_dfs[wid], nb_indices, nb_dists,
            profiles_matrix, train_sig_df
        )
        post = df_knn[df_knn["is_post_ps"] == 1].copy()
        test_post_ps_dfs.append(post)

    test_out = pd.concat(test_post_ps_dfs, ignore_index=True) if test_post_ps_dfs else pd.DataFrame()
    print(f"  Test feature matrix: {test_out.shape}")

    # ── 8. Save outputs ────────────────────────────────────────────────────
    train_out.to_parquet(FEAT_DIR / "train_features_v2.parquet", index=False)
    test_out.to_parquet(FEAT_DIR / "test_features_v2.parquet", index=False)
    print(f"\nSaved: {FEAT_DIR / 'train_features_v2.parquet'}")
    print(f"Saved: {FEAT_DIR / 'test_features_v2.parquet'}")

    # ── 9. Summary & Validation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    knn_feature_cols = [
        "knn_mean_corr", "knn_std_corr", "knn_med_corr",
        "knn_min_corr", "knn_max_corr", "knn_wt_mean_corr",
        "knn_n_valid", "knn_neighbor_slope", "knn_slope_diff",
    ]
    tw_feature_cols = [
        "tw_gradient_at_pos", "gr_tw_at_physics",
        "gr_dev_from_tw", "tw_curvature_at_pos", "tw_local_gr_range",
    ]
    new_features = knn_feature_cols + tw_feature_cols

    print(f"\nNew features added ({len(new_features)}):")
    for f in new_features:
        print(f"  {f}")

    print(f"\nOutput shapes:")
    print(f"  train_features_v2.parquet : {train_out.shape}")
    print(f"  test_features_v2.parquet  : {test_out.shape}")
    print(f"  well_signatures.parquet   : {all_sig_df.shape}")
    print(f"  knn_correction_profiles   : {profiles_matrix.shape}")

    # First test well kNN feature preview
    first_test_wid = test_well_ids_valid[0] if test_well_ids_valid else None
    if first_test_wid and not test_out.empty:
        print(f"\nFirst test well ({first_test_wid}) — kNN feature sample (first 5 post-PS rows):")
        sub = test_out[test_out["well_id"] == first_test_wid][knn_feature_cols].head(5)
        print(sub.to_string())

    # Correlation of knn features with target_correction (training set)
    if "target_correction" in train_out.columns:
        print(f"\nCorrelation with target_correction (training post-PS rows):")
        for feat_col in knn_feature_cols:
            if feat_col in train_out.columns:
                valid = train_out[["target_correction", feat_col]].dropna()
                if len(valid) > 10:
                    corr = valid["target_correction"].corr(valid[feat_col])
                    print(f"  {feat_col:25s}: {corr:+.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
