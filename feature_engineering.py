"""
Feature Engineering for ROGII Wellbore Geology Prediction
==========================================================
Builds per-row feature matrices for all train/val/test wells.
Outputs: features/train_features.parquet, val_features.parquet, test_features.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
FEAT_DIR  = DATA_DIR / "features"
FEAT_DIR.mkdir(exist_ok=True)

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── Helpers ────────────────────────────────────────────────────────────────
def get_all_train_ids():
    return sorted(set(
        f.stem.split("__")[0]
        for f in TRAIN_DIR.glob("*__horizontal_well.csv")
    ))

def load_well(well_id, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(d / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{well_id}__typewell.csv")
    return hw, tw

def get_ps_index(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def fill_gr(series):
    return series.ffill().bfill().astype(float).values

# ── Typewell GR matching ───────────────────────────────────────────────────
def build_typewell_lookup(tw_tvt, tw_gr):
    """Return function: estimated_tvt → (matched_tvt, matched_gr, corr_score)"""
    tw_gr_filled = pd.Series(tw_gr).ffill().bfill().values.astype(float)
    tw_tvt_arr   = tw_tvt.astype(float).values if hasattr(tw_tvt, 'values') else np.array(tw_tvt, float)

    def lookup(est_tvt, local_gr_window, hw_window=25, tw_search=60):
        """
        For a given estimated TVT, find the typewell TVT position that best
        correlates with the local horizontal GR window.
        """
        tw_center = np.searchsorted(tw_tvt_arr, est_tvt)
        tw_lo = max(0, tw_center - tw_search)
        tw_hi = min(len(tw_gr_filled), tw_center + tw_search)
        tw_seg = tw_gr_filled[tw_lo:tw_hi]

        # Best-match cross-correlation
        if len(local_gr_window) < 3 or len(tw_seg) < 3:
            return est_tvt, tw_gr_filled[tw_center] if tw_center < len(tw_gr_filled) else np.nan, 0.0

        lw = local_gr_window - local_gr_window.mean()
        best_corr, best_offset = -1.0, 0
        for offset in range(-min(tw_search, len(tw_seg)-3), min(tw_search, len(tw_seg)-3)):
            idx = tw_center + offset - tw_lo
            seg_start = max(0, idx - len(lw)//2)
            seg_end   = seg_start + len(lw)
            if seg_end > len(tw_seg) or seg_start < 0:
                continue
            seg = tw_seg[seg_start:seg_end] - tw_seg[seg_start:seg_end].mean()
            if seg.std() < 1e-6 or lw.std() < 1e-6:
                continue
            corr = np.corrcoef(lw, seg)[0, 1]
            if corr > best_corr:
                best_corr, best_offset = corr, offset

        best_tw_idx = np.clip(tw_center + best_offset, 0, len(tw_tvt_arr) - 1)
        return float(tw_tvt_arr[best_tw_idx]), float(tw_gr_filled[best_tw_idx]), float(best_corr)

    return lookup, tw_tvt_arr, tw_gr_filled

# ── Main feature builder ───────────────────────────────────────────────────
def build_features(well_id, split="train", is_test=False):
    hw, tw = load_well(well_id, split)
    ps_idx = get_ps_index(hw)
    n      = len(hw)

    # GR arrays
    gr_raw  = fill_gr(hw["GR"])
    gr_s    = pd.Series(gr_raw)

    # Savitzky-Golay smoothed GR (only if enough points)
    try:
        gr_smooth = savgol_filter(gr_raw, window_length=min(21, n if n % 2 == 1 else n-1), polyorder=3)
    except Exception:
        gr_smooth = gaussian_filter1d(gr_raw, sigma=3)

    # MD, XYZ
    md = hw["MD"].astype(float).values
    z  = hw["Z"].astype(float).values
    x  = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values

    # Last known TVT (forward-filled)
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = tvt_inp.ffill().values

    # Typewell data
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_gr(tw["GR"])

    feat = pd.DataFrame(index=hw.index)
    feat["well_id"] = well_id

    # ── GR features ──
    feat["gr"]        = gr_raw
    feat["gr_smooth"] = gr_smooth
    feat["gr_diff"]   = np.gradient(gr_raw)
    feat["gr_diff2"]  = np.gradient(np.gradient(gr_raw))

    for w in [5, 11, 21, 51, 101]:
        roll = gr_s.rolling(w, center=True, min_periods=1)
        feat[f"gr_mean_{w}"]  = roll.mean().values
        feat[f"gr_std_{w}"]   = roll.std().fillna(0).values
        feat[f"gr_min_{w}"]   = roll.min().values
        feat[f"gr_max_{w}"]   = roll.max().values
        feat[f"gr_range_{w}"] = feat[f"gr_max_{w}"] - feat[f"gr_min_{w}"]

    # GR lag features (only useful in causal direction)
    for lag in [1, 3, 5, 10]:
        feat[f"gr_lag_{lag}"]  = gr_s.shift(lag).bfill().values
        feat[f"gr_lead_{lag}"] = gr_s.shift(-lag).ffill().values

    # ── Depth & trajectory features ──
    feat["md"]           = md
    feat["z"]            = z
    feat["dz_dmd"]       = np.gradient(z, md)
    feat["dx_dmd"]       = np.gradient(x, md)
    feat["dy_dmd"]       = np.gradient(y, md)
    feat["inclination"]  = np.arctan2(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2),
                                       np.abs(np.gradient(z))) * 180 / np.pi

    # ── TVT position features ──
    feat["last_known_tvt"]  = last_known_tvt
    feat["rows_since_ps"]   = np.maximum(0, np.arange(n) - ps_idx)
    feat["rows_before_ps"]  = np.maximum(0, ps_idx - np.arange(n))
    feat["ps_idx"]          = ps_idx

    # Trend of last N known TVT values (estimated dTVT/dMD)
    tvt_known_series = pd.Series(last_known_tvt)
    feat["tvt_trend_5"]  = tvt_known_series.diff(5).fillna(0).values / 5.0
    feat["tvt_trend_20"] = tvt_known_series.diff(20).fillna(0).values / 20.0

    # ── Typewell GR at estimated TVT ──
    # Fast vectorized lookup (nearest neighbor in TVT space)
    for i, est_tvt in enumerate(last_known_tvt):
        pass  # Will do vectorized below

    tw_tvt_arr = tw_tvt
    tw_gr_arr  = pd.Series(tw_gr).ffill().bfill().values

    # Vectorized nearest typewell lookup
    tw_indices = np.searchsorted(tw_tvt_arr, last_known_tvt).clip(0, len(tw_tvt_arr) - 1)
    feat["tw_gr_at_tvt"]  = tw_gr_arr[tw_indices]
    feat["tw_tvt_at_idx"] = tw_tvt_arr[tw_indices]
    feat["gr_tw_diff"]    = gr_raw - tw_gr_arr[tw_indices]

    # Rolling mean of typewell GR at ±10 samples
    tw_gr_at_tvt_smooth = np.array([
        tw_gr_arr[max(0, ti-10):min(len(tw_gr_arr), ti+10)].mean()
        for ti in tw_indices
    ])
    feat["tw_gr_smooth_at_tvt"] = tw_gr_at_tvt_smooth
    feat["gr_tw_smooth_diff"]   = gr_raw - tw_gr_at_tvt_smooth

    # Typewell GR gradient at estimated TVT
    tw_gr_grad = np.gradient(tw_gr_arr, tw_tvt_arr)
    feat["tw_gr_grad_at_tvt"] = tw_gr_grad[tw_indices]

    # ── Normalised GR position relative to typewell range ──
    tw_gr_min, tw_gr_max = tw_gr_arr.min(), tw_gr_arr.max()
    feat["gr_norm_tw"] = (gr_raw - tw_gr_min) / (tw_gr_max - tw_gr_min + 1e-9)

    # ── Target (train only) ──
    if not is_test and split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        feat["target_tvt"]       = tvt_true
        feat["target_dtvt"]      = tvt_true - last_known_tvt
        feat["is_post_ps"]       = (np.arange(n) >= ps_idx).astype(int)

    return feat

# ── Process all wells ──────────────────────────────────────────────────────
def process_split(well_ids, split="train", is_test=False, label=""):
    dfs = []
    total = len(well_ids)
    for i, wid in enumerate(well_ids):
        try:
            df = build_features(wid, split, is_test)
            dfs.append(df)
            if (i + 1) % 50 == 0 or i == total - 1:
                print(f"  [{label}] {i+1}/{total} wells processed...")
        except Exception as e:
            print(f"  WARNING: skipping {wid} — {e}")
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Engineering — ROGII Wellbore Geology Prediction")
    print("=" * 60)

    all_ids = get_all_train_ids()
    print(f"Total training wells: {len(all_ids)}")

    # Well-level train/val split (15% validation)
    np.random.shuffle(all_ids)
    n_val    = max(1, int(len(all_ids) * 0.15))
    val_ids  = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Train wells: {len(train_ids)}, Val wells: {len(val_ids)}")

    # Save split IDs for reproducibility
    pd.Series(train_ids).to_csv(DATA_DIR / "features" / "train_ids.csv", index=False, header=False)
    pd.Series(val_ids).to_csv(DATA_DIR / "features" / "val_ids.csv", index=False, header=False)

    # Build train features
    print("\nBuilding train features...")
    train_df = process_split(train_ids, "train", False, "TRAIN")
    print(f"Train features shape: {train_df.shape}")
    train_df.to_parquet(FEAT_DIR / "train_features.parquet", index=False)
    print(f"Saved: {FEAT_DIR / 'train_features.parquet'}")

    # Build val features
    print("\nBuilding val features...")
    val_df = process_split(val_ids, "train", False, "VAL")
    print(f"Val features shape: {val_df.shape}")
    val_df.to_parquet(FEAT_DIR / "val_features.parquet", index=False)
    print(f"Saved: {FEAT_DIR / 'val_features.parquet'}")

    # Build test features (post-PS only for submission)
    print("\nBuilding test features...")
    test_df = process_split(TEST_WELLS, "test", True, "TEST")
    print(f"Test features shape: {test_df.shape}")
    test_df.to_parquet(FEAT_DIR / "test_features.parquet", index=False)
    print(f"Saved: {FEAT_DIR / 'test_features.parquet'}")

    print("\nFeature columns:", [c for c in train_df.columns if c not in ["well_id"]])
    print("\nDone!")
