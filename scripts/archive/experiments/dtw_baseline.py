"""
DTW Baseline for ROGII Wellbore Geology Prediction
===================================================
Uses Dynamic Time Warping to align horizontal well GR with typewell GR,
then extracts TVT predictions for post-PS rows.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dtaidistance import dtw
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TEST_DIR  = DATA_DIR / "test"
SUBS_DIR  = DATA_DIR / "submissions"
SUBS_DIR.mkdir(exist_ok=True)

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── Helpers ────────────────────────────────────────────────────────────────
def load_well(well_id: str, split: str = "test"):
    d = DATA_DIR / split
    hw = pd.read_csv(d / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(d / f"{well_id}__typewell.csv")
    return hw, tw

def get_ps_index(hw: pd.DataFrame) -> int:
    """First row index where TVT_input is missing (Prediction Start)."""
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def fill_gr(series: pd.Series) -> np.ndarray:
    """Fill NaN GR with forward-fill then back-fill, return float array."""
    return series.ffill().bfill().astype(float).values

def smooth_gr(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Light Gaussian smoothing to reduce noise before DTW."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma)

# ── Core DTW Prediction ────────────────────────────────────────────────────
def predict_tvt_dtw(well_id: str):
    """
    Strategy:
    1. Anchor: use last ~200 pre-PS horizontal GR samples (higher resolution)
    2. Query:  anchor + all post-PS GR samples
    3. DTW-align query against typewell GR window centred on last known TVT
    4. Map DTW path → typewell TVT values → predicted TVT
    """
    hw, tw = load_well(well_id, "test")
    ps_idx = get_ps_index(hw)
    print(f"\n[{well_id}] PS index={ps_idx}, post-PS rows={len(hw)-ps_idx}")

    # ── Horizontal GR ──
    hw_gr_raw = fill_gr(hw["GR"])
    hw_gr     = smooth_gr(hw_gr_raw, sigma=1.5)

    # Last known TVT (at PS-1)
    known_tvt_series = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_known_tvt = float(known_tvt_series.ffill().iloc[ps_idx - 1])
    print(f"  Last known TVT: {last_known_tvt:.2f} ft")

    # ── Typewell ──
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = fill_gr(tw["GR"])
    tw_gr  = smooth_gr(tw_gr, sigma=1.0)

    # ── Build query: anchor (pre-PS tail) + post-PS ──
    ANCHOR_LEN  = min(300, ps_idx)
    anchor_gr   = hw_gr[ps_idx - ANCHOR_LEN : ps_idx]
    post_ps_gr  = hw_gr[ps_idx:]
    query_gr    = np.concatenate([anchor_gr, post_ps_gr])
    n_post      = len(post_ps_gr)

    # ── Typewell window ──
    # Find anchor start in typewell by TVT
    anchor_tvt_start = last_known_tvt - (ANCHOR_LEN * 0.5)  # ~0.5 ft/sample
    tw_start_idx = max(0, np.searchsorted(tw_tvt, anchor_tvt_start) - 20)
    # Search forward up to 3000 typewell samples beyond last known TVT
    tw_end_idx   = min(len(tw_tvt), np.searchsorted(tw_tvt, last_known_tvt) + 3000)
    tw_window_gr  = tw_gr[tw_start_idx : tw_end_idx]
    tw_window_tvt = tw_tvt[tw_start_idx : tw_end_idx]
    print(f"  Typewell window: {tw_window_tvt[0]:.1f}–{tw_window_tvt[-1]:.1f} ft ({len(tw_window_gr)} samples)")
    print(f"  Query length: anchor={ANCHOR_LEN} + post={n_post} = {len(query_gr)} samples")

    # ── Normalise GR to [0,1] ──
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    q_norm  = norm(query_gr).astype(np.double)
    tw_norm = norm(tw_window_gr).astype(np.double)

    # ── DTW with Sakoe-Chiba band ──
    # window = 15% of max length to avoid extreme warping
    window = max(50, int(max(len(q_norm), len(tw_norm)) * 0.15))
    print(f"  Running DTW (window={window})...", end="", flush=True)
    path = dtw.warping_path(q_norm, tw_norm, window=window)
    print(f" done. Path length={len(path)}")

    # ── Map path: for each query index → mean mapped typewell index ──
    qi_to_ti = {}
    for qi, ti in path:
        qi_to_ti.setdefault(qi, []).append(ti)

    # Resolve: mean typewell index per query position
    query_to_tvt = {}
    for qi, tis in qi_to_ti.items():
        ti_mean = int(round(np.mean(tis)))
        ti_mean = np.clip(ti_mean, 0, len(tw_window_tvt) - 1)
        query_to_tvt[qi] = tw_window_tvt[ti_mean]

    # Interpolate any missing positions (shouldn't happen but safety net)
    all_q_tvt = np.array([
        query_to_tvt.get(i, np.nan) for i in range(len(query_gr))
    ])
    # Linear interpolate NaNs
    nans = np.isnan(all_q_tvt)
    if nans.any():
        idx = np.arange(len(all_q_tvt))
        all_q_tvt[nans] = np.interp(idx[nans], idx[~nans], all_q_tvt[~nans])

    # ── Extract post-PS predictions (skip anchor portion) ──
    post_tvt_pred = all_q_tvt[ANCHOR_LEN:]
    assert len(post_tvt_pred) == n_post, f"Length mismatch {len(post_tvt_pred)} vs {n_post}"

    # ── Light smoothing on output ──
    from scipy.ndimage import gaussian_filter1d
    post_tvt_pred = gaussian_filter1d(post_tvt_pred, sigma=2.0)

    print(f"  Predicted TVT range: [{post_tvt_pred.min():.2f}, {post_tvt_pred.max():.2f}] ft")
    print(f"  Typewell TVT range:  [{tw_tvt.min():.2f}, {tw_tvt.max():.2f}] ft")

    return post_tvt_pred

# ── Build Submission ───────────────────────────────────────────────────────
def make_submission(predictions: dict, name: str) -> Path:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"\nBuilding submission '{name}' ({len(sub)} rows)...")

    for well_id, pred_tvt in predictions.items():
        hw = pd.read_csv(TEST_DIR / f"{well_id}__horizontal_well.csv")
        empty_mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
        row_indices = hw.index[empty_mask].tolist()
        assert len(row_indices) == len(pred_tvt), \
            f"[{well_id}] {len(row_indices)} indices vs {len(pred_tvt)} predictions"
        ids = [f"{well_id}_{i}" for i in row_indices]
        id_to_tvt = dict(zip(ids, pred_tvt))
        mask = sub["id"].isin(set(ids))
        sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_to_tvt)

    nan_count = sub["tvt"].isna().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in submission!")
    else:
        print(f"  All {len(sub)} rows filled successfully.")

    out = SUBS_DIR / f"{name}.csv"
    sub.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return out

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DTW Baseline — ROGII Wellbore Geology Prediction")
    print("=" * 60)

    predictions = {}
    for well_id in TEST_WELLS:
        try:
            pred = predict_tvt_dtw(well_id)
            predictions[well_id] = pred
        except Exception as e:
            print(f"  ERROR for {well_id}: {e}")
            raise

    sub_path = make_submission(predictions, "dtw_baseline")
    print(f"\nDone! Submission: {sub_path}")

    # Quick sanity check
    sub = pd.read_csv(sub_path)
    print(f"\nSubmission stats:")
    print(f"  Rows: {len(sub)}")
    print(f"  TVT range: [{sub['tvt'].min():.2f}, {sub['tvt'].max():.2f}]")
    print(f"  NaN count: {sub['tvt'].isna().sum()}")
