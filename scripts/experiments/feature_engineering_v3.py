"""
feature_engineering_v3.py
=========================
GR-Typewell Matching Features for Wellbore TVT Prediction
==========================================================

This module computes GR-based features that exploit the competition's key insight:
  TVT is defined by GR-typewell correlation (PPT Slide 9).

Existing models (v8, v17) are ~2000x more influenced by trajectory/physics features
than by GR. This module creates high-quality GR-typewell alignment features to close
the gap between our 12.269 ft LB score and 1st place at 6.693 ft.

Feature Groups
--------------
1. Pre-PS GR Calibration
   - Fit linear transform hw_gr = a * tw_gr(TVT_input) + b on pre-PS section
   - Provides well-level gain/offset correction between HW and TW GR scales
   - Features: gr_calib_a, gr_calib_b, gr_calib_r2

2. GR Value Matching (per-row, vectorised O(n×N))
   - For each row: invert calibration, find TW TVT matching calibrated GR
   - Search centred on last_known_tvt (not drifting physics TVT!)
   - Multiple radii [15, 30, 50, 80 ft] × smoothing scales [raw, σ3, σ7]
   - Features: gr_val_tvt_*, gr_val_delta_*

3. Global GR-Typewell Cross-Correlation (per-well constant)
   - Shift entire post-PS GR signal against typewell to find best TVT offset
   - Features: xcorr_global_best_delta, xcorr_global_peak_corr, etc.

4. Local Sliding-Window Cross-Correlation (per-row, vectorised)
   - For each row: take HW GR window, compare to typewell at delta offsets
   - Multiple window sizes [200, 500, 1000] centred on lkt, K=10 stride
   - Features: local_xcorr_tvt_*, local_xcorr_score_*, local_xcorr_delta_*

5. GR Gradient Matching (per-row)
   - Compare d(hw_gr)/d(MD) to d(tw_gr)/d(TVT) at estimated TVT position
   - Features: gr_grad_hw, tw_grad_at_lkt, tw_grad_at_val_tvt, gr_grad_sign_match

6. Formation Boundary Features (per-row)
   - Detect sharp GR changes; distance to nearest boundary
   - Features: gr_boundary_strength, rows_to_prev_bound, rows_to_next_bound

Outputs
-------
  features/gr_xcorr_features_train.pkl   - post-PS rows, train wells
  features/gr_xcorr_features_val.pkl     - post-PS rows, val wells
  features/gr_xcorr_features_test.pkl    - post-PS rows, test wells

Usage
-----
  python feature_engineering_v3.py

Validation
----------
  After running, prints MAE comparison for well 000d7d20:
    gr_val_tvt_lkt_sr30 vs physics_tvt vs last_known_tvt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
import warnings
import pickle
import time

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
FEAT_DIR  = DATA_DIR / "features"
FEAT_DIR.mkdir(exist_ok=True)

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

# ── GR smoothing sigmas and xcorr configs ─────────────────────────────────────
GR_SIGMAS       = [0, 3, 7]              # sigma=0 → no smoothing (raw)
SEARCH_RADII    = [15, 30, 50, 80]       # ft, for GR value matching
LOCAL_WINDOWS   = [200, 500, 1000]       # half-windows (rows) for local xcorr
LOCAL_STRIDE    = 10                     # compute every K rows, interpolate rest
GLOBAL_DELTAS   = np.arange(-120, 121, 2.0)   # ±120 ft in 2 ft steps
LOCAL_DELTAS    = np.arange(-60, 61, 2.0)     # ±60 ft in 2 ft steps
TW_STEP         = 0.5                    # typical typewell TVT spacing (ft)
BOUNDARY_THRESH = 3.0                    # |d(GR)/d(MD)| threshold for boundary

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def fill_arr(series_or_array):
    """Forward-fill then back-fill NaNs; return float numpy array."""
    return pd.Series(np.asarray(series_or_array, dtype=float)).ffill().bfill().astype(float).values


def get_ps(hw: pd.DataFrame) -> int:
    """Return index of first row where TVT_input is NaN / empty (Prediction Start)."""
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def load_well(wid: str, split: str = "train"):
    """Load horizontal-well and typewell CSVs for a given well ID."""
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (
        pd.read_csv(d / f"{wid}__horizontal_well.csv"),
        pd.read_csv(d / f"{wid}__typewell.csv"),
    )


def get_physics_tvt(hw: pd.DataFrame, ps: int, use_tvt_col: bool = True) -> np.ndarray:
    """
    Compute physics-based TVT estimate from trajectory.

    Pre-PS: linear regression of Z → TVT_input (or training TVT column).
    Post-PS: extrapolate with the fitted slope from the anchor point.
    """
    z   = hw["Z"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)

    if use_tvt_col and "TVT" in hw.columns:
        pre_tvt = hw["TVT"].astype(float).values[:ps]
        pre_z   = z[:ps]
    else:
        known  = tvt_inp[:ps].values
        valid  = ~np.isnan(known)
        pre_z  = z[:ps][valid]
        pre_tvt = known[valid]

    if len(pre_z) > 5:
        reg   = LinearRegression().fit(pre_z.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
    else:
        slope = -1.0

    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    z_anchor   = float(z[max(0, ps - 1)])
    return anchor_tvt + slope * (z - z_anchor)


# ══════════════════════════════════════════════════════════════════════════════
# Core Feature Functions
# ══════════════════════════════════════════════════════════════════════════════

def _pre_ps_calibration(hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                         ps: int, tvt_inp: np.ndarray) -> tuple:
    """
    Fit linear transform: hw_gr[pre-PS] = a * tw_gr(TVT_input) + b.

    Uses TVT_input (available in both train & test) for consistency.
    Returns (a, b, r2_calibration).
    """
    known = tvt_inp[:ps]
    valid = ~np.isnan(known)
    if valid.sum() < 5:
        return 1.0, 0.0, 0.0

    pre_hw  = hw_gr[:ps][valid]
    tw_at_p = np.interp(known[valid], tw_tvt, tw_gr)

    if tw_at_p.std() < 0.1 or pre_hw.std() < 0.1:
        return 1.0, 0.0, 0.0

    reg = LinearRegression().fit(tw_at_p.reshape(-1, 1), pre_hw)
    a   = float(reg.coef_[0])
    b   = float(reg.intercept_)
    pred = a * tw_at_p + b
    r2  = float(1.0 - np.var(pre_hw - pred) / (np.var(pre_hw) + 1e-9))
    return a, b, max(0.0, r2)


def _gr_value_matching(hw_gr_raw: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                        lkt: np.ndarray, calib_a: float, calib_b: float) -> dict:
    """
    Vectorised GR value matching: for each row find TW TVT where tw_gr ≈ hw_gr_calibrated.

    Search is centred on last_known_tvt (lkt) — NOT physics_tvt — because lkt is much
    more stable for nearly-horizontal post-PS sections.

    Returns dict of arrays (shape n), one per (sigma, radius) combination.
    """
    n         = len(hw_gr_raw)
    tw_gr_sm  = gaussian_filter1d(tw_gr, sigma=2.0)  # mild TW smoothing

    out = {}
    for sigma in GR_SIGMAS:
        if sigma == 0:
            gr_use = hw_gr_raw.copy()
        else:
            gr_use = gaussian_filter1d(hw_gr_raw, sigma=float(sigma))

        # Invert calibration: expected tw_gr = (hw_gr - b) / a
        gr_tgt = (gr_use - calib_b) / (calib_a + 1e-9)

        for sr in SEARCH_RADII:
            N   = max(1, int(sr / TW_STEP))  # window half-width in TW samples
            # Centre indices into tw_tvt for each row
            tw_c = np.searchsorted(tw_tvt, lkt).clip(N, len(tw_tvt) - N - 1)

            # Build (n, 2N+1) windows of TW TVT and GR
            offsets      = np.arange(-N, N + 1)                              # (2N+1,)
            tw_idx_mat   = (tw_c[:, None] + offsets[None, :]).clip(0, len(tw_tvt) - 1)  # (n, 2N+1)
            tw_gr_wins   = tw_gr_sm[tw_idx_mat]                              # (n, 2N+1)
            tw_tvt_wins  = tw_tvt[tw_idx_mat]                                # (n, 2N+1)

            # Find best match per row
            diffs       = np.abs(tw_gr_wins - gr_tgt[:, None])               # (n, 2N+1)
            best_local  = np.argmin(diffs, axis=1)                           # (n,)
            best_tvt    = tw_tvt_wins[np.arange(n), best_local]              # (n,)
            best_gr_err = diffs[np.arange(n), best_local]                   # (n,) min abs diff

            tag = f"s{sigma}_r{sr}"
            out[f"gr_val_tvt_{tag}"]   = best_tvt
            out[f"gr_val_delta_{tag}"] = best_tvt - lkt                      # offset from lkt
            out[f"gr_val_err_{tag}"]   = best_gr_err                        # residual GR error

    return out


def _global_xcorr(hw_gr: np.ndarray, phys: np.ndarray, tw_tvt: np.ndarray,
                   tw_gr: np.ndarray, ps: int) -> dict:
    """
    Per-well global GR-typewell cross-correlation.

    Correlates entire post-PS HW GR (σ=5 smoothed) against tw_gr(physics_tvt + delta)
    for each delta in GLOBAL_DELTAS.  Finds the constant TVT offset that best explains
    the full post-PS GR signal — same approach as v17 but with more diagnostic fields.

    Returns scalar dict that gets broadcast to all rows in compute_gr_features().
    """
    n_post = len(hw_gr) - ps
    if n_post < 20:
        mid = len(GLOBAL_DELTAS) // 2
        return {
            "xcorr_global_best_delta": 0.0,
            "xcorr_global_peak_corr":  0.0,
            "xcorr_global_corr_at_0":  0.0,
            "xcorr_global_corr_std":   0.0,
            "xcorr_global_corr_rel":   0.0,
            "xcorr_global_second_delta": 0.0,
        }

    gr_post  = gaussian_filter1d(hw_gr[ps:], sigma=5.0)
    phys_post = phys[ps:]

    corrs = np.zeros(len(GLOBAL_DELTAS))
    for k, delta in enumerate(GLOBAL_DELTAS):
        tw_at = np.interp(phys_post + delta, tw_tvt, tw_gr)
        if gr_post.std() > 0.1 and tw_at.std() > 0.1:
            c = float(np.corrcoef(gr_post, tw_at)[0, 1])
            corrs[k] = c if not np.isnan(c) else 0.0

    mid_idx   = len(GLOBAL_DELTAS) // 2
    best_idx  = int(np.argmax(corrs))
    best_d    = float(GLOBAL_DELTAS[best_idx])
    best_corr = float(corrs[best_idx])
    corr_at_0 = float(corrs[mid_idx])

    # Second-best peak (local peak search excluding ±10 ft around best)
    excl_half = max(1, int(10 / (GLOBAL_DELTAS[1] - GLOBAL_DELTAS[0])))
    corrs2 = corrs.copy()
    lo = max(0, best_idx - excl_half)
    hi = min(len(corrs), best_idx + excl_half + 1)
    corrs2[lo:hi] = -np.inf
    second_idx = int(np.argmax(corrs2))
    second_d   = float(GLOBAL_DELTAS[second_idx])

    return {
        "xcorr_global_best_delta":   best_d,
        "xcorr_global_peak_corr":    best_corr,
        "xcorr_global_corr_at_0":    corr_at_0,
        "xcorr_global_corr_std":     float(corrs.std()),
        "xcorr_global_corr_rel":     best_corr - corr_at_0,
        "xcorr_global_second_delta": second_d,
    }


def _local_xcorr(hw_gr: np.ndarray, phys: np.ndarray, lkt: np.ndarray,
                  tw_tvt: np.ndarray, tw_gr: np.ndarray, ps: int) -> dict:
    """
    Per-row sliding-window GR-typewell cross-correlation.

    For each sampled row i (stride LOCAL_STRIDE), extract a window [i-W, i+W] of
    smoothed HW GR.  Compare to typewell GR sampled at:
        lkt[i] + (phys_win_rel + delta)    for delta in LOCAL_DELTAS

    where phys_win_rel = phys[window] - phys[i] encodes the relative TVT shape.
    The best delta gives the TVT correction above lkt.

    All deltas are computed simultaneously via broadcasting (vectorised inner loop).
    Non-sampled rows are filled by linear interpolation.

    Returns dict of per-row arrays of shape (n,).
    """
    n       = len(hw_gr)
    gr_sm   = gaussian_filter1d(hw_gr, sigma=5.0)
    tw_gr_s = gaussian_filter1d(tw_gr, sigma=1.5)

    out = {}
    for W in LOCAL_WINDOWS:
        xcorr_best_delta = np.zeros(n)
        xcorr_best_score = np.zeros(n)

        sample_rows = np.arange(ps, n, LOCAL_STRIDE)

        for i in sample_rows:
            i0 = max(0, i - W)
            i1 = min(n, i + W + 1)
            hw_win = gr_sm[i0:i1]
            hw_std = float(hw_win.std())
            if hw_std < 0.1:
                continue

            hw_norm = (hw_win - hw_win.mean()) / hw_std

            # Relative TVT variation within window (encodes local formation dip signal)
            phys_rel = phys[i0:i1] - phys[i]  # shape (window_len,)

            # Build TW windows for all deltas simultaneously
            # all_tvt shape: (n_deltas, window_len)
            all_tvt = lkt[i] + LOCAL_DELTAS[:, None] + phys_rel[None, :]
            # Clamp to valid TW range
            all_tvt = np.clip(all_tvt, tw_tvt[0], tw_tvt[-1])
            # Interpolate TW GR — ravel for speed, then reshape
            tw_wins = np.interp(all_tvt.ravel(), tw_tvt, tw_gr_s).reshape(
                len(LOCAL_DELTAS), len(hw_win)
            )  # (n_deltas, window_len)

            # Normalise each row (per-delta)
            tw_stds = tw_wins.std(axis=1, keepdims=True).clip(0.01)
            tw_norms = (tw_wins - tw_wins.mean(axis=1, keepdims=True)) / tw_stds

            # Pearson correlation via dot product (hw_norm is already unit-normalised)
            corrs = (hw_norm[None, :] * tw_norms).mean(axis=1)  # (n_deltas,)

            best_idx = int(np.argmax(corrs))
            xcorr_best_delta[i] = float(LOCAL_DELTAS[best_idx])
            xcorr_best_score[i] = float(corrs[best_idx])

        # Linear interpolation for non-sampled rows
        if len(sample_rows) > 1:
            full_delta = np.interp(np.arange(n), sample_rows,
                                   xcorr_best_delta[sample_rows])
            full_score = np.interp(np.arange(n), sample_rows,
                                   xcorr_best_score[sample_rows])
        else:
            full_delta = xcorr_best_delta.copy()
            full_score = xcorr_best_score.copy()

        out[f"local_xcorr_tvt_{W}"]   = lkt + full_delta
        out[f"local_xcorr_delta_{W}"] = full_delta
        out[f"local_xcorr_score_{W}"] = full_score

    return out


def _gradient_features(hw_gr: np.ndarray, md: np.ndarray, tw_tvt: np.ndarray,
                        tw_gr: np.ndarray, lkt: np.ndarray,
                        gr_val_tvt: np.ndarray) -> dict:
    """
    GR gradient matching features.

    Compares the local derivative of HW GR (w.r.t. MD) against the TW GR gradient
    at the estimated TVT position.  Sign and magnitude agreement indicate reliable
    TVT estimates.
    """
    # HW GR gradient (smooth first to reduce noise)
    gr_sm       = gaussian_filter1d(hw_gr, sigma=3.0)
    gr_grad_hw  = np.gradient(gr_sm, md)                     # d(hw_gr)/d(MD)

    # TW GR gradient (precompute)
    tw_gr_sm    = gaussian_filter1d(tw_gr, sigma=2.0)
    tw_grad     = np.gradient(tw_gr_sm, tw_tvt)              # d(tw_gr)/d(TVT)

    # TW gradient at lkt position
    tw_grad_at_lkt  = np.interp(lkt, tw_tvt, tw_grad,
                                 left=tw_grad[0], right=tw_grad[-1])
    # TW gradient at GR-value matched TVT
    tw_grad_at_val  = np.interp(gr_val_tvt, tw_tvt, tw_grad,
                                 left=tw_grad[0], right=tw_grad[-1])

    # Sign match: +1 if same sign, -1 if opposite, 0 if either near zero
    sign_match_lkt = np.sign(gr_grad_hw) * np.sign(tw_grad_at_lkt)
    sign_match_val = np.sign(gr_grad_hw) * np.sign(tw_grad_at_val)

    # Gradient ratio (how similar are the magnitudes?)
    gr_grad_ratio_lkt = gr_grad_hw / (np.abs(tw_grad_at_lkt) + 1e-9)

    # TW GR curvature at lkt
    tw_grad2 = np.gradient(tw_grad, tw_tvt)
    tw_curv_at_lkt = np.interp(lkt, tw_tvt, tw_grad2,
                                left=tw_grad2[0], right=tw_grad2[-1])

    return {
        "gr_grad_hw":          gr_grad_hw,
        "tw_grad_at_lkt":      tw_grad_at_lkt,
        "tw_grad_at_val_tvt":  tw_grad_at_val,
        "gr_grad_sign_match":  sign_match_lkt,
        "gr_grad_sign_val":    sign_match_val,
        "gr_grad_ratio_lkt":   gr_grad_ratio_lkt,
        "tw_curv_at_lkt":      tw_curv_at_lkt,
    }


def _formation_boundary_features(hw_gr: np.ndarray, md: np.ndarray,
                                   lkt: np.ndarray, tw_tvt: np.ndarray,
                                   tw_gr: np.ndarray) -> dict:
    """
    Detect sharp GR changes (formation boundaries) and compute distance features.

    Sharp GR changes mark formation bed boundaries that are recognisable in both
    the horizontal well and the typewell at specific TVT positions.
    """
    n = len(hw_gr)

    # GR boundary strength: absolute gradient of smoothed GR
    gr_sm     = gaussian_filter1d(hw_gr, sigma=2.0)
    gr_grad   = np.abs(np.gradient(gr_sm, md))
    thresh    = BOUNDARY_THRESH

    # Boundary locations
    is_bound  = (gr_grad > thresh).astype(float)

    # Distance to previous boundary (rows) — forward-fill from last boundary
    rows_to_prev = np.full(n, n, dtype=float)
    last_b = 0
    for i in range(n):
        if is_bound[i] > 0.5:
            last_b = i
        rows_to_prev[i] = float(i - last_b)

    # Distance to next boundary — backward from next boundary
    rows_to_next = np.full(n, n, dtype=float)
    next_b = n - 1
    for i in range(n - 1, -1, -1):
        if is_bound[i] > 0.5:
            next_b = i
        rows_to_next[i] = float(next_b - i)

    # GR change magnitude (raw) at boundary vicinity — smoothed rolling max of gradient
    boundary_strength = gr_grad  # per-row

    # TW GR local range at lkt (how much GR varies near current estimated position)
    tw_gr_sm = gaussian_filter1d(tw_gr, sigma=1.5)
    tw_idx   = np.searchsorted(tw_tvt, lkt).clip(0, len(tw_tvt) - 1)
    half_w   = 25  # ±25 TW samples = ±12.5 ft
    tw_idx_lo = np.clip(tw_idx - half_w, 0, len(tw_tvt) - 1)
    tw_idx_hi = np.clip(tw_idx + half_w, 0, len(tw_tvt) - 1)
    tw_local_range = np.array([
        float(np.ptp(tw_gr_sm[lo:hi + 1])) if hi > lo else 0.0
        for lo, hi in zip(tw_idx_lo, tw_idx_hi)
    ])

    return {
        "gr_boundary_strength": boundary_strength,
        "rows_to_prev_bound":   rows_to_prev,
        "rows_to_next_bound":   rows_to_next,
        "tw_local_range_at_lkt": tw_local_range,
    }


def _pre_ps_pattern_features(hw_gr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                               ps: int, tvt_inp: np.ndarray, lkt: np.ndarray) -> dict:
    """
    Pre-PS GR vs typewell correlation features.

    Computes the correlation of the pre-PS GR section against the typewell at
    various delta offsets (using TVT_input as the reference mapping).
    The best pre-PS delta indicates how well the physics TVT is calibrated.
    """
    n = len(hw_gr)

    if ps < 20:
        return {
            "pre_ps_xcorr_best_delta": 0.0,
            "pre_ps_xcorr_peak_corr":  0.0,
            "pre_ps_xcorr_corr_at_0":  0.0,
        }

    gr_pre  = gaussian_filter1d(hw_gr[:ps], sigma=3.0)
    tvt_pre = tvt_inp[:ps]
    valid   = ~np.isnan(tvt_pre)
    if valid.sum() < 10:
        return {
            "pre_ps_xcorr_best_delta": 0.0,
            "pre_ps_xcorr_peak_corr":  0.0,
            "pre_ps_xcorr_corr_at_0":  0.0,
        }

    tvt_v  = tvt_pre[valid]
    gr_v   = gr_pre[valid]

    deltas = np.arange(-30, 31, 2.0)  # narrower range for pre-PS (should be well-calibrated)
    corrs  = np.zeros(len(deltas))
    for k, d in enumerate(deltas):
        tw_at = np.interp(tvt_v + d, tw_tvt, tw_gr)
        if gr_v.std() > 0.1 and tw_at.std() > 0.1:
            c = float(np.corrcoef(gr_v, tw_at)[0, 1])
            corrs[k] = c if not np.isnan(c) else 0.0

    mid_idx  = len(deltas) // 2
    best_idx = int(np.argmax(corrs))

    return {
        "pre_ps_xcorr_best_delta": float(deltas[best_idx]),
        "pre_ps_xcorr_peak_corr":  float(corrs[best_idx]),
        "pre_ps_xcorr_corr_at_0":  float(corrs[mid_idx]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main per-well function
# ══════════════════════════════════════════════════════════════════════════════

def compute_gr_features(hw: pd.DataFrame, tw: pd.DataFrame,
                         ps_idx: int, physics_tvt_array: np.ndarray,
                         split: str = "train") -> pd.DataFrame:
    """
    Compute all GR-typewell matching features for ONE well.

    Parameters
    ----------
    hw : pd.DataFrame
        Horizontal well data (pre-PS + post-PS rows).
    tw : pd.DataFrame
        Typewell data with columns ['TVT', 'GR'].
    ps_idx : int
        Row index of Prediction Start (first NaN TVT_input row).
    physics_tvt_array : np.ndarray
        Pre-computed physics-based TVT estimate for all rows (shape n).
    split : str
        'train' or 'test' — determines whether target TVT is available.

    Returns
    -------
    pd.DataFrame
        One row per MD point with all GR features + metadata columns.
        No data leakage: all per-row features use only pre-PS knowledge
        (calibration from pre-PS section) or typewell data (which is known).
    """
    n   = len(hw)
    ps  = ps_idx

    # ── Raw arrays ─────────────────────────────────────────────────────────────
    hw_gr     = fill_arr(hw["GR"])
    md        = hw["MD"].astype(float).values
    tw_tvt    = tw["TVT"].astype(float).values
    tw_gr     = fill_arr(tw["GR"])
    tvt_inp   = hw["TVT_input"].astype(str).replace("", np.nan).astype(float).values
    lkt       = pd.Series(tvt_inp).ffill().bfill().astype(float).values
    phys      = physics_tvt_array.copy()

    # ── 1. Pre-PS GR calibration ───────────────────────────────────────────────
    calib_a, calib_b, calib_r2 = _pre_ps_calibration(
        hw_gr, tw_tvt, tw_gr, ps, tvt_inp
    )

    # ── 2. GR value matching (vectorised) ─────────────────────────────────────
    val_match = _gr_value_matching(hw_gr, tw_tvt, tw_gr, lkt, calib_a, calib_b)
    # Primary GR-value TVT estimate: σ3, search_radius=30 ft
    gr_val_tvt_primary = val_match.get("gr_val_tvt_s3_r30",
                                        val_match.get("gr_val_tvt_s0_r30", lkt))

    # ── 3. Global GR-typewell cross-correlation ────────────────────────────────
    global_xc = _global_xcorr(hw_gr, phys, tw_tvt, tw_gr, ps)
    global_best_delta = global_xc["xcorr_global_best_delta"]
    xcorr_corrected_tvt = phys + global_best_delta  # per-row corrected TVT

    # ── 4. Local sliding-window cross-correlation ──────────────────────────────
    local_xc = _local_xcorr(hw_gr, phys, lkt, tw_tvt, tw_gr, ps)
    # Primary local xcorr: window=500
    local_xcorr_tvt_primary = local_xc.get("local_xcorr_tvt_500",
                                             local_xc.get("local_xcorr_tvt_200", lkt))

    # ── 5. GR gradient features ────────────────────────────────────────────────
    grad_feats = _gradient_features(hw_gr, md, tw_tvt, tw_gr, lkt, gr_val_tvt_primary)

    # ── 6. Formation boundary features ────────────────────────────────────────
    bound_feats = _formation_boundary_features(hw_gr, md, lkt, tw_tvt, tw_gr)

    # ── 7. Pre-PS pattern features ─────────────────────────────────────────────
    pre_ps_feats = _pre_ps_pattern_features(hw_gr, tw_tvt, tw_gr, ps, tvt_inp, lkt)

    # ── 8. Derived combination features ───────────────────────────────────────
    # TW GR at various TVT estimates
    tw_gr_sm = gaussian_filter1d(tw_gr, sigma=2.0)
    gr_at_lkt            = np.interp(lkt, tw_tvt, tw_gr_sm)
    gr_at_xcorr_global   = np.interp(xcorr_corrected_tvt, tw_tvt, tw_gr_sm)
    gr_at_local_xcorr    = np.interp(local_xcorr_tvt_primary, tw_tvt, tw_gr_sm)
    gr_at_val_match      = np.interp(gr_val_tvt_primary, tw_tvt, tw_gr_sm)

    # GR deviation from TW at each estimate
    hw_gr_sm = gaussian_filter1d(hw_gr, sigma=3.0)
    gr_dev_lkt          = hw_gr_sm - gr_at_lkt
    gr_dev_xcorr_global = hw_gr_sm - gr_at_xcorr_global
    gr_dev_local_xcorr  = hw_gr_sm - gr_at_local_xcorr
    gr_dev_val_match    = hw_gr_sm - gr_at_val_match

    # Agreement between different TVT estimates
    val_vs_local_xcorr  = gr_val_tvt_primary - local_xcorr_tvt_primary
    val_vs_global_xcorr = gr_val_tvt_primary - xcorr_corrected_tvt
    xcorr_vs_lkt        = xcorr_corrected_tvt - lkt

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    feat: dict = {
        # Metadata
        "is_post_ps":      (np.arange(n) >= ps).astype(int),
        "row_idx":         np.arange(n),
        "ps_idx":          float(ps),
        "last_known_tvt":  lkt,
        "physics_tvt":     phys,

        # Calibration scalars (broadcast)
        "gr_calib_a":      np.full(n, calib_a),
        "gr_calib_b":      np.full(n, calib_b),
        "gr_calib_r2":     np.full(n, calib_r2),

        # Derived TW GR at key positions
        "gr_at_lkt":            gr_at_lkt,
        "gr_at_xcorr_global":   gr_at_xcorr_global,
        "gr_at_local_xcorr":    gr_at_local_xcorr,
        "gr_at_val_match":      gr_at_val_match,

        # GR deviations
        "gr_dev_lkt":           gr_dev_lkt,
        "gr_dev_xcorr_global":  gr_dev_xcorr_global,
        "gr_dev_local_xcorr":   gr_dev_local_xcorr,
        "gr_dev_val_match":     gr_dev_val_match,

        # Agreement between TVT estimates
        "val_vs_local_xcorr":  val_vs_local_xcorr,
        "val_vs_global_xcorr": val_vs_global_xcorr,
        "xcorr_vs_lkt":        xcorr_vs_lkt,

        # Global xcorr (broadcast per-well constants + per-row corrected TVT)
        "xcorr_global_corrected_tvt": xcorr_corrected_tvt,
    }

    # Add all per-row feature dicts
    feat.update(val_match)
    feat.update(local_xc)
    feat.update(grad_feats)
    feat.update(bound_feats)
    # Broadcast global xcorr scalars
    for k, v in global_xc.items():
        feat[k] = np.full(n, float(v))
    # Broadcast pre-PS xcorr scalars
    for k, v in pre_ps_feats.items():
        feat[k] = np.full(n, float(v))

    df = pd.DataFrame(feat)
    df["well_id"] = None  # will be overwritten by build_feature_matrix()

    # Target (train only)
    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        df["target_tvt"]         = tvt_true
        df["target_tvt_delta"]   = tvt_true - lkt           # delta from last known
        df["target_correction"]  = tvt_true - phys          # correction to physics

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Build full feature matrix for a split
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(split: str = "train") -> pd.DataFrame:
    """
    Build and return the GR-feature matrix for all wells in a given split.

    Parameters
    ----------
    split : str
        One of 'train', 'val', or 'test'.

    Returns
    -------
    pd.DataFrame
        Concatenated per-well DataFrames (ALL rows, is_post_ps flag).
        Saved to features/gr_xcorr_features_{split}.pkl automatically.
    """
    if split in ("train", "val"):
        ids_path = FEAT_DIR / f"{split}_ids.csv"
        well_ids = pd.read_csv(ids_path, header=None)[0].tolist()
        data_split = "train"
    else:
        well_ids  = TEST_WELLS
        data_split = "test"

    total = len(well_ids)
    print(f"\n{'='*60}")
    print(f"Building GR features — split='{split}' ({total} wells)")
    print(f"{'='*60}")

    dfs   = []
    skips = []
    t_all = time.time()

    for i, wid in enumerate(well_ids):
        print(f"  Processing {wid} ({i+1}/{total})", end="\r", flush=True)
        try:
            hw, tw = load_well(wid, data_split)
            ps     = get_ps(hw)
            phys   = get_physics_tvt(hw, ps, use_tvt_col=(data_split == "train"))

            df = compute_gr_features(hw, tw, ps, phys, split=data_split)
            df["well_id"] = wid
            dfs.append(df)

        except Exception as exc:
            print(f"\n  WARNING: skipping {wid} — {exc}")
            skips.append(wid)

    print(f"\n  Done: {len(dfs)} wells processed, {len(skips)} skipped "
          f"({time.time()-t_all:.1f}s total)")

    if not dfs:
        raise RuntimeError(f"No wells processed for split='{split}'")

    combined = pd.concat(dfs, ignore_index=True)
    out_path = FEAT_DIR / f"gr_xcorr_features_{split}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(combined, f, protocol=4)
    print(f"  Saved → {out_path}  shape={combined.shape}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Validation: per-row TVT accuracy on well 000d7d20
# ══════════════════════════════════════════════════════════════════════════════

def validate_well_000d7d20():
    """
    Validate GR-based TVT estimates on well 000d7d20.
    Prints MAE comparison for key feature signals vs physics.

    Expected: gr_val_tvt_s3_r30 should give ~15-20 ft MAE (vs ~49 ft for physics).
    TVT range should be within ~11734-11750 ft for well 000d7d20.
    """
    WID = "000d7d20"
    print(f"\n{'='*60}")
    print(f"Validation: well {WID}")
    print(f"{'='*60}")

    hw_tr, tw_tr = load_well(WID, "train")
    hw_te, tw_te = load_well(WID, "test")
    ps = get_ps(hw_te)

    # Use test-version (no TVT col) for features, train for ground truth
    phys = get_physics_tvt(hw_te, ps, use_tvt_col=False)
    df   = compute_gr_features(hw_te, tw_te, ps, phys, split="test")

    tvt_true = hw_tr["TVT"].astype(float).values[ps:]
    df_post  = df[df["is_post_ps"] == 1].reset_index(drop=True)

    print(f"\nPost-PS rows: {len(df_post)}")
    print(f"True TVT range: [{tvt_true.min():.2f}, {tvt_true.max():.2f}] ft")
    print(f"Last known TVT: {df_post['last_known_tvt'].iloc[0]:.2f} ft\n")

    def mae(est):
        return float(np.mean(np.abs(np.asarray(est) - tvt_true)))
    def bias(est):
        return float(np.mean(np.asarray(est) - tvt_true))

    candidates = {
        "physics_tvt":                df_post["physics_tvt"].values,
        "last_known_tvt":             df_post["last_known_tvt"].values,
        "xcorr_global (phys+delta)":  df_post["xcorr_global_corrected_tvt"].values,
    }
    # Add all GR value match features
    val_cols = [c for c in df_post.columns if c.startswith("gr_val_tvt_")]
    for c in sorted(val_cols):
        candidates[c] = df_post[c].values
    # Add local xcorr
    for c in sorted([c for c in df_post.columns if c.startswith("local_xcorr_tvt_")]):
        candidates[c] = df_post[c].values

    print(f"{'Feature':<45} {'MAE':>7}  {'Bias':>7}  {'Range':>18}")
    print("-" * 82)
    for name, vals in candidates.items():
        lo, hi = vals.min(), vals.max()
        print(f"{name:<45} {mae(vals):>7.2f}  {bias(vals):>7.2f}  [{lo:.1f}, {hi:.1f}]")

    print()
    # Check feature column count
    feat_cols = [c for c in df_post.columns
                 if c not in {"well_id", "is_post_ps", "row_idx", "ps_idx",
                               "target_tvt", "target_tvt_delta", "target_correction"}]
    print(f"Total features: {len(feat_cols)}")
    print(f"DataFrame shape (post-PS): {df_post.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("Feature Engineering v3 — GR-Typewell Matching Features")
    print("=" * 60)

    # ── 1. Validate on reference well first ───────────────────────────────────
    validate_well_000d7d20()

    # ── 2. Build feature matrices for all splits ──────────────────────────────
    t0 = time.time()

    print("\n\n[Phase 1/3] Building TRAIN features...")
    df_train = build_feature_matrix("train")

    print("\n[Phase 2/3] Building VAL features...")
    df_val = build_feature_matrix("val")

    print("\n[Phase 3/3] Building TEST features...")
    df_test = build_feature_matrix("test")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    total_sec = time.time() - t0
    feat_cols = [c for c in df_train.columns
                 if c not in {"well_id", "is_post_ps", "row_idx", "ps_idx",
                               "target_tvt", "target_tvt_delta", "target_correction"}]

    print(f"\n{'='*60}")
    print("Feature Engineering v3 — Summary")
    print(f"{'='*60}")
    print(f"  Train shape  : {df_train.shape}")
    print(f"  Val shape    : {df_val.shape}")
    print(f"  Test shape   : {df_test.shape}")
    print(f"  Feature count: {len(feat_cols)}")
    print(f"  Total time   : {total_sec/60:.1f} min")
    print()

    print("Feature list:")
    for i, c in enumerate(sorted(feat_cols), 1):
        print(f"  {i:3d}. {c}")

    print(f"\nOutput files:")
    for split in ("train", "val", "test"):
        p = FEAT_DIR / f"gr_xcorr_features_{split}.pkl"
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
