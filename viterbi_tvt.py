"""
Viterbi TVT Prediction via GR-based HMM
=========================================
Uses dynamic programming to find the TVT trajectory that:
1. Matches the observed GR pattern (emission from typewell GR)
2. Follows physics-based transitions (anchor + slope × dZ + noise)

Key advantage over DTW: ALLOWS NON-MONOTONIC TVT trajectories
All 3 test wells have decreasing TVT — DTW (forward-only) cannot handle this.

Validates on val wells, then generates test well predictions.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN = DATA / "train"
TEST  = DATA / "test"
FEAT  = DATA / "features"
SUBS  = DATA / "submissions"

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

val_ids   = pd.read_csv(FEAT / "val_ids.csv",   header=None)[0].tolist()

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

def fill_ser(s): return s.ffill().bfill().astype(float)


def estimate_gr_sigma(hw, tw, ps, last_tvt, tw_interp):
    """Estimate GR observation noise from pre-PS section where TVT is known."""
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    pre_tvt = tvt_inp[:ps].dropna().values
    pre_gr  = fill_ser(hw["GR"][:ps]).values[:ps]
    # Use last 500 pre-PS rows with known TVT
    n = min(500, len(pre_tvt))
    if n < 20:
        return 15.0
    pre_tvt = pre_tvt[-n:]
    pre_gr  = pre_gr[-n:]
    valid   = np.isfinite(pre_gr) & np.isfinite(pre_tvt)
    if valid.sum() < 10:
        return 15.0
    tw_gr_at_true = tw_interp(pre_tvt[valid])
    resid = pre_gr[valid] - tw_gr_at_true
    return float(np.std(resid)) + 1e-6


def predict_tvt_viterbi(hw, tw, ps, split="test", use_physics=True,
                        tvt_range=100.0, tvt_step=0.5, sigma_trans=None,
                        sigma_gr=None):
    """
    Viterbi TVT prediction.

    Parameters
    ----------
    tvt_range : float  — search ±tvt_range ft from anchor
    tvt_step  : float  — TVT grid resolution in ft
    sigma_trans : float — TVT random-walk sigma per row (if None, auto)
    sigma_gr    : float — GR observation noise sigma (if None, auto-estimate)
    """
    # ── Data ────────────────────────────────────────────────────────────────
    x  = hw["X"].astype(float).values
    y  = hw["Y"].astype(float).values
    z  = hw["Z"].astype(float).values
    gr_raw = fill_ser(hw["GR"]).values  # NaN handled as missing obs later

    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    last_tvt = float(tvt_inp.ffill().iloc[ps - 1])

    # Fit pre-PS TVT ~ slope*Z model
    pre_tvt_valid = tvt_inp[:ps].dropna()
    pre_z_valid   = z[:ps][ pre_tvt_valid.index]
    if len(pre_tvt_valid) >= 10:
        A = np.column_stack([pre_z_valid, np.ones(len(pre_z_valid))])
        coef, *_ = np.linalg.lstsq(A, pre_tvt_valid.values, rcond=None)
        phy_slope, phy_inter = coef[0], coef[1]
    else:
        phy_slope = 0.0
        phy_inter = last_tvt

    # ── Typewell interpolation ───────────────────────────────────────────────
    tw_tvt  = tw["TVT"].astype(float).values
    tw_gr   = fill_ser(tw["GR"]).values
    # Smooth typewell GR slightly
    tw_gr_s = gaussian_filter1d(tw_gr, sigma=1.5)
    tw_interp = interp1d(tw_tvt, tw_gr_s, bounds_error=False, fill_value="extrapolate")

    # ── Calibrate sigma_gr ──────────────────────────────────────────────────
    if sigma_gr is None:
        sigma_gr = estimate_gr_sigma(hw, tw, ps, last_tvt, tw_interp)
    sigma_gr = max(sigma_gr, 5.0)

    # ── TVT grid ─────────────────────────────────────────────────────────────
    tvt_grid = np.arange(last_tvt - tvt_range, last_tvt + tvt_range + 1e-9, tvt_step)
    n_states = len(tvt_grid)

    # Pre-compute typewell GR at all TVT grid points
    tw_gr_grid = tw_interp(tvt_grid)  # [n_states]

    # ── Physics-expected TVT per row ─────────────────────────────────────────
    z_post = z[ps:]
    n_post = len(z_post)
    if use_physics:
        # TVT_t = anchor + slope*(Z_t - Z_anchor)
        z_anchor = z[ps - 1]
        phy_tvt = last_tvt + phy_slope * (z_post - z_anchor)
    else:
        phy_tvt = np.full(n_post, last_tvt)

    # ── Per-row transition sigma ─────────────────────────────────────────────
    if sigma_trans is None:
        sigma_trans = 0.3   # ft/row default

    # ── GR at each post-PS row ───────────────────────────────────────────────
    gr_raw_post = hw["GR"].values[ps:]
    # NaN = missing observation

    # ── Viterbi ─────────────────────────────────────────────────────────────
    # log alpha[t, s]: log prob of best path reaching state s at time t
    # backtrack[t, s]: best previous state index
    INF = -1e30
    log_alpha = np.full(n_states, INF)

    # Init: Gaussian prior around anchor
    init_sigma = 3.0  # ft — tight prior at anchor
    log_alpha = -0.5 * ((tvt_grid - last_tvt) / init_sigma) ** 2

    backtrack = np.zeros((n_post, n_states), dtype=np.int16)

    # Precompute: inv(2*sigma²) for transition and emission
    inv_2sig2_tr = 0.5 / sigma_trans ** 2
    inv_2sig2_gr = 0.5 / sigma_gr ** 2

    # State differences matrix for transition cost
    # diff[i, j] = (tvt_grid[j] - tvt_grid[i])^2
    # Too expensive to do n_states x n_states.
    # Use banded approach: only update within ±band states
    # max TVT change per row = tvt_range (very generous but banded)
    band_states = min(n_states, int(4.0 / tvt_step) + 1)  # ±4 ft / step

    tvt_col = tvt_grid.reshape(-1, 1)  # [n_states, 1]
    tvt_row = tvt_grid.reshape(1, -1)  # [1, n_states]

    for t in range(n_post):
        # Physics-expected TVT at this step
        mu = phy_tvt[t]

        # Transition cost: for each current state s,
        # find best prev state = argmax(log_alpha_prev + trans_log)
        # trans_log[prev_s, curr_s] = -inv_2sig2_tr * (tvt_grid[curr_s] - tvt_grid[prev_s])^2

        # Vectorized: diff[i] = (tvt_grid - tvt_grid[i])^2 for all j
        # = (tvt_row - tvt_col)^2 shaped [n_states_prev, n_states_curr]
        # This is O(n_states^2) — optimize with band constraint

        # For each new state s_new:
        #   best_prev = argmax_over_s_prev { log_alpha[s_prev]
        #               - inv_2sig2_tr * (tvt_grid[s_new] - tvt_grid[s_prev])^2 }
        # We vectorize over s_new using broadcasting

        # d2[s_prev, s_new] = (tvt_grid[s_new] - tvt_grid[s_prev])^2
        d2 = (tvt_row - tvt_col) ** 2   # [n_states, n_states]
        trans_log = -inv_2sig2_tr * d2   # [n_states, n_states]

        # Optionally add physics bias: prefer transitions toward phy_tvt[t]
        # (bias toward physics prediction rather than free random walk)
        # physics_bias[s_new] = -inv_2sig2_tr * (tvt_grid[s_new] - mu)^2 * 0.3
        phy_bias = -0.3 * inv_2sig2_tr * (tvt_grid - mu) ** 2   # [n_states]

        # Score for each (prev, curr) pair
        scores = log_alpha[:, None] + trans_log  # [n_prev, n_curr]

        # Best previous state for each current state
        best_prev_idx = np.argmax(scores, axis=0)  # [n_states]
        best_prev_score = scores[best_prev_idx, np.arange(n_states)]

        new_log_alpha = best_prev_score + phy_bias

        # Emission: add GR log likelihood
        gr_t = gr_raw_post[t]
        if np.isfinite(gr_t):
            gr_diff2 = (tw_gr_grid - gr_t) ** 2
            new_log_alpha += -inv_2sig2_gr * gr_diff2

        backtrack[t] = best_prev_idx.astype(np.int16)
        log_alpha = new_log_alpha

    # ── Backtrack ────────────────────────────────────────────────────────────
    tvt_pred = np.zeros(n_post)
    s = int(np.argmax(log_alpha))
    tvt_pred[n_post - 1] = tvt_grid[s]
    for t in range(n_post - 2, -1, -1):
        s = int(backtrack[t + 1, s])
        tvt_pred[t] = tvt_grid[s]

    # Light smoothing
    tvt_pred = gaussian_filter1d(tvt_pred, sigma=2.0)

    return tvt_pred, sigma_gr


# ── Validation on val wells ──────────────────────────────────────────────────
print("Validating Viterbi on val wells...")

results_vit = []
results_phy = []

for i, wid in enumerate(val_ids[:40]):
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/40]")
    try:
        hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv")
        ps = get_ps(hw)
        tvt_true = hw["TVT"].astype(float).values[ps:]

        if len(tvt_true) < 10:
            continue

        # Physics only
        tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
        last_tvt = float(tvt_inp.ffill().iloc[ps - 1])

        z = hw["Z"].astype(float).values
        z_post = z[ps:]
        z_anchor = z[ps - 1]
        pre_tvt_valid = tvt_inp[:ps].dropna()
        pre_z_valid   = z[:ps][pre_tvt_valid.index]
        A = np.column_stack([pre_z_valid, np.ones(len(pre_z_valid))])
        coef, *_ = np.linalg.lstsq(A, pre_tvt_valid.values, rcond=None)
        phy_tvt = last_tvt + coef[0] * (z_post - z_anchor)

        rmse_phy    = float(np.sqrt(np.mean((phy_tvt - tvt_true) ** 2)))
        rmse_anchor = float(np.sqrt(np.mean((last_tvt - tvt_true) ** 2)))

        # Viterbi
        pred, sg = predict_tvt_viterbi(hw, tw, ps,
                                       tvt_range=60.0, tvt_step=0.5,
                                       sigma_trans=0.3, sigma_gr=None)
        rmse_vit = float(np.sqrt(np.mean((pred - tvt_true) ** 2)))

        results_vit.append({"wid": wid, "n": len(tvt_true),
                             "rmse_viterbi": rmse_vit,
                             "rmse_physics": rmse_phy,
                             "rmse_anchor":  rmse_anchor,
                             "sigma_gr": sg})
    except Exception as e:
        print(f"  SKIP {wid}: {e}")

df = pd.DataFrame(results_vit)

def rw(df, col):
    return float(np.sqrt(np.sum(df[col]**2 * df.n) / df.n.sum()))

print(f"\nResults on {len(df)} val wells:")
print(f"  Anchor     RMSE: {rw(df,'rmse_anchor'):.3f} ft (row-weighted)")
print(f"  Physics    RMSE: {rw(df,'rmse_physics'):.3f} ft (row-weighted)")
print(f"  Viterbi    RMSE: {rw(df,'rmse_viterbi'):.3f} ft (row-weighted)")
print(f"  Per-well mean:")
print(f"    Anchor  : {df.rmse_anchor.mean():.3f} ft")
print(f"    Physics : {df.rmse_physics.mean():.3f} ft")
print(f"    Viterbi : {df.rmse_viterbi.mean():.3f} ft")
print(f"  Sigma_gr mean: {df.sigma_gr.mean():.1f} API units")

print("\nTop improvement wells (anchor→viterbi):")
df["improvement"] = df.rmse_anchor - df.rmse_viterbi
print(df.nlargest(5, "improvement")[["wid","rmse_anchor","rmse_physics","rmse_viterbi","improvement","sigma_gr"]].to_string(index=False))
print("\nWorst regression wells:")
print(df.nsmallest(5, "improvement")[["wid","rmse_anchor","rmse_physics","rmse_viterbi","improvement","sigma_gr"]].to_string(index=False))
