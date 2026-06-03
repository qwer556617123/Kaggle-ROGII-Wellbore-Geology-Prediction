"""
GR/typewell TVT path search.

This is a sequence-alignment style experiment, not another row-level LGBM.
It predicts the whole post-PS TVT path by minimizing:

  GR mismatch to the typewell + smooth transition cost + weak physics prior

If the same well ID exists in train, the script can also write an oracle
diagnostic report. The prediction path itself never reads train TVT.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SUBS_DIR = DATA_DIR / "submissions"
DEFAULT_WELLS = ("000d7d20", "00bbac68", "00e12e8b")


@dataclass
class PathResult:
    well_id: str
    ps: int
    anchor: float
    slope: float
    sigma_gr: float
    tvt_pred: np.ndarray
    physics: np.ndarray
    anchor_pred: np.ndarray


def get_ps(hw: pd.DataFrame) -> int:
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def fill_float(values) -> np.ndarray:
    return pd.Series(values).ffill().bfill().astype(float).to_numpy()


def tvt_input(hw: pd.DataFrame) -> pd.Series:
    return hw["TVT_input"].astype(str).replace("", np.nan).astype(float)


def sorted_typewell(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tvt = tw["TVT"].astype(float).to_numpy()
    gr = fill_float(tw["GR"])
    order = np.argsort(tvt)
    return tvt[order], gr[order]


def robust_loss(z: np.ndarray, delta: float = 2.5) -> np.ndarray:
    a = np.abs(z)
    return np.where(a <= delta, 0.5 * z * z, delta * (a - 0.5 * delta))


def fit_physics(hw: pd.DataFrame, ps: int) -> tuple[float, float, np.ndarray]:
    z = hw["Z"].astype(float).to_numpy()
    ti = tvt_input(hw)
    known = ti.iloc[:ps].dropna()
    anchor = float(ti.ffill().iloc[max(0, ps - 1)])
    z_anchor = float(z[max(0, ps - 1)])

    if len(known) >= 8:
        pre_z = z[known.index.to_numpy()]
        A = np.column_stack([pre_z, np.ones(len(pre_z))])
        coef, *_ = np.linalg.lstsq(A, known.to_numpy(), rcond=None)
        slope = float(coef[0])
    else:
        slope = -1.0

    physics_full = anchor + slope * (z - z_anchor)
    return anchor, slope, physics_full[ps:]


def calibrate_gr(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    ps: int,
) -> tuple[float, float, float]:
    """Fit hw_gr ~= a * tw_gr(TVT_input) + b on pre-PS rows."""
    ti = tvt_input(hw)
    known = ti.iloc[:ps].dropna()
    hw_gr = fill_float(hw["GR"])
    if len(known) < 20:
        return 1.0, 0.0, 15.0

    idx = known.index.to_numpy()
    tw_at_known = np.interp(known.to_numpy(), tw_tvt, gaussian_filter1d(tw_gr, sigma=1.5))
    hw_known = hw_gr[idx]
    mask = np.isfinite(tw_at_known) & np.isfinite(hw_known)
    if mask.sum() < 20 or np.std(tw_at_known[mask]) < 0.1:
        return 1.0, 0.0, 15.0

    x = tw_at_known[mask]
    y = hw_known[mask]
    A = np.column_stack([x, np.ones(len(x))])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if abs(a) < 1e-6:
        return 1.0, 0.0, 15.0

    resid = y - (a * x + b)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma = max(5.0, 1.4826 * mad)
    return float(a), float(b), sigma


def build_emission_matrix(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    tvt_grid: np.ndarray,
    ps: int,
    calib_a: float,
    calib_b: float,
    sigma_gr: float,
    gr_sigmas: tuple[float, ...],
) -> np.ndarray:
    """Return cost matrix shaped (n_post, n_states)."""
    hw_gr_raw = fill_float(hw["GR"])
    n_post = len(hw_gr_raw) - ps
    costs = np.zeros((n_post, len(tvt_grid)), dtype=np.float32)

    weights = np.array([1.0 / (1.0 + s) for s in gr_sigmas], dtype=float)
    weights = weights / weights.sum()

    for sigma, weight in zip(gr_sigmas, weights):
        if sigma > 0:
            hw_gr_use = gaussian_filter1d(hw_gr_raw, sigma=sigma)
            tw_gr_use = gaussian_filter1d(tw_gr, sigma=sigma)
        else:
            hw_gr_use = hw_gr_raw
            tw_gr_use = tw_gr

        hw_as_tw_units = (hw_gr_use[ps:] - calib_b) / calib_a
        tw_at_grid = np.interp(tvt_grid, tw_tvt, tw_gr_use, left=tw_gr_use[0], right=tw_gr_use[-1])
        z = (hw_as_tw_units[:, None] - tw_at_grid[None, :]) / sigma_gr
        costs += (weight * robust_loss(z)).astype(np.float32)

    return costs


def viterbi_min_cost(
    emission: np.ndarray,
    tvt_grid: np.ndarray,
    physics: np.ndarray,
    anchor: float,
    tvt_step: float,
    transition_sigma: float,
    prior_sigma: float,
    transition_band_ft: float,
) -> np.ndarray:
    n_rows, n_states = emission.shape
    back = np.zeros((n_rows, n_states), dtype=np.int32)
    grid = tvt_grid.astype(np.float32)
    band = max(1, int(round(transition_band_ft / tvt_step)))

    dp = 0.5 * ((grid - anchor) / 3.0) ** 2
    dp += 0.5 * ((grid - physics[0]) / prior_sigma) ** 2
    dp += emission[0]

    for t in range(1, n_rows):
        expected_step = float(physics[t] - physics[t - 1])
        prior = 0.5 * ((grid - physics[t]) / prior_sigma) ** 2
        best = np.full(n_states, np.inf, dtype=np.float32)
        best_prev = np.zeros(n_states, dtype=np.int32)

        for offset in range(-band, band + 1):
            if offset >= 0:
                curr = np.arange(offset, n_states)
                prev = curr - offset
            else:
                curr = np.arange(0, n_states + offset)
                prev = curr - offset

            step = grid[curr] - grid[prev]
            transition = 0.5 * ((step - expected_step) / transition_sigma) ** 2
            candidate = dp[prev] + transition.astype(np.float32)
            improve = candidate < best[curr]
            if np.any(improve):
                curr_i = curr[improve]
                best[curr_i] = candidate[improve]
                best_prev[curr_i] = prev[improve]

        dp = best + prior.astype(np.float32) + emission[t]
        back[t] = best_prev

    path_idx = np.zeros(n_rows, dtype=np.int32)
    path_idx[-1] = int(np.argmin(dp))
    for t in range(n_rows - 1, 0, -1):
        path_idx[t - 1] = back[t, path_idx[t]]

    return tvt_grid[path_idx]


def predict_well(
    well_id: str,
    test_dir: Path,
    tvt_range: float,
    tvt_step: float,
    transition_sigma: float,
    prior_sigma: float,
    transition_band_ft: float,
    gr_sigmas: tuple[float, ...],
    smooth_sigma: float,
) -> PathResult:
    hw = pd.read_csv(test_dir / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(test_dir / f"{well_id}__typewell.csv")
    ps = get_ps(hw)
    anchor, slope, physics = fit_physics(hw, ps)
    tw_tvt, tw_gr = sorted_typewell(tw)
    calib_a, calib_b, sigma_gr = calibrate_gr(hw, tw_tvt, tw_gr, ps)

    grid_min = min(anchor - tvt_range, float(np.min(physics)) - 40.0)
    grid_max = max(anchor + tvt_range, float(np.max(physics)) + 40.0)
    tvt_grid = np.arange(grid_min, grid_max + 1e-9, tvt_step)

    emission = build_emission_matrix(
        hw=hw,
        tw_tvt=tw_tvt,
        tw_gr=tw_gr,
        tvt_grid=tvt_grid,
        ps=ps,
        calib_a=calib_a,
        calib_b=calib_b,
        sigma_gr=sigma_gr,
        gr_sigmas=gr_sigmas,
    )
    pred = viterbi_min_cost(
        emission=emission,
        tvt_grid=tvt_grid,
        physics=physics,
        anchor=anchor,
        tvt_step=tvt_step,
        transition_sigma=transition_sigma,
        prior_sigma=prior_sigma,
        transition_band_ft=transition_band_ft,
    )
    if smooth_sigma > 0:
        pred = gaussian_filter1d(pred, sigma=smooth_sigma)

    return PathResult(
        well_id=well_id,
        ps=ps,
        anchor=anchor,
        slope=slope,
        sigma_gr=sigma_gr,
        tvt_pred=pred,
        physics=physics,
        anchor_pred=np.full(len(pred), anchor, dtype=float),
    )


def oracle_metrics(well_id: str, result: PathResult) -> dict[str, float] | None:
    train_hw_path = TRAIN_DIR / f"{well_id}__horizontal_well.csv"
    if not train_hw_path.exists():
        return None
    hw = pd.read_csv(train_hw_path)
    if "TVT" not in hw.columns or len(hw) < result.ps + len(result.tvt_pred):
        return None

    true = hw["TVT"].astype(float).to_numpy()[result.ps : result.ps + len(result.tvt_pred)]
    rmse_path = float(np.sqrt(np.mean((result.tvt_pred - true) ** 2)))
    rmse_physics = float(np.sqrt(np.mean((result.physics - true) ** 2)))
    rmse_anchor = float(np.sqrt(np.mean((result.anchor_pred - true) ** 2)))
    return {
        "oracle_rmse_path": rmse_path,
        "oracle_rmse_physics": rmse_physics,
        "oracle_rmse_anchor": rmse_anchor,
        "true_start": float(true[0]),
        "true_end": float(true[-1]),
        "true_range": float(np.max(true) - np.min(true)),
    }


def build_submission(results: list[PathResult], output_path: Path) -> pd.DataFrame:
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    for result in results:
        ids = [f"{result.well_id}_{i}" for i in range(result.ps, result.ps + len(result.tvt_pred))]
        mapping = dict(zip(ids, result.tvt_pred))
        mask = sub["id"].isin(mapping.keys())
        sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    return sub


def summarize_result(result: PathResult) -> dict[str, float | str | int]:
    pred = result.tvt_pred
    physics = result.physics
    return {
        "well_id": result.well_id,
        "ps": result.ps,
        "n_post": len(pred),
        "anchor": result.anchor,
        "slope": result.slope,
        "sigma_gr": result.sigma_gr,
        "pred_start": float(pred[0]),
        "pred_end": float(pred[-1]),
        "pred_net": float(pred[-1] - pred[0]),
        "pred_range": float(np.max(pred) - np.min(pred)),
        "pred_std": float(np.std(pred)),
        "physics_start": float(physics[0]),
        "physics_end": float(physics[-1]),
        "physics_net": float(physics[-1] - physics[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR/typewell TVT path-search submission experiment.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--wells", nargs="+", default=list(DEFAULT_WELLS))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "submissions" / "gr_path_search.csv")
    parser.add_argument("--report", type=Path, default=DATA_DIR / "submissions" / "gr_path_search_report.csv")
    parser.add_argument("--tvt-range", type=float, default=140.0)
    parser.add_argument("--tvt-step", type=float, default=0.5)
    parser.add_argument("--transition-sigma", type=float, default=0.35)
    parser.add_argument("--transition-band-ft", type=float, default=4.0)
    parser.add_argument("--prior-sigma", type=float, default=45.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.25)
    parser.add_argument("--gr-sigmas", nargs="+", type=float, default=[0.0, 2.0, 6.0])
    parser.add_argument("--no-oracle-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_DIR, TRAIN_DIR, TEST_DIR, SUBS_DIR
    DATA_DIR = args.data_dir
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    SUBS_DIR = DATA_DIR / "submissions"

    results: list[PathResult] = []
    rows: list[dict[str, float | str | int]] = []
    for well_id in args.wells:
        print(f"[{well_id}] running GR path search...")
        result = predict_well(
            well_id=well_id,
            test_dir=TEST_DIR,
            tvt_range=args.tvt_range,
            tvt_step=args.tvt_step,
            transition_sigma=args.transition_sigma,
            prior_sigma=args.prior_sigma,
            transition_band_ft=args.transition_band_ft,
            gr_sigmas=tuple(args.gr_sigmas),
            smooth_sigma=args.smooth_sigma,
        )
        row = summarize_result(result)
        if not args.no_oracle_report:
            metrics = oracle_metrics(well_id, result)
            if metrics:
                row.update(metrics)
        rows.append(row)
        results.append(result)
        print(
            f"  pred=[{row['pred_start']:.1f}, {row['pred_end']:.1f}] "
            f"net={row['pred_net']:.1f} range={row['pred_range']:.1f} "
            f"sigma_gr={row['sigma_gr']:.1f}"
        )
        if "oracle_rmse_path" in row:
            print(
                f"  oracle RMSE path={row['oracle_rmse_path']:.3f} "
                f"physics={row['oracle_rmse_physics']:.3f} "
                f"anchor={row['oracle_rmse_anchor']:.3f}"
            )

    sub = build_submission(results, args.output)
    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    print(f"Saved submission: {args.output}")
    print(f"Saved report: {args.report}")
    print(f"Submission rows={len(sub)} NaN={sub['tvt'].isna().sum()}")


if __name__ == "__main__":
    main()
