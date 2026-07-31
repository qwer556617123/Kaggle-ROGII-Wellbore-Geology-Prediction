"""Synthetic smoke tests for the fault-aware residual smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fault_aware_smoother import FaultSmootherConfig, smooth_fault_residual


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def make_case(fault: float = 0.0, missing: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = 360
    known_n = 120
    md = np.arange(n, dtype=float)
    x = 1000.0 + md
    y = 2000.0 + 0.18 * md + 8.0 * np.sin(md / 80.0)
    z = -9000.0 - 0.03 * md
    base = 11000.0 + 0.018 * md - z - 9000.0
    true = base.copy()
    true[known_n + 70 :] += fault
    tw_tvt = np.arange(10940.0, 11180.0, 0.5)
    tw_gr = 75.0 + 18.0 * np.sin(tw_tvt / 5.7) + 9.0 * np.cos(tw_tvt / 13.0)
    gr = 1.15 * np.interp(true, tw_tvt, tw_gr) + 4.0
    if missing:
        gr[known_n + 30 : known_n + 80] = np.nan
    tvt_input = np.where(np.arange(n) < known_n, true, np.nan)
    hw = pd.DataFrame(
        {"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT_input": tvt_input, "TVT": true}
    )
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    anchor = base.copy()
    anchor[:known_n] = true[:known_n]
    return hw, tw, anchor


def main() -> None:
    cfg = FaultSmootherConfig(
        max_residual=30.0,
        grid_step=0.5,
        fault_hazard=0.01,
        feature_windows=(1, 5, 17),
    )
    smooth_hw, tw, smooth_anchor = make_case()
    smooth = smooth_fault_residual(smooth_hw, tw, smooth_anchor, cfg, allow_faults=True)
    eval_mask = smooth_hw["TVT_input"].isna().to_numpy()
    assert np.isfinite(smooth.posterior).all()
    assert np.max(np.abs(smooth.posterior[eval_mask] - smooth_anchor[eval_mask])) <= cfg.max_residual + 1e-6

    fault_hw, tw, fault_anchor = make_case(fault=15.0)
    no_fault = smooth_fault_residual(fault_hw, tw, fault_anchor, cfg, allow_faults=False)
    with_fault = smooth_fault_residual(fault_hw, tw, fault_anchor, cfg, allow_faults=True)
    truth = fault_hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
    anchor_score = rmse(truth, fault_anchor[eval_mask])
    no_fault_score = rmse(truth, no_fault.posterior[eval_mask])
    fault_score = rmse(truth, with_fault.posterior[eval_mask])
    assert fault_score < anchor_score
    assert fault_score <= no_fault_score + 0.25
    assert np.max(with_fault.fault_probability) > 0

    missing_hw, tw, missing_anchor = make_case(fault=-15.0, missing=True)
    missing = smooth_fault_residual(missing_hw, tw, missing_anchor, cfg, allow_faults=True)
    assert np.isfinite(missing.posterior).all()
    print(
        {
            "smooth_mean_move": smooth.metadata["residual_mean_abs"],
            "fault_anchor_rmse": anchor_score,
            "fault_no_fault_rmse": no_fault_score,
            "fault_posterior_rmse": fault_score,
            "fault_probability_max": with_fault.metadata["fault_probability_max"],
            "missing_finite": True,
        }
    )


if __name__ == "__main__":
    main()
