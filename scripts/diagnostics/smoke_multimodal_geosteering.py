"""Synthetic smoke tests for deterministic multimodal geosteering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal_geosteering import PosteriorConfig, predict_multimodal


def make_typewell(periodic: bool = False) -> pd.DataFrame:
    tvt = np.arange(0.0, 220.0, 0.5)
    if periodic:
        gr = 80.0 + 25.0 * np.sin(2.0 * np.pi * tvt / 20.0)
    else:
        gr = 70.0 + 0.18 * tvt + 18.0 * np.sin(tvt / 13.0) + 9.0 * np.cos(tvt / 5.0)
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def make_horizontal(tw: pd.DataFrame, fault: float = 0.0, missing: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    n = 900
    md = np.arange(n, dtype=float)
    z = 1200.0 + 0.018 * md
    u = 1280.0 + 0.004 * md
    if fault:
        u[600:] += fault
    tvt = u - z
    gr = np.interp(tvt, tw["TVT"], tw["GR"])
    gr = (gr - 4.0) / 1.15
    known = np.arange(n) < 260
    if missing:
        gr[~known] = np.nan
    else:
        gr[520:610] = np.nan
    hw = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": np.where(known, tvt, np.nan),
        }
    )
    return hw, ~known


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    cfg = PosteriorConfig(temperature=0.08)

    unique_tw = make_typewell(periodic=False)
    unique_hw, eval_mask = make_horizontal(unique_tw)
    unique = predict_multimodal(unique_hw, unique_tw, config=cfg, alpha=1.0)
    assert np.isfinite(unique.posterior).all()
    assert np.max(np.abs(unique.posterior[eval_mask] - unique.prior[eval_mask])) <= cfg.max_shift + 1e-6
    unique_rmse = rmse(unique_hw.loc[eval_mask, "TVT"], unique.posterior[eval_mask])

    periodic_tw = make_typewell(periodic=True)
    periodic_hw, periodic_mask = make_horizontal(periodic_tw)
    periodic = predict_multimodal(periodic_hw, periodic_tw, config=cfg, alpha=1.0)
    assert periodic.metadata["mode_count"] >= 2
    assert periodic.metadata["entropy"] > 0.05

    missing_hw, missing_mask = make_horizontal(unique_tw, missing=True)
    missing = predict_multimodal(missing_hw, unique_tw, config=cfg)
    assert np.isfinite(missing.posterior).all()
    assert missing.metadata["gr_coverage"] < 0.05
    assert missing.metadata["alpha"] == 0.0

    fault_hw, fault_mask = make_horizontal(unique_tw, fault=15.0)
    fault = predict_multimodal(fault_hw, unique_tw, config=cfg, alpha=1.0)
    prior_fault_rmse = rmse(fault_hw.loc[fault_mask, "TVT"], fault.prior[fault_mask])
    map_fault_rmse = rmse(fault_hw.loc[fault_mask, "TVT"], fault.map_path[fault_mask])
    assert map_fault_rmse < prior_fault_rmse
    assert "fault" in fault.metadata["map_name"] or map_fault_rmse < 0.75 * prior_fault_rmse

    print(
        {
            "unique_rmse": unique_rmse,
            "unique_map": unique.metadata["map_name"],
            "periodic_entropy": periodic.metadata["entropy"],
            "periodic_modes": periodic.metadata["mode_datums"][:4],
            "missing_alpha": missing.metadata["alpha"],
            "fault_prior_rmse": prior_fault_rmse,
            "fault_map_rmse": map_fault_rmse,
            "fault_map": fault.metadata["map_name"],
        }
    )


if __name__ == "__main__":
    main()
