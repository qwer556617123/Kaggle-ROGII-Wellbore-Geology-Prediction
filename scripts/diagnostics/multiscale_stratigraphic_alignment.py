"""Multiscale probabilistic stratigraphic alignment for ROGII wells.

The module reuses the exact forward-backward transition engine from the public
second-order HMM, but replaces its raw-GR emission with prefix-calibrated,
physical-scale stratigraphic descriptors.  It has no dependency on target-tail
TVT or formation/contact columns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "raw": {"raw": 1.0},
    "balanced": {
        "raw": 0.75,
        "dog_2_8": 0.10,
        "dog_8_24": 0.08,
        "dog_24_64": 0.05,
        "low_64": 0.02,
    },
    "coarse": {
        "raw": 0.50,
        "dog_2_8": 0.10,
        "dog_8_24": 0.15,
        "dog_24_64": 0.15,
        "low_64": 0.10,
    },
}


@dataclass(frozen=True)
class MPSCConfig:
    scales_ft: tuple[float, ...] = (2.0, 8.0, 24.0, 64.0)
    profile: str = "balanced"
    min_support: float = 0.60
    min_prefix_rows: int = 40
    student_df: float = 4.0
    emission_lambda: float = 1.0
    fine_step: float = 0.35
    coarse_step: float = 1.0
    fine_rates: int = 41
    coarse_rates: int = 21
    rate_span: float = 0.10
    sig_r: float = 0.002
    sig_p: float = 0.02
    momentum: float = 0.998
    start_sigma: float = 0.75
    rate_start_sigma: float = 0.01
    band_pad: float = 100.0
    corridor_half_width: float = 35.0
    top_modes: int = 5
    mode_separation_ft: float = 10.0
    use_coarse_corridor: bool = True
    raw_identity: bool = True
    horizontal_scale_mode: str = "prefix"
    path_resample_step_ft: float = 0.25
    path_min_step_ft: float = 0.02
    path_max_step_ft: float = 2.0
    anchor_corridor_half_width_ft: float = 12.0


@dataclass
class MPSCResult:
    pred: np.ndarray
    std_eval: np.ndarray
    ev_index: np.ndarray
    mean_eval: np.ndarray
    grid: np.ndarray
    post: np.ndarray
    metadata: dict[str, Any]


def _mad(values: np.ndarray, floor: float = 1e-3) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return floor
    center = float(np.median(values))
    return max(floor, 1.4826 * float(np.median(np.abs(values - center))))


def _huber_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    if len(x) < 3 or np.std(x) < 1e-8:
        bias = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, bias, float("inf"), int(len(x))
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(8):
        residual = y - design @ coef
        scale = _mad(residual)
        ratio = np.abs(residual) / max(2.5 * scale, 1e-9)
        weights = 1.0 / np.maximum(1.0, ratio)
        root = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    slope = float(np.clip(coef[0], 0.10, 10.0))
    bias = float(np.median(y - slope * x))
    residual = y - (slope * x + bias)
    return slope, bias, _mad(residual), int(len(x))


def _nan_gaussian(
    values: np.ndarray,
    sigma_rows: float,
    min_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    sigma_rows = max(float(sigma_rows), 0.35)
    numerator = gaussian_filter1d(filled, sigma=sigma_rows, mode="nearest")
    support = gaussian_filter1d(finite.astype(float), sigma=sigma_rows, mode="nearest")
    smooth = numerator / np.maximum(support, 1e-9)
    smooth[support < min_support] = np.nan
    return smooth, support


def multiscale_features(
    values: np.ndarray,
    spacing_ft: float,
    config: MPSCConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build deterministic DoG descriptors at physical thickness scales."""
    values = np.asarray(values, dtype=float)
    spacing = max(float(spacing_ft), 1e-6)
    finite = np.isfinite(values)
    raw = values.copy()
    smooth: dict[float, np.ndarray] = {}
    support: dict[float, np.ndarray] = {}
    for scale in config.scales_ft:
        smoothed, available = _nan_gaussian(
            values, sigma_rows=scale / spacing, min_support=config.min_support
        )
        smooth[float(scale)] = smoothed
        support[float(scale)] = available
    s2, s8, s24, s64 = (smooth[float(scale)] for scale in config.scales_ft)
    p2, p8, p24, p64 = (support[float(scale)] for scale in config.scales_ft)
    features = {
        "raw": raw,
        "dog_2_8": s2 - s8,
        "dog_8_24": s8 - s24,
        "dog_24_64": s24 - s64,
        "low_64": s64,
    }
    supports = {
        "raw": finite.astype(float),
        "dog_2_8": np.minimum(p2, p8),
        "dog_8_24": np.minimum(p8, p24),
        "dog_24_64": np.minimum(p24, p64),
        "low_64": p64,
    }
    for name in features:
        features[name] = np.asarray(features[name], dtype=float)
        features[name][supports[name] < config.min_support] = np.nan
    return features, supports


def _path_stratigraphic_coordinate(
    hw: pd.DataFrame,
    scale_path: np.ndarray,
    config: MPSCConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    path = np.asarray(scale_path, dtype=float).copy()
    if len(path) != len(hw) or not np.isfinite(path).all():
        raise ValueError("Scale path must be finite and row-aligned")
    known = hw["TVT_input"].to_numpy(dtype=float)
    known_mask = np.isfinite(known)
    path[known_mask] = known[known_mask]
    step = np.abs(np.diff(path))
    step = pd.Series(step).rolling(9, center=True, min_periods=1).median().to_numpy(float)
    unclipped = step.copy()
    step = np.clip(step, config.path_min_step_ft, config.path_max_step_ft)
    coordinate = np.concatenate([[0.0], np.cumsum(step)])
    return coordinate, {
        "path_distance_span_ft": float(coordinate[-1]),
        "path_step_median_ft": float(np.median(step)) if len(step) else 0.0,
        "path_step_p90_ft": float(np.quantile(step, 0.90)) if len(step) else 0.0,
        "path_step_clipped_fraction": float(
            np.mean(
                (unclipped < config.path_min_step_ft)
                | (unclipped > config.path_max_step_ft)
            )
        ) if len(step) else 0.0,
    }


def multiscale_features_on_path(
    values: np.ndarray,
    coordinate_ft: np.ndarray,
    config: MPSCConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract physical-scale descriptors on a nonuniform path-distance axis."""
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate_ft, dtype=float)
    if len(values) != len(coordinate) or np.any(np.diff(coordinate) < 0):
        raise ValueError("Path coordinate must be row-aligned and nondecreasing")
    step = max(float(config.path_resample_step_ft), 1e-3)
    grid = np.arange(float(coordinate[0]), float(coordinate[-1]) + step, step)
    index = np.clip(np.rint((coordinate - grid[0]) / step).astype(int), 0, len(grid) - 1)
    finite = np.isfinite(values)
    total = np.zeros(len(grid), dtype=float)
    count = np.zeros(len(grid), dtype=float)
    np.add.at(total, index[finite], values[finite])
    np.add.at(count, index[finite], 1.0)
    binned = np.full(len(grid), np.nan, dtype=float)
    occupied = count > 0
    binned[occupied] = total[occupied] / count[occupied]
    grid_features, grid_supports = multiscale_features(binned, step, config)
    features: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    for name in grid_features:
        valid_grid = np.isfinite(grid_features[name])
        if valid_grid.sum() < 2:
            features[name] = np.full(len(values), np.nan)
            supports[name] = np.zeros(len(values))
            continue
        features[name] = np.interp(
            coordinate,
            grid[valid_grid],
            grid_features[name][valid_grid],
            left=np.nan,
            right=np.nan,
        )
        supports[name] = np.interp(
            coordinate,
            grid,
            grid_supports[name],
            left=0.0,
            right=0.0,
        )
        features[name][supports[name] < config.min_support] = np.nan
    features["raw"] = values.copy()
    supports["raw"] = finite.astype(float)
    return features, supports


def _median_spacing(values: np.ndarray, default: float) -> float:
    values = np.asarray(values, dtype=float)
    delta = np.diff(values[np.isfinite(values)])
    delta = delta[delta > 0]
    return float(np.median(delta)) if len(delta) else default


def _horizontal_stratigraphic_spacing(hw: pd.DataFrame, default: float = 0.25) -> float:
    """Estimate TVT feet represented by one horizontal-log row from the prefix."""
    known = hw[["MD", "TVT_input"]].dropna()
    if len(known) < 4:
        return default
    md = known["MD"].to_numpy(dtype=float)
    tvt = known["TVT_input"].to_numpy(dtype=float)
    dm = np.diff(md)
    dtvt = np.abs(np.diff(tvt))
    valid = (dm > 0) & np.isfinite(dtvt)
    if int(valid.sum()) < 3:
        return default
    row_md = _median_spacing(hw["MD"].to_numpy(dtype=float), 1.0)
    spacing = float(np.median(dtvt[valid] / dm[valid]) * row_md)
    return float(np.clip(spacing, 0.02, 2.0))


def _prefix_calibrations(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    horizontal_features: dict[str, np.ndarray],
    typewell_features: dict[str, np.ndarray],
    names: list[str],
    config: MPSCConfig,
) -> tuple[dict[str, dict[str, float]], float]:
    known_tvt = hw["TVT_input"].to_numpy(dtype=float)
    finite_known = known_tvt[np.isfinite(known_tvt)]
    prefix_span = float(np.ptp(finite_known)) if len(finite_known) else 0.0
    minimum_spans = {
        "raw": 0.0,
        "dog_2_8": 16.0,
        "dog_8_24": 48.0,
        "dog_24_64": 128.0,
        "low_64": 128.0,
    }
    calibrations: dict[str, dict[str, float]] = {}
    losses = []
    for name in names:
        observed = horizontal_features[name]
        expected = np.interp(known_tvt, tw_tvt, typewell_features[name], left=np.nan, right=np.nan)
        valid = np.isfinite(known_tvt) & np.isfinite(observed) & np.isfinite(expected)
        if name == "raw" and config.raw_identity:
            known = np.isfinite(known_tvt) & np.isfinite(expected)
            source_observed = hw["GR"].fillna(0.0).to_numpy(dtype=float)
            rows = int(known.sum())
            residual = source_observed[known] - expected[known]
            sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0)) if rows else 30.0
            quality_valid = known & np.isfinite(hw["GR"].to_numpy(dtype=float))
            quality_residual = (
                hw["GR"].to_numpy(dtype=float)[quality_valid] - expected[quality_valid]
            )
            loss = float(
                np.sqrt(np.mean(np.square(quality_residual)))
                / max(_mad(hw["GR"].to_numpy(dtype=float)[quality_valid]), 1e-6)
            ) if int(quality_valid.sum()) else float("inf")
            calibrations[name] = {
                "slope": 1.0,
                "bias": 0.0,
                "sigma": sigma,
                "rows": rows,
                "loss": loss,
                "active": rows >= config.min_prefix_rows,
                "correlation": float("nan"),
                "reason": "source_hmm_identity_calibration",
            }
            if rows >= config.min_prefix_rows:
                losses.append(loss)
            continue
        if prefix_span < minimum_spans.get(name, 0.0):
            calibrations[name] = {
                "slope": 1.0,
                "bias": 0.0,
                "sigma": float("inf"),
                "rows": int(valid.sum()),
                "loss": float("inf"),
                "active": False,
                "reason": "insufficient_prefix_span",
            }
            continue
        # Calibrate the reference response into horizontal-log amplitude space.
        # Regressing the noisy horizontal trace onto the reference attenuates
        # the slope severely, especially for broad low-frequency descriptors.
        slope, bias, sigma, rows = _huber_affine(expected[valid], observed[valid])
        correlation = (
            float(np.corrcoef(observed[valid], expected[valid])[0, 1])
            if int(valid.sum()) >= 3
            else float("nan")
        )
        if (
            rows < config.min_prefix_rows
            or not np.isfinite(sigma)
            or not np.isfinite(correlation)
            or correlation < 0.10
        ):
            calibrations[name] = {
                "slope": 1.0,
                "bias": 0.0,
                "sigma": float("inf"),
                "rows": int(rows),
                "loss": float("inf"),
                "active": False,
                "correlation": correlation,
                "reason": "weak_prefix_calibration",
            }
            continue
        calibrated_expected = slope * expected[valid] + bias
        residual = observed[valid] - calibrated_expected
        loss = float(
            np.sqrt(np.mean(np.square(residual))) / max(_mad(observed[valid]), 1e-6)
        )
        calibrations[name] = {
            "slope": slope,
            "bias": bias,
            "sigma": max(float(sigma), 1e-3),
            "rows": int(rows),
            "loss": loss,
            "active": True,
            "correlation": correlation,
        }
        losses.append(loss)
    return calibrations, float(np.mean(losses)) if losses else float("inf")


def reference_prefix_loss(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    config: MPSCConfig | None = None,
) -> dict[str, float]:
    """Truth-free reference compatibility used for analog selection."""
    config = config or MPSCConfig()
    tw_valid = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    known = hw["TVT_input"].notna() & hw["GR"].notna()
    if len(tw_valid) < 20 or int(known.sum()) < config.min_prefix_rows:
        return {"loss": float("inf"), "rows": 0, "rmse": float("inf")}
    expected = np.interp(
        hw.loc[known, "TVT_input"].to_numpy(dtype=float),
        tw_valid["TVT"].to_numpy(dtype=float),
        tw_valid["GR"].to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )
    observed = hw.loc[known, "GR"].to_numpy(dtype=float)
    valid = np.isfinite(expected) & np.isfinite(observed)
    if int(valid.sum()) < config.min_prefix_rows:
        return {"loss": float("inf"), "rows": int(valid.sum()), "rmse": float("inf")}
    slope, bias, sigma, rows = _huber_affine(expected[valid], observed[valid])
    residual = observed[valid] - (slope * expected[valid] + bias)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    loss = float(rmse / max(_mad(observed[valid]), 1e-6))
    return {"loss": loss, "rows": int(rows), "rmse": rmse}


def _emission_matrix(
    eval_index: np.ndarray,
    grid: np.ndarray,
    tw_tvt: np.ndarray,
    horizontal_features: dict[str, np.ndarray],
    typewell_features: dict[str, np.ndarray],
    calibrations: dict[str, dict[str, float]],
    weights: dict[str, float],
    config: MPSCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(eval_index)
    emission = np.zeros((rows, len(grid)), dtype=np.float64)
    used = np.zeros(rows, dtype=np.float64)
    for name, weight in weights.items():
        calibration = calibrations.get(name, {})
        if not calibration.get("active", False) or weight <= 0:
            continue
        observed = horizontal_features[name][eval_index]
        expected = np.interp(grid, tw_tvt, typewell_features[name], left=np.nan, right=np.nan)
        expected = float(calibration["slope"]) * expected + float(calibration["bias"])
        valid_rows = np.isfinite(observed)
        valid_grid = np.isfinite(expected)
        if not valid_rows.any() or not valid_grid.any():
            continue
        z = (observed[:, None] - expected[None, :]) / float(calibration["sigma"])
        component = -0.5 * (config.student_df + 1.0) * np.log1p(
            np.square(z) / config.student_df
        )
        component[:, ~valid_grid] = 0.0
        component[~valid_rows, :] = 0.0
        emission += float(weight) * component
        used[valid_rows] += float(weight)
    active = used > 0
    emission[active] /= used[active, None]
    emission[~active] = 0.0
    return (config.emission_lambda * emission).astype(np.float32), used


def _geometry(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    step: float,
    n_rates: int,
    config: MPSCConfig,
) -> dict[str, Any]:
    known = hw[hw["TVT_input"].notna()]
    eval_frame = hw[hw["TVT_input"].isna()]
    if known.empty:
        raise ValueError("MPSC requires a known TVT_input prefix")
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    gmin = max(float(tw_tvt.min()) - 40.0, last_tvt - config.band_pad)
    gmax = min(float(tw_tvt.max()) + 40.0, last_tvt + config.band_pad)
    grid = np.arange(gmin, gmax + step, step)
    md = eval_frame["MD"].to_numpy(dtype=float)
    z = eval_frame["Z"].to_numpy(dtype=float)
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    tail = known.tail(30)
    tail_dm = np.diff(tail["MD"].to_numpy(dtype=float))
    tail_du = np.diff(
        tail["TVT_input"].to_numpy(dtype=float) + tail["Z"].to_numpy(dtype=float)
    )
    valid = tail_dm > 0
    initial_rate = float(np.median(tail_du[valid] / tail_dm[valid])) if valid.sum() >= 3 else 0.0
    span = max(config.rate_span, abs(initial_rate) + 0.04)
    rates = np.linspace(-span, span, n_rates)
    return {
        "eval_index": eval_frame.index.to_numpy(dtype=int),
        "grid": grid,
        "dm": dm,
        "dz": dz,
        "rates": rates,
        "start_position": float((last_tvt - gmin) / step),
        "initial_rate": initial_rate,
    }


def _run_fb(
    emission: np.ndarray,
    geometry: dict[str, Any],
    step: float,
    fb_engine: Callable[..., tuple[np.ndarray, float]],
    config: MPSCConfig,
) -> tuple[np.ndarray, float]:
    return fb_engine(
        emission,
        geometry["dm"].astype(np.float64),
        geometry["dz"].astype(np.float64),
        float(step),
        geometry["rates"].astype(np.float64),
        float(config.sig_r),
        float(config.sig_p),
        float(geometry["start_position"]),
        float(config.start_sigma),
        float(geometry["initial_rate"]),
        float(config.rate_start_sigma),
        1.0,
        float(config.momentum),
    )


def _row_mode_centers(
    post: np.ndarray,
    grid: np.ndarray,
    top_modes: int,
    separation_ft: float,
) -> list[np.ndarray]:
    centers: list[np.ndarray] = []
    for row in post:
        selected: list[float] = []
        for index in np.argsort(row)[::-1]:
            value = float(grid[index])
            if all(abs(value - current) >= separation_ft for current in selected):
                selected.append(value)
            if len(selected) >= top_modes:
                break
        centers.append(np.asarray(selected, dtype=float))
    return centers


def _posterior_summary(post: np.ndarray, grid: np.ndarray, config: MPSCConfig) -> dict[str, Any]:
    if not len(post):
        return {"mode_datums": [], "mode_weights": [], "entropy": 0.0}
    aggregate = np.mean(post, axis=0)
    indices = []
    for index in np.argsort(aggregate)[::-1]:
        if all(abs(float(grid[index] - grid[current])) >= config.mode_separation_ft for current in indices):
            indices.append(int(index))
        if len(indices) >= config.top_modes:
            break
    weights = np.asarray([aggregate[index] for index in indices], dtype=float)
    if weights.sum() > 0:
        weights /= weights.sum()
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))) if len(weights) else 0.0
    if len(weights) > 1:
        entropy /= float(np.log(len(weights)))
    return {
        "mode_datums": [float(grid[index]) for index in indices],
        "mode_weights": [float(value) for value in weights],
        "entropy": entropy,
    }


def run_mpsc(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    fb_engine: Callable[..., tuple[np.ndarray, float]],
    config: MPSCConfig | None = None,
    scale_path: np.ndarray | None = None,
) -> MPSCResult:
    config = config or MPSCConfig()
    if config.profile not in PROFILE_WEIGHTS:
        raise ValueError(f"Unknown MPSC profile: {config.profile}")
    required = {"MD", "Z", "GR", "TVT_input"}
    if not required.issubset(hw.columns):
        raise ValueError(f"Missing horizontal columns: {sorted(required - set(hw.columns))}")
    tw_valid = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    if len(tw_valid) < 20:
        raise ValueError("Typewell has too few valid rows")
    tw_tvt = tw_valid["TVT"].to_numpy(dtype=float)
    tw_gr = tw_valid["GR"].to_numpy(dtype=float)
    output = hw["TVT_input"].to_numpy(dtype=float).copy()
    eval_index = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
    if not len(eval_index):
        metadata = {**asdict(config), "status": "no_eval", "profile": config.profile}
        return MPSCResult(output, np.empty(0), eval_index, np.empty(0), np.empty(0), np.empty((0, 0)), metadata)

    if config.horizontal_scale_mode not in {"prefix", "anchor_path"}:
        raise ValueError(f"Unknown horizontal scale mode: {config.horizontal_scale_mode}")
    horizontal_spacing = _horizontal_stratigraphic_spacing(hw)
    typewell_spacing = _median_spacing(tw_tvt, 0.5)
    path_metadata: dict[str, float] = {}
    if config.horizontal_scale_mode == "anchor_path":
        if scale_path is None:
            raise ValueError("anchor_path scale mode requires scale_path")
        path_coordinate, path_metadata = _path_stratigraphic_coordinate(
            hw, scale_path, config
        )
        horizontal_features, _ = multiscale_features_on_path(
            hw["GR"].to_numpy(dtype=float), path_coordinate, config
        )
    else:
        horizontal_features, _ = multiscale_features(
            hw["GR"].to_numpy(dtype=float), horizontal_spacing, config
        )
    # Preserve the exact public-HMM missing-value behavior for the control
    # channel.  Multiscale descriptors retain their support-aware masks.
    horizontal_features["raw"] = (
        hw["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(dtype=float)
    )
    typewell_features, _ = multiscale_features(tw_gr, typewell_spacing, config)
    weights = PROFILE_WEIGHTS[config.profile]
    names = list(weights)
    calibrations, prefix_loss = _prefix_calibrations(
        hw, tw_tvt, horizontal_features, typewell_features, names, config
    )

    coarse_loglik = float("nan")
    corridor_fraction = 1.0
    coarse_modes: list[np.ndarray] | None = None
    if config.use_coarse_corridor and config.profile != "raw":
        coarse_geometry = _geometry(hw, tw_tvt, config.coarse_step, config.coarse_rates, config)
        coarse_weights = {"dog_24_64": 0.65, "low_64": 0.35}
        coarse_emission, _ = _emission_matrix(
            coarse_geometry["eval_index"],
            coarse_geometry["grid"],
            tw_tvt,
            horizontal_features,
            typewell_features,
            calibrations,
            coarse_weights,
            config,
        )
        coarse_post, coarse_loglik = _run_fb(
            coarse_emission, coarse_geometry, config.coarse_step, fb_engine, config
        )
        coarse_modes = _row_mode_centers(
            coarse_post,
            coarse_geometry["grid"],
            config.top_modes,
            config.mode_separation_ft,
        )
        # Never let an ambiguous low-frequency correlation remove the
        # heel-continuous geological path from the fine search corridor.
        heel_path = (
            float(coarse_geometry["grid"][0])
            + config.coarse_step * float(coarse_geometry["start_position"])
            + np.cumsum(
                coarse_geometry["initial_rate"] * coarse_geometry["dm"]
                - coarse_geometry["dz"]
            )
        )
        for row, value in enumerate(heel_path):
            coarse_modes[row] = np.append(coarse_modes[row], float(value))

    geometry = _geometry(hw, tw_tvt, config.fine_step, config.fine_rates, config)
    emission, used = _emission_matrix(
        geometry["eval_index"],
        geometry["grid"],
        tw_tvt,
        horizontal_features,
        typewell_features,
        calibrations,
        weights,
        config,
    )
    if (
        config.horizontal_scale_mode == "anchor_path"
        and scale_path is not None
        and config.anchor_corridor_half_width_ft > 0
    ):
        anchor_center = np.asarray(scale_path, dtype=float)[geometry["eval_index"]]
        anchor_corridor = (
            np.abs(geometry["grid"][None, :] - anchor_center[:, None])
            <= config.anchor_corridor_half_width_ft
        )
        emission[~anchor_corridor] = -60.0
        corridor_fraction = float(np.mean(anchor_corridor))
    if coarse_modes is not None:
        corridor = np.zeros_like(emission, dtype=bool)
        fine_grid = geometry["grid"]
        for row, centers in enumerate(coarse_modes):
            for center in centers:
                corridor[row] |= np.abs(fine_grid - center) <= config.corridor_half_width
        empty = ~corridor.any(axis=1)
        corridor[empty] = True
        emission[~corridor] = -60.0
        corridor_fraction = float(np.mean(corridor))
    post, loglik = _run_fb(emission, geometry, config.fine_step, fb_engine, config)
    grid = geometry["grid"]
    mean = post @ grid
    variance = post @ np.square(grid) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    output[geometry["eval_index"]] = mean
    summary = _posterior_summary(post, grid, config)
    metadata: dict[str, Any] = {
        **asdict(config),
        **summary,
        **path_metadata,
        "status": "ok",
        "profile": config.profile,
        "prefix_loss": prefix_loss,
        "eval_rows": int(len(mean)),
        "gr_coverage": float(np.mean(np.isfinite(hw["GR"].to_numpy(dtype=float)[eval_index]))),
        "active_features": [name for name in names if calibrations[name]["active"]],
        "calibrations": calibrations,
        "coarse_loglik": coarse_loglik,
        "fine_loglik": float(loglik),
        "corridor_fraction": corridor_fraction,
        "horizontal_spacing_ft": horizontal_spacing,
        "typewell_spacing_ft": typewell_spacing,
    }
    return MPSCResult(
        pred=output,
        std_eval=std,
        ev_index=geometry["eval_index"],
        mean_eval=mean,
        grid=grid,
        post=post,
        metadata=metadata,
    )


def mix_reference_results(
    own: MPSCResult,
    analog: MPSCResult,
    own_prior: float = 0.75,
    loss_temperature: float = 0.10,
) -> MPSCResult:
    """Posterior-moment mixture for own and prefix-compatible analog references."""
    if not np.array_equal(own.ev_index, analog.ev_index):
        raise ValueError("Reference results use different eval rows")
    losses = np.asarray(
        [own.metadata.get("prefix_loss", np.inf), analog.metadata.get("prefix_loss", np.inf)],
        dtype=float,
    )
    priors = np.asarray([own_prior, 1.0 - own_prior], dtype=float)
    if not np.isfinite(losses[1]):
        weights = np.asarray([1.0, 0.0])
    else:
        logits = -(losses - np.nanmin(losses)) / max(loss_temperature, 1e-6)
        logits += np.log(np.maximum(priors, 1e-12))
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
    mean = weights[0] * own.mean_eval + weights[1] * analog.mean_eval
    second = weights[0] * (np.square(own.std_eval) + np.square(own.mean_eval))
    second += weights[1] * (np.square(analog.std_eval) + np.square(analog.mean_eval))
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    output = own.pred.copy()
    output[own.ev_index] = mean
    metadata = dict(own.metadata)
    metadata.update(
        {
            "reference_mode": "own_analog_mixture",
            "own_weight": float(weights[0]),
            "analog_weight": float(weights[1]),
            "analog_prefix_loss": float(losses[1]),
            "entropy": float(
                -np.sum(weights * np.log(np.maximum(weights, 1e-12))) / np.log(2.0)
            ),
        }
    )
    return MPSCResult(output, std, own.ev_index.copy(), mean, own.grid, own.post, metadata)
