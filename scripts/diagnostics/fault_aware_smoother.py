"""Fault-aware posterior smoothing of residuals around an existing TVT path."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class FaultSmootherConfig:
    max_residual: float = 40.0
    grid_step: float = 0.5
    smooth_sigma: float = 0.7
    fault_sigma: float = 1.5
    fault_hazard: float = 0.0008
    fault_jumps: tuple[float, ...] = (
        -35.0,
        -30.0,
        -25.0,
        -20.0,
        -15.0,
        -10.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
    )
    student_df: float = 4.0
    emission_temperature: float = 1.0
    anchor_prior_sigma: float = 12.0
    anchor_prior_strength: float = 0.004
    prefix_tail_rows: int = 240
    max_lag_rows: int = 4
    response_windows: tuple[int, ...] = (1, 5, 11)
    feature_windows: tuple[int, ...] = (1, 5, 17, 49)
    min_prefix_rows: int = 40
    drift_clip_per_row: float = 0.08


@dataclass
class FaultSmootherResult:
    posterior: np.ndarray
    map_path: np.ndarray
    posterior_std: np.ndarray
    fault_probability: np.ndarray
    metadata: dict[str, Any]


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return (
        series.interpolate(limit_direction="both")
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _mad(values: np.ndarray, floor: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return floor
    center = float(np.median(values))
    return max(floor, 1.4826 * float(np.median(np.abs(values - center))))


def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    if len(x) < 3 or np.std(x) < 1e-8:
        bias = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, bias, 30.0
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(6):
        residual = y - design @ coef
        scale = _mad(residual)
        weight = 1.0 / np.maximum(1.0, np.abs(residual) / (2.5 * scale))
        weighted = design * np.sqrt(weight)[:, None]
        coef, *_ = np.linalg.lstsq(weighted, y * np.sqrt(weight), rcond=None)
    slope = float(np.clip(coef[0], 0.2, 5.0))
    bias = float(np.median(y - slope * x))
    sigma = float(np.clip(_mad(y - (slope * x + bias)), 6.0, 60.0))
    return slope, bias, sigma


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if lag == 0:
        return values.copy()
    out = np.empty_like(values)
    if lag > 0:
        out[:lag] = values[0]
        out[lag:] = values[:-lag]
    else:
        out[lag:] = values[-1]
        out[:lag] = values[-lag:]
    return out


def _calibrate_gr(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: FaultSmootherConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    raw = hw["GR"].to_numpy(dtype=float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(raw))
    if len(known) < config.min_prefix_rows:
        filled = pd.Series(raw).interpolate(limit_direction="both").fillna(np.nanmean(tw_gr))
        return filled.to_numpy(dtype=float), {
            "lag_rows": 0,
            "response_window": 1,
            "prefix_rmse": float("inf"),
            "slope": 1.0,
            "bias": 0.0,
            "sigma": 30.0,
        }
    known = known[-min(config.prefix_tail_rows, len(known)) :]
    best: tuple[float, np.ndarray, dict[str, float]] | None = None
    for window in config.response_windows:
        smoothed = _rolling(raw, window)
        for lag in range(-config.max_lag_rows, config.max_lag_rows + 1):
            observed = _shift(smoothed, lag)
            expected = np.interp(tvt_input[known], tw_tvt, tw_gr)
            slope, bias, sigma = _robust_affine(expected, observed[known])
            residual = observed[known] - (slope * expected + bias)
            rmse = float(np.sqrt(np.mean(residual * residual)))
            payload = {
                "lag_rows": int(lag),
                "response_window": int(window),
                "prefix_rmse": rmse,
                "slope": slope,
                "bias": bias,
                "sigma": sigma,
            }
            if best is None or rmse < best[0]:
                best = (rmse, observed, payload)
    assert best is not None
    return best[1], best[2]


def _residual_drift(hw: pd.DataFrame, anchor: np.ndarray, config: FaultSmootherConfig) -> np.ndarray:
    x = hw["X"].to_numpy(dtype=float)
    y = hw["Y"].to_numpy(dtype=float)
    z = hw["Z"].to_numpy(dtype=float)
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input))
    eval_idx = np.flatnonzero(~np.isfinite(tvt_input))
    if len(known_idx) < 20 or len(eval_idx) < 2:
        return np.zeros(len(eval_idx), dtype=float)
    tail = known_idx[-min(config.prefix_tail_rows, len(known_idx)) :]
    early = eval_idx[: min(config.prefix_tail_rows, len(eval_idx))]

    def gradient(indices: np.ndarray, u: np.ndarray) -> np.ndarray:
        xx = x[indices] - np.mean(x[indices])
        yy = y[indices] - np.mean(y[indices])
        design = np.column_stack([xx, yy, np.ones(len(indices))])
        ridge = np.diag([1e-3, 1e-3, 0.0])
        coef = np.linalg.solve(design.T @ design + ridge, design.T @ u[indices])
        return coef[:2]

    true_u = np.where(np.isfinite(tvt_input), tvt_input, anchor) + z
    anchor_u = anchor + z
    try:
        residual_gradient = gradient(tail, true_u) - gradient(early, anchor_u)
    except np.linalg.LinAlgError:
        residual_gradient = np.zeros(2, dtype=float)
    dx = np.diff(np.r_[x[known_idx[-1]], x[eval_idx]])
    dy = np.diff(np.r_[y[known_idx[-1]], y[eval_idx]])
    drift = residual_gradient[0] * dx + residual_gradient[1] * dy
    return np.clip(drift, -config.drift_clip_per_row, config.drift_clip_per_row)


def _transition_kernel(
    grid: np.ndarray,
    drift: float,
    config: FaultSmootherConfig,
    allow_faults: bool,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.arange(-(len(grid) - 1), len(grid), dtype=float) * config.grid_step
    smooth = np.exp(-0.5 * ((delta - drift) / config.smooth_sigma) ** 2)
    smooth /= max(float(smooth.sum()), 1e-15)
    if not allow_faults or config.fault_hazard <= 0:
        return smooth, np.zeros_like(smooth)
    fault = np.zeros_like(delta)
    for jump in config.fault_jumps:
        fault += np.exp(-0.5 * ((delta - drift - jump) / config.fault_sigma) ** 2)
    fault /= max(float(fault.sum()), 1e-15)
    kernel = (1.0 - config.fault_hazard) * smooth + config.fault_hazard * fault
    kernel /= max(float(kernel.sum()), 1e-15)
    return kernel, fault


def _apply_transition(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a delta-indexed kernel and preserve the residual state grid."""
    values = np.asarray(values, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    full = np.convolve(values, kernel, mode="full")
    start = len(values) - 1
    return full[start : start + len(values)]


def _shift_states(values: np.ndarray, shift: float) -> np.ndarray:
    index = np.arange(len(values), dtype=float)
    return np.interp(index - shift, index, values, left=0.0, right=0.0)


def _apply_mixture(
    values: np.ndarray,
    drift: float,
    config: FaultSmootherConfig,
    allow_faults: bool,
    reverse: bool = False,
    fault_only: bool = False,
) -> np.ndarray:
    direction = -1.0 if reverse else 1.0
    smooth_values = gaussian_filter1d(
        np.asarray(values, dtype=float),
        sigma=max(config.smooth_sigma / config.grid_step, 0.35),
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    smooth = _shift_states(smooth_values, direction * drift / config.grid_step)
    if not allow_faults or config.fault_hazard <= 0:
        return np.zeros_like(smooth) if fault_only else smooth
    fault_values = gaussian_filter1d(
        np.asarray(values, dtype=float),
        sigma=max(config.fault_sigma / config.grid_step, 0.35),
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    fault = np.zeros_like(fault_values)
    for jump in config.fault_jumps:
        shift = direction * (drift + jump) / config.grid_step
        fault += _shift_states(fault_values, shift)
    fault /= max(len(config.fault_jumps), 1)
    if fault_only:
        return fault
    return (1.0 - config.fault_hazard) * smooth + config.fault_hazard * fault


def _transition_tables(
    drift: np.ndarray,
    config: FaultSmootherConfig,
    allow_faults: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[tuple[int, float]]] = []
    fault_rows: list[list[tuple[int, float]]] = []
    smooth_radius = max(1, int(np.ceil(3.0 * config.smooth_sigma / config.grid_step)))
    smooth_offsets = np.arange(-smooth_radius, smooth_radius + 1)
    for value in np.asarray(drift, dtype=float):
        components: dict[int, float] = {}
        center = value / config.grid_step
        base = int(np.floor(center))
        smooth_weight = np.exp(
            -0.5
            * ((smooth_offsets + base - center) * config.grid_step / config.smooth_sigma)
            ** 2
        )
        smooth_weight /= smooth_weight.sum()
        for offset, weight in zip(smooth_offsets + base, smooth_weight):
            components[int(offset)] = components.get(int(offset), 0.0) + (
                (1.0 - config.fault_hazard) * float(weight)
                if allow_faults
                else float(weight)
            )

        fault_components: dict[int, float] = {}
        if allow_faults and config.fault_hazard > 0:
            per_jump = 1.0 / max(len(config.fault_jumps), 1)
            for jump in config.fault_jumps:
                fault_center = (value + jump) / config.grid_step
                lower = int(np.floor(fault_center))
                fraction = float(fault_center - lower)
                local = (
                    (lower - 1, 0.15 * (1.0 - fraction)),
                    (lower, 0.70 * (1.0 - fraction) + 0.15 * fraction),
                    (lower + 1, 0.15 * (1.0 - fraction) + 0.70 * fraction),
                    (lower + 2, 0.15 * fraction),
                )
                for shift, weight in local:
                    value_weight = per_jump * weight
                    fault_components[shift] = fault_components.get(shift, 0.0) + value_weight
                    components[shift] = components.get(shift, 0.0) + (
                        config.fault_hazard * value_weight
                    )
        total = sum(components.values())
        rows.append(sorted((shift, weight / total) for shift, weight in components.items()))
        fault_total = sum(fault_components.values())
        fault_rows.append(
            sorted(
                (shift, weight / fault_total)
                for shift, weight in fault_components.items()
            )
            if fault_total > 0
            else []
        )

    width = max((len(row) for row in rows), default=1)
    fault_width = max((len(row) for row in fault_rows), default=1)
    shifts = np.zeros((len(rows), width), dtype=np.int16)
    weights = np.zeros((len(rows), width), dtype=np.float32)
    fault_shifts = np.zeros((len(rows), fault_width), dtype=np.int16)
    fault_weights = np.zeros((len(rows), fault_width), dtype=np.float32)
    for index, row in enumerate(rows):
        for column, (shift, weight) in enumerate(row):
            shifts[index, column] = shift
            weights[index, column] = weight
    for index, row in enumerate(fault_rows):
        for column, (shift, weight) in enumerate(row):
            fault_shifts[index, column] = shift
            fault_weights[index, column] = weight
    return shifts, weights, fault_shifts, fault_weights


@njit(cache=True, nogil=True)
def _sparse_forward_backward(
    emission: np.ndarray,
    prior: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    fault_shifts: np.ndarray,
    fault_weights: np.ndarray,
    fault_hazard: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_states = emission.shape
    forward = np.zeros((n_rows, n_states), dtype=np.float64)
    previous = prior.copy()
    fault_probability = np.zeros(n_rows, dtype=np.float64)

    for t in range(n_rows):
        predicted = np.zeros(n_states, dtype=np.float64)
        fault_predicted = np.zeros(n_states, dtype=np.float64)
        for source in range(n_states):
            value = previous[source]
            if value <= 0.0:
                continue
            for h in range(shifts.shape[1]):
                weight = weights[t, h]
                if weight <= 0.0:
                    continue
                target = source + shifts[t, h]
                if 0 <= target < n_states:
                    predicted[target] += value * weight
            if t > 0 and fault_hazard > 0.0:
                for h in range(fault_shifts.shape[1]):
                    weight = fault_weights[t, h]
                    if weight <= 0.0:
                        continue
                    target = source + fault_shifts[t, h]
                    if 0 <= target < n_states:
                        fault_predicted[target] += value * weight
        norm = 0.0
        fault_evidence = 0.0
        for state in range(n_states):
            predicted[state] *= emission[t, state]
            norm += predicted[state]
            if t > 0:
                fault_evidence += fault_predicted[state] * emission[t, state]
        if norm <= 1e-300:
            norm = 1e-300
        for state in range(n_states):
            forward[t, state] = predicted[state] / norm
        if t > 0:
            fault_probability[t] = min(1.0, fault_hazard * fault_evidence / norm)
        previous = forward[t].copy()

    backward = np.ones((n_rows, n_states), dtype=np.float64)
    posterior = np.zeros((n_rows, n_states), dtype=np.float64)
    for t in range(n_rows - 2, -1, -1):
        norm = 0.0
        for source in range(n_states):
            total = 0.0
            for h in range(shifts.shape[1]):
                weight = weights[t + 1, h]
                if weight <= 0.0:
                    continue
                target = source + shifts[t + 1, h]
                if 0 <= target < n_states:
                    total += weight * emission[t + 1, target] * backward[t + 1, target]
            backward[t, source] = total
            norm += total
        if norm <= 1e-300:
            norm = 1e-300
        for state in range(n_states):
            backward[t, state] /= norm

    for t in range(n_rows):
        norm = 0.0
        for state in range(n_states):
            posterior[t, state] = forward[t, state] * backward[t, state]
            norm += posterior[t, state]
        if norm <= 1e-300:
            norm = 1e-300
        for state in range(n_states):
            posterior[t, state] /= norm
    return posterior, fault_probability


def _emission_matrix(
    observed: np.ndarray,
    eval_idx: np.ndarray,
    anchor: np.ndarray,
    grid: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    calibration: dict[str, float],
    config: FaultSmootherConfig,
) -> np.ndarray:
    total = np.zeros((len(eval_idx), len(grid)), dtype=np.float64)
    used = 0
    for window in config.feature_windows:
        obs_scale = observed if window == 1 else _rolling(observed, window)
        tw_scale = tw_gr if window == 1 else _rolling(tw_gr, max(1, 2 * window - 1))
        candidate = anchor[eval_idx, None] + grid[None, :]
        expected = np.interp(candidate, tw_tvt, tw_scale)
        expected = calibration["slope"] * expected + calibration["bias"]
        residual = (obs_scale[eval_idx, None] - expected) / calibration["sigma"]
        df = config.student_df
        total += -0.5 * (df + 1.0) * np.log1p((residual * residual) / df)
        used += 1
    total /= max(used * config.emission_temperature, 1e-9)
    if config.anchor_prior_strength > 0:
        total += (
            -0.5
            * config.anchor_prior_strength
            * (grid[None, :] / max(config.anchor_prior_sigma, 1e-6)) ** 2
        )
    total -= np.max(total, axis=1, keepdims=True)
    return np.exp(np.clip(total, -80.0, 0.0))


def smooth_fault_residual(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    anchor: np.ndarray,
    config: FaultSmootherConfig | None = None,
    allow_faults: bool = True,
) -> FaultSmootherResult:
    config = config or FaultSmootherConfig()
    required_hw = {"X", "Y", "Z", "GR", "TVT_input"}
    if not required_hw.issubset(hw.columns):
        raise ValueError(f"missing horizontal columns: {sorted(required_hw - set(hw.columns))}")
    if not {"TVT", "GR"}.issubset(tw.columns):
        raise ValueError("typewell must contain TVT and GR")
    anchor = np.asarray(anchor, dtype=float)
    if len(anchor) != len(hw) or not np.isfinite(anchor).all():
        raise ValueError("anchor must be finite and row-aligned")

    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    eval_idx = np.flatnonzero(~np.isfinite(tvt_input))
    if not len(eval_idx):
        zeros = np.zeros(len(hw), dtype=float)
        return FaultSmootherResult(anchor.copy(), anchor.copy(), zeros, zeros, {"status": "no_eval"})
    tw_valid = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_valid["TVT"].to_numpy(dtype=float)
    tw_gr = tw_valid["GR"].to_numpy(dtype=float)
    if len(tw_tvt) < 20:
        raise ValueError("typewell has too few finite rows")

    grid = np.arange(
        -config.max_residual,
        config.max_residual + 0.5 * config.grid_step,
        config.grid_step,
    )
    observed, calibration = _calibrate_gr(hw, tw_tvt, tw_gr, config)
    emission = _emission_matrix(
        observed, eval_idx, anchor, grid, tw_tvt, tw_gr, calibration, config
    )
    drift = _residual_drift(hw, anchor, config)
    prior = np.exp(-0.5 * (grid / 3.0) ** 2)
    prior /= prior.sum()
    shifts, weights, fault_shifts, fault_weights = _transition_tables(
        drift, config, allow_faults
    )
    posterior_weight, fault_probability = _sparse_forward_backward(
        emission,
        prior,
        shifts,
        weights,
        fault_shifts,
        fault_weights,
        config.fault_hazard if allow_faults else 0.0,
    )
    residual_mean = posterior_weight @ grid
    residual_var = posterior_weight @ (grid * grid) - residual_mean * residual_mean
    residual_std = np.sqrt(np.maximum(residual_var, 0.0))
    map_residual = grid[np.argmax(posterior_weight, axis=1)]

    posterior = anchor.copy()
    map_path = anchor.copy()
    std_full = np.zeros(len(hw), dtype=float)
    fault_full = np.zeros(len(hw), dtype=float)
    posterior[eval_idx] += residual_mean
    map_path[eval_idx] += map_residual
    std_full[eval_idx] = residual_std
    fault_full[eval_idx] = fault_probability
    posterior[np.isfinite(tvt_input)] = tvt_input[np.isfinite(tvt_input)]
    map_path[np.isfinite(tvt_input)] = tvt_input[np.isfinite(tvt_input)]

    metadata = {
        "status": "ok",
        "config": asdict(config),
        "allow_faults": bool(allow_faults),
        "eval_rows": int(len(eval_idx)),
        "grid_states": int(len(grid)),
        "calibration": calibration,
        "residual_mean_abs": float(np.mean(np.abs(residual_mean))),
        "residual_max_abs": float(np.max(np.abs(residual_mean))),
        "posterior_std_mean": float(np.mean(residual_std)),
        "posterior_std_p95": float(np.quantile(residual_std, 0.95)),
        "fault_probability_mean": float(np.mean(fault_probability)),
        "fault_probability_max": float(np.max(fault_probability)),
        "drift_mean": float(np.mean(drift)),
        "drift_max_abs": float(np.max(np.abs(drift))),
    }
    return FaultSmootherResult(posterior, map_path, std_full, fault_full, metadata)


def fit_native_fault_prior(
    wells: list[pd.DataFrame],
    threshold: float = 8.0,
    min_separation: int = 50,
) -> dict[str, Any]:
    magnitudes: list[float] = []
    total_rows = 0
    for hw in wells:
        if not {"TVT", "Z"}.issubset(hw.columns):
            continue
        u = hw["TVT"].to_numpy(dtype=float) + hw["Z"].to_numpy(dtype=float)
        valid = np.isfinite(u)
        if valid.sum() < 100:
            continue
        filled = pd.Series(u).interpolate(limit_direction="both").to_numpy(dtype=float)
        du = np.diff(filled)
        trend = pd.Series(du).rolling(101, center=True, min_periods=1).median().to_numpy()
        innovation = du - trend
        candidates = np.flatnonzero(np.abs(innovation) >= threshold)
        last = -min_separation
        for index in candidates:
            if index - last < min_separation:
                continue
            stop = min(len(innovation), index + min_separation)
            local = index + int(np.argmax(np.abs(innovation[index:stop])))
            magnitudes.append(float(innovation[local]))
            last = local
        total_rows += int(valid.sum())
    hazard = float(len(magnitudes) / max(total_rows, 1))
    clipped = np.clip(np.abs(magnitudes), 10.0, 35.0) if magnitudes else np.array([])
    return {
        "wells": int(len(wells)),
        "rows": total_rows,
        "events": int(len(magnitudes)),
        "fault_hazard": hazard,
        "magnitude_median": float(np.median(clipped)) if len(clipped) else None,
        "magnitude_p90": float(np.quantile(clipped, 0.9)) if len(clipped) else None,
    }
