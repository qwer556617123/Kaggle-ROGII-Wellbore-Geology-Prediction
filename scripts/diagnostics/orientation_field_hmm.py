"""Fault-aware residual HMM driven by cross-well formation orientation fields."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


CONTACT_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


@dataclass(frozen=True)
class OrientationFieldConfig:
    segment_rows: int = 128
    neighbor_count: int = 48
    prediction_stride: int = 128
    min_neighbors: int = 8
    distance_fallback_ft: float = 4500.0
    condition_fallback: float = 1.0e6
    ridge: float = 0.05
    huber_iterations: int = 5
    drift_scale: float = 1.0
    fault_threshold_floor: float = 0.15
    fault_mad_multiplier: float = 6.0
    fault_min_separation: int = 50
    fault_radius_ft: float = 2500.0
    base_fault_hazard: float = 0.0008
    max_fault_hazard: float = 0.02
    max_residual: float = 40.0
    grid_step: float = 0.5
    transition_sigma: float = 0.7
    student_df: float = 4.0
    emission_temperature: float = 1.0
    min_prefix_rows: int = 40
    min_gr_coverage: float = 0.60
    max_prefix_rmse: float = 60.0
    response_widths_ft: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0)
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


@dataclass
class OrientationPathResult:
    field_path: np.ndarray
    residual_drift: np.ndarray
    confidence: np.ndarray
    fault_hazard: np.ndarray
    metadata: dict[str, Any]


@dataclass
class OrientationHMMResult:
    posterior: np.ndarray
    posterior_std: np.ndarray
    fault_probability: np.ndarray
    field_path: np.ndarray
    metadata: dict[str, Any]


def _mad(values: np.ndarray, floor: float = 1.0e-6) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return floor
    center = float(np.median(values))
    return max(floor, 1.4826 * float(np.median(np.abs(values - center))))


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _robust_affine(expected: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(expected) & np.isfinite(observed)
    x = np.asarray(expected, dtype=float)[valid]
    y = np.asarray(observed, dtype=float)[valid]
    if len(x) < 3 or float(np.std(x)) < 1.0e-8:
        bias = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, bias, 60.0
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(6):
        residual = y - design @ coef
        scale = _mad(residual, 1.0)
        weight = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1.0e-9))
        root = np.sqrt(weight)
        coef, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    slope = float(np.clip(coef[0], 0.2, 5.0))
    bias = float(np.median(y - slope * x))
    rmse = float(np.sqrt(np.mean(np.square(y - (slope * x + bias)))))
    return slope, bias, rmse


class OrientationFieldModel:
    """Local robust orientation field built only from train contact derivatives."""

    def __init__(
        self,
        observations: np.ndarray,
        wells: np.ndarray,
        fault_observations: np.ndarray,
        fault_wells: np.ndarray,
        config: OrientationFieldConfig,
    ) -> None:
        if observations.ndim != 2 or observations.shape[1] != 6:
            raise ValueError("observations must contain X,Y,vx,vy,slope,disagreement")
        self.observations = np.asarray(observations, dtype=float)
        self.wells = np.asarray(wells, dtype=object)
        self.fault_observations = np.asarray(fault_observations, dtype=float).reshape(-1, 3)
        self.fault_wells = np.asarray(fault_wells, dtype=object)
        self.config = config
        self.tree = cKDTree(self.observations[:, :2])
        self.fault_tree = (
            cKDTree(self.fault_observations[:, :2]) if len(self.fault_observations) else None
        )

    def _unique_neighbors(self, xy: np.ndarray, excluded_well: str | None) -> tuple[np.ndarray, np.ndarray]:
        query_count = min(
            len(self.observations),
            max(512, self.config.neighbor_count * 16),
        )
        distance, indices = self.tree.query(np.asarray(xy, dtype=float), k=query_count)
        chosen: list[int] = []
        chosen_distance: list[float] = []
        seen: set[str] = set()
        for value, index in zip(np.atleast_1d(distance), np.atleast_1d(indices)):
            well = str(self.wells[int(index)])
            if well == excluded_well or well in seen:
                continue
            velocity = self.observations[int(index), 2:4]
            if float(velocity @ velocity) < 0.1:
                continue
            seen.add(well)
            chosen.append(int(index))
            chosen_distance.append(float(value))
            if len(chosen) >= self.config.neighbor_count:
                break
        return np.asarray(chosen, dtype=int), np.asarray(chosen_distance, dtype=float)

    def predict_gradient(
        self,
        xy: np.ndarray,
        target_velocity: np.ndarray,
        excluded_well: str | None = None,
    ) -> dict[str, float]:
        indices, distance = self._unique_neighbors(xy, excluded_well)
        if len(indices) < self.config.min_neighbors:
            return {
                "slope": 0.0,
                "gx": 0.0,
                "gy": 0.0,
                "confidence": 0.0,
                "neighbors": float(len(indices)),
                "distance_median": float("inf"),
                "condition": float("inf"),
                "residual_mad": float("inf"),
                "contact_disagreement": float("inf"),
                "fallback": 1.0,
            }

        design = self.observations[indices, 2:4]
        target = self.observations[indices, 4]
        distance_scale = max(float(np.median(distance)), 200.0)
        spatial_weight = np.exp(-distance / distance_scale)
        weight = spatial_weight.copy()
        coef = np.zeros(2, dtype=float)
        lhs = np.eye(2)
        for _ in range(self.config.huber_iterations):
            lhs = design.T @ (weight[:, None] * design) + np.eye(2) * self.config.ridge
            rhs = design.T @ (weight * target)
            coef = np.linalg.solve(lhs, rhs)
            residual = target - design @ coef
            scale = _mad(residual, 1.0e-4)
            robust = np.minimum(1.0, 2.0 * scale / np.maximum(np.abs(residual), 1.0e-9))
            weight = spatial_weight * robust

        residual = target - design @ coef
        condition = float(np.linalg.cond(lhs))
        distance_median = float(np.median(distance))
        residual_mad = _mad(residual, 1.0e-4)
        disagreement = float(np.median(self.observations[indices, 5]))
        distance_confidence = float(
            np.clip(1.0 - (distance_median / self.config.distance_fallback_ft) ** 2, 0.0, 1.0)
        )
        condition_confidence = float(1.0 / (1.0 + condition / 1000.0))
        residual_confidence = float(1.0 / (1.0 + residual_mad / 0.02))
        disagreement_confidence = float(1.0 / (1.0 + disagreement / 0.01))
        confidence = (
            distance_confidence
            * condition_confidence
            * np.sqrt(residual_confidence * disagreement_confidence)
        )
        fallback = bool(
            distance_median > self.config.distance_fallback_ft
            or condition > self.config.condition_fallback
        )
        if fallback:
            confidence = 0.0
        slope = float(np.asarray(target_velocity, dtype=float) @ coef)
        return {
            "slope": slope,
            "gx": float(coef[0]),
            "gy": float(coef[1]),
            "confidence": float(confidence),
            "neighbors": float(len(indices)),
            "distance_median": distance_median,
            "condition": condition,
            "residual_mad": residual_mad,
            "contact_disagreement": disagreement,
            "fallback": float(fallback),
        }

    def predict_fault(self, xy: np.ndarray, excluded_well: str | None = None) -> dict[str, float]:
        base = self.config.base_fault_hazard
        if self.fault_tree is None:
            return {"hazard": base, "distance": float("inf"), "jump": 0.0, "support": 0.0}
        query_count = min(len(self.fault_observations), 128)
        distance, indices = self.fault_tree.query(np.asarray(xy, dtype=float), k=query_count)
        selected: list[tuple[float, int]] = []
        seen: set[str] = set()
        for value, index in zip(np.atleast_1d(distance), np.atleast_1d(indices)):
            well = str(self.fault_wells[int(index)])
            if well == excluded_well or well in seen:
                continue
            if float(value) > self.config.fault_radius_ft:
                continue
            seen.add(well)
            selected.append((float(value), int(index)))
            if len(selected) >= 16:
                break
        if not selected:
            return {"hazard": base, "distance": float("inf"), "jump": 0.0, "support": 0.0}
        distance = np.asarray([item[0] for item in selected], dtype=float)
        indices = np.asarray([item[1] for item in selected], dtype=int)
        weight = np.exp(-distance / max(float(np.median(distance)), 250.0))
        weight /= weight.sum()
        proximity = float(np.sum(weight * np.exp(-distance / 1000.0)))
        hazard = float(
            np.clip(
                base + proximity * 0.01,
                base,
                self.config.max_fault_hazard,
            )
        )
        jump = float(np.sum(weight * self.fault_observations[indices, 2]))
        return {
            "hazard": hazard,
            "distance": float(np.median(distance)),
            "jump": jump,
            "support": float(len(indices)),
        }

    def predict_path(
        self,
        hw: pd.DataFrame,
        anchor: np.ndarray,
        target_well: str | None = None,
        use_faults: bool = False,
    ) -> OrientationPathResult:
        required = {"MD", "X", "Y", "Z", "TVT_input"}
        if not required.issubset(hw.columns):
            raise ValueError(f"missing horizontal columns: {sorted(required - set(hw.columns))}")
        anchor = np.asarray(anchor, dtype=float)
        if len(anchor) != len(hw) or not np.isfinite(anchor).all():
            raise ValueError("anchor must be finite and row-aligned")
        md = hw["MD"].to_numpy(dtype=float)
        x = hw["X"].to_numpy(dtype=float)
        y = hw["Y"].to_numpy(dtype=float)
        z = hw["Z"].to_numpy(dtype=float)
        tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(dtype=float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        eval_idx = np.flatnonzero(~np.isfinite(tvt_input))
        if not len(known) or not len(eval_idx):
            zeros = np.zeros(len(eval_idx), dtype=float)
            return OrientationPathResult(anchor.copy(), zeros, zeros, zeros, {"status": "no_eval"})
        if int(eval_idx[0]) != int(known[-1]) + 1:
            raise ValueError("orientation field requires a native suffix mask")

        centers = np.unique(np.r_[eval_idx[:: self.config.prediction_stride], eval_idx[-1]])
        rows: list[dict[str, float]] = []
        fault_rows: list[dict[str, float]] = []
        for center in centers:
            lo = max(0, int(center) - 32)
            hi = min(len(hw) - 1, int(center) + 32)
            span = max(float(md[hi] - md[lo]), 1.0e-6)
            velocity = np.asarray([(x[hi] - x[lo]) / span, (y[hi] - y[lo]) / span])
            rows.append(self.predict_gradient(np.asarray([x[center], y[center]]), velocity, target_well))
            fault_rows.append(
                self.predict_fault(np.asarray([x[center], y[center]]), target_well)
                if use_faults
                else {
                    "hazard": self.config.base_fault_hazard,
                    "distance": float("inf"),
                    "jump": 0.0,
                    "support": 0.0,
                }
            )

        center_slope = np.asarray([row["slope"] for row in rows], dtype=float)
        center_confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
        center_hazard = np.asarray([row["hazard"] for row in fault_rows], dtype=float)
        slope = np.interp(eval_idx, centers, center_slope)
        confidence = np.interp(eval_idx, centers, center_confidence)
        fault_hazard = np.interp(eval_idx, centers, center_hazard)
        dm = np.diff(np.r_[md[known[-1]], md[eval_idx]])
        field_increment = slope * dm
        heel_u = float(tvt_input[known[-1]] + z[known[-1]])
        field_u = heel_u + np.cumsum(field_increment)
        field_path = anchor.copy()
        field_path[eval_idx] = field_u - z[eval_idx]
        field_path[known] = tvt_input[known]

        anchor_u = anchor[eval_idx] + z[eval_idx]
        anchor_increment = np.diff(np.r_[heel_u, anchor_u])
        residual_drift = (
            self.config.drift_scale
            * confidence
            * (field_increment - anchor_increment)
        )
        metadata = {
            "status": "ok",
            "target_well": target_well,
            "eval_rows": int(len(eval_idx)),
            "centers": int(len(centers)),
            "confidence_mean": float(np.mean(confidence)),
            "confidence_p10": float(np.quantile(confidence, 0.10)),
            "fallback_centers": int(sum(bool(row["fallback"]) for row in rows)),
            "distance_median": float(np.median([row["distance_median"] for row in rows])),
            "condition_median": float(np.median([row["condition"] for row in rows])),
            "gradient_residual_mad": float(np.median([row["residual_mad"] for row in rows])),
            "contact_disagreement": float(
                np.median([row["contact_disagreement"] for row in rows])
            ),
            "gx_median": float(np.median([row["gx"] for row in rows])),
            "gy_median": float(np.median([row["gy"] for row in rows])),
            "field_move_mean": float(np.mean(field_path[eval_idx] - anchor[eval_idx])),
            "field_move_std": float(np.std(field_path[eval_idx] - anchor[eval_idx])),
            "residual_drift_mean_abs": float(np.mean(np.abs(residual_drift))),
            "fault_hazard_mean": float(np.mean(fault_hazard)),
            "fault_hazard_max": float(np.max(fault_hazard)),
            "fault_support_centers": int(sum(row["support"] > 0 for row in fault_rows)),
        }
        return OrientationPathResult(field_path, residual_drift, confidence, fault_hazard, metadata)


def _contact_observations(
    frame: pd.DataFrame,
    well: str,
    config: OrientationFieldConfig,
) -> tuple[list[tuple[float, ...]], list[str], list[tuple[float, ...]], list[str]]:
    md = frame["MD"].to_numpy(dtype=float)
    x = frame["X"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    contacts = frame[list(CONTACT_COLUMNS)].to_numpy(dtype=float)
    dm = np.diff(md)
    valid_dm = dm > 0
    derivative = np.divide(
        np.diff(contacts, axis=0),
        dm[:, None],
        out=np.full((len(dm), len(CONTACT_COLUMNS)), np.nan),
        where=valid_dm[:, None],
    )
    common = np.nanmedian(derivative, axis=1)
    disagreement = np.nanstd(derivative, axis=1)
    vx = np.divide(np.diff(x), dm, out=np.full(len(dm), np.nan), where=valid_dm)
    vy = np.divide(np.diff(y), dm, out=np.full(len(dm), np.nan), where=valid_dm)
    half = max(16, config.segment_rows // 2)
    observations: list[tuple[float, ...]] = []
    wells: list[str] = []
    for center in range(half, len(frame) - half, config.segment_rows):
        section = slice(center - half, center + half)
        finite = np.isfinite(common[section]) & np.isfinite(vx[section]) & np.isfinite(vy[section])
        if finite.sum() < max(16, half // 2):
            continue
        observations.append(
            (
                float(x[center]),
                float(y[center]),
                float(np.nanmedian(vx[section])),
                float(np.nanmedian(vy[section])),
                float(np.nanmedian(common[section])),
                float(np.nanmedian(disagreement[section])),
            )
        )
        wells.append(well)

    trend = _rolling_median(common, 101)
    innovation = common - trend
    threshold = max(
        config.fault_threshold_floor,
        config.fault_mad_multiplier * _mad(innovation, 1.0e-4),
    )
    candidates = np.flatnonzero(np.isfinite(innovation) & (np.abs(innovation) >= threshold))
    fault_observations: list[tuple[float, ...]] = []
    fault_wells: list[str] = []
    cursor = 0
    while cursor < len(candidates):
        start = int(candidates[cursor])
        stop = start + config.fault_min_separation
        group = []
        while cursor < len(candidates) and int(candidates[cursor]) < stop:
            group.append(int(candidates[cursor]))
            cursor += 1
        index = max(group, key=lambda value: abs(float(innovation[value])))
        fault_observations.append(
            (float(x[index + 1]), float(y[index + 1]), float(innovation[index] * dm[index]))
        )
        fault_wells.append(well)
    return observations, wells, fault_observations, fault_wells


def build_orientation_field(
    train_dir: Path,
    config: OrientationFieldConfig | None = None,
    well_paths: Iterable[Path] | None = None,
) -> OrientationFieldModel:
    config = config or OrientationFieldConfig()
    paths = list(well_paths) if well_paths is not None else sorted(Path(train_dir).glob("*__horizontal_well.csv"))
    observations: list[tuple[float, ...]] = []
    wells: list[str] = []
    fault_observations: list[tuple[float, ...]] = []
    fault_wells: list[str] = []
    usecols = ["MD", "X", "Y", *CONTACT_COLUMNS]
    for path in paths:
        well = Path(path).name.split("__", 1)[0]
        frame = pd.read_csv(path, usecols=usecols)
        result = _contact_observations(frame, well, config)
        observations.extend(result[0])
        wells.extend(result[1])
        fault_observations.extend(result[2])
        fault_wells.extend(result[3])
    if not observations:
        raise RuntimeError("no finite orientation observations were built")
    return OrientationFieldModel(
        np.asarray(observations, dtype=float),
        np.asarray(wells, dtype=object),
        np.asarray(fault_observations, dtype=float),
        np.asarray(fault_wells, dtype=object),
        config,
    )


def calibrate_tool_response(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    config: OrientationFieldConfig,
    enabled: bool,
) -> dict[str, Any]:
    valid_tw = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    tvt = valid_tw["TVT"].to_numpy(dtype=float)
    raw = valid_tw["GR"].to_numpy(dtype=float)
    if len(tvt) < 20:
        raise ValueError("typewell has too few finite rows")
    widths = config.response_widths_ft if enabled else (0.0,)
    step = max(float(np.median(np.diff(tvt))), 1.0e-3)
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(hw["GR"], errors="coerce").to_numpy(dtype=float)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(observed))
    coverage = float(np.mean(np.isfinite(observed)))
    candidates = []
    for width in widths:
        smoothed = raw if width <= 0 else gaussian_filter1d(raw, sigma=width / step, mode="nearest")
        expected = np.interp(tvt_input[known], tvt, smoothed) if len(known) else np.array([])
        slope, bias, rmse = _robust_affine(expected, observed[known])
        candidates.append(
            {
                "width_ft": float(width),
                "gr": smoothed,
                "slope": slope,
                "bias": bias,
                "prefix_rmse": rmse,
            }
        )
    candidates.sort(key=lambda item: item["prefix_rmse"])
    selected = candidates[: min(2, len(candidates))]
    losses = np.asarray([item["prefix_rmse"] for item in selected], dtype=float)
    scale = max(float(np.std(losses)), 1.0)
    weights = np.exp(-(losses - losses.min()) / scale)
    weights /= weights.sum()
    neutral = bool(
        len(known) < config.min_prefix_rows
        or coverage < config.min_gr_coverage
        or float(losses[0]) > config.max_prefix_rmse
    )
    return {
        "tvt": tvt,
        "selected": selected,
        "weights": weights,
        "known_rows": int(len(known)),
        "coverage": coverage,
        "neutral": neutral,
    }


def _emission_matrix(
    hw: pd.DataFrame,
    anchor: np.ndarray,
    eval_idx: np.ndarray,
    grid: np.ndarray,
    calibration: dict[str, Any],
    config: OrientationFieldConfig,
) -> np.ndarray:
    emission = np.ones((len(eval_idx), len(grid)), dtype=np.float64)
    if calibration["neutral"]:
        return emission
    observed = pd.to_numeric(hw["GR"], errors="coerce").to_numpy(dtype=float)[eval_idx]
    finite = np.isfinite(observed)
    if not finite.any():
        return emission
    candidate_tvt = anchor[eval_idx, None] + grid[None, :]
    likelihood = np.zeros_like(emission)
    for weight, item in zip(calibration["weights"], calibration["selected"]):
        expected = np.interp(candidate_tvt, calibration["tvt"], item["gr"])
        expected = item["slope"] * expected + item["bias"]
        sigma = float(np.clip(item["prefix_rmse"], 6.0, 60.0))
        residual = (observed[:, None] - expected) / sigma
        df = config.student_df
        log_likelihood = -0.5 * (df + 1.0) * np.log1p((residual * residual) / df)
        likelihood += float(weight) * np.exp(log_likelihood / config.emission_temperature)
    likelihood[~finite] = 1.0
    row_max = np.maximum(np.max(likelihood, axis=1, keepdims=True), 1.0e-300)
    return np.clip(likelihood / row_max, 1.0e-80, 1.0)


def _transition_tables(
    drift: np.ndarray,
    hazard: np.ndarray,
    config: OrientationFieldConfig,
    use_faults: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_rows: list[list[tuple[int, float]]] = []
    fault_mass_rows: list[list[tuple[int, float]]] = []
    smooth_radius = max(1, int(np.ceil(3.0 * config.transition_sigma / config.grid_step)))
    smooth_offsets = np.arange(-smooth_radius, smooth_radius + 1)
    for value, local_hazard in zip(np.asarray(drift), np.asarray(hazard)):
        local_hazard = float(local_hazard if use_faults else 0.0)
        center = float(value / config.grid_step)
        base = int(np.floor(center))
        smooth_weight = np.exp(
            -0.5
            * ((smooth_offsets + base - center) * config.grid_step / config.transition_sigma) ** 2
        )
        smooth_weight /= smooth_weight.sum()
        components: dict[int, float] = {}
        fault_components: dict[int, float] = {}
        for shift, weight in zip(smooth_offsets + base, smooth_weight):
            components[int(shift)] = components.get(int(shift), 0.0) + (
                (1.0 - local_hazard) * float(weight)
            )
        if local_hazard > 0:
            per_jump = 1.0 / len(config.fault_jumps)
            for jump in config.fault_jumps:
                fault_center = (value + jump) / config.grid_step
                lower = int(np.floor(fault_center))
                fraction = float(fault_center - lower)
                for shift, weight in (
                    (lower, 1.0 - fraction),
                    (lower + 1, fraction),
                ):
                    amount = per_jump * weight
                    fault_components[shift] = fault_components.get(shift, 0.0) + amount
                    components[shift] = components.get(shift, 0.0) + local_hazard * amount
        total = max(sum(components.values()), 1.0e-15)
        all_rows.append(sorted((shift, weight / total) for shift, weight in components.items()))
        fault_total = sum(fault_components.values())
        fault_mass_rows.append(
            sorted((shift, weight / fault_total) for shift, weight in fault_components.items())
            if fault_total > 0
            else []
        )
    width = max(len(row) for row in all_rows)
    fault_width = max(max((len(row) for row in fault_mass_rows), default=0), 1)
    shifts = np.zeros((len(all_rows), width), dtype=np.int16)
    weights = np.zeros((len(all_rows), width), dtype=np.float64)
    fault_shifts = np.zeros((len(all_rows), fault_width), dtype=np.int16)
    fault_weights = np.zeros((len(all_rows), fault_width), dtype=np.float64)
    for row_index, row in enumerate(all_rows):
        for column, (shift, weight) in enumerate(row):
            shifts[row_index, column] = shift
            weights[row_index, column] = weight
    for row_index, row in enumerate(fault_mass_rows):
        for column, (shift, weight) in enumerate(row):
            fault_shifts[row_index, column] = shift
            fault_weights[row_index, column] = weight
    return shifts, weights, fault_shifts, fault_weights


@njit(cache=True, nogil=True)
def _forward_backward(
    emission: np.ndarray,
    prior: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    fault_shifts: np.ndarray,
    fault_weights: np.ndarray,
    hazard: np.ndarray,
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
            if hazard[t] > 0.0:
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
            fault_evidence += fault_predicted[state] * emission[t, state]
        if norm <= 1.0e-300:
            norm = 1.0e-300
        for state in range(n_states):
            forward[t, state] = predicted[state] / norm
        fault_probability[t] = min(1.0, hazard[t] * fault_evidence / norm)
        previous = forward[t].copy()

    backward = np.ones((n_rows, n_states), dtype=np.float64)
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
        if norm <= 1.0e-300:
            norm = 1.0e-300
        for state in range(n_states):
            backward[t, state] /= norm

    posterior = np.zeros((n_rows, n_states), dtype=np.float64)
    for t in range(n_rows):
        norm = 0.0
        for state in range(n_states):
            posterior[t, state] = forward[t, state] * backward[t, state]
            norm += posterior[t, state]
        if norm <= 1.0e-300:
            norm = 1.0e-300
        for state in range(n_states):
            posterior[t, state] /= norm
    return posterior, fault_probability


def run_orientation_hmm(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    anchor: np.ndarray,
    model: OrientationFieldModel,
    target_well: str | None = None,
    use_faults: bool = False,
    use_tool_response: bool = False,
) -> OrientationHMMResult:
    config = model.config
    anchor = np.asarray(anchor, dtype=float)
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(dtype=float)
    eval_idx = np.flatnonzero(~np.isfinite(tvt_input))
    path = model.predict_path(hw, anchor, target_well, use_faults=use_faults)
    if not len(eval_idx):
        zeros = np.zeros(len(hw), dtype=float)
        return OrientationHMMResult(anchor.copy(), zeros, zeros, path.field_path, {"status": "no_eval"})
    grid = np.arange(
        -config.max_residual,
        config.max_residual + 0.5 * config.grid_step,
        config.grid_step,
    )
    calibration = calibrate_tool_response(hw, tw, config, enabled=use_tool_response)
    emission = _emission_matrix(hw, anchor, eval_idx, grid, calibration, config)
    prior = np.exp(-0.5 * (grid / 3.0) ** 2)
    prior /= prior.sum()
    hazard = path.fault_hazard if use_faults else np.zeros(len(eval_idx), dtype=float)
    tables = _transition_tables(path.residual_drift, hazard, config, use_faults)
    posterior_weight, fault_probability = _forward_backward(
        emission,
        prior,
        tables[0],
        tables[1],
        tables[2],
        tables[3],
        hazard,
    )
    residual_mean = posterior_weight @ grid
    residual_var = posterior_weight @ (grid * grid) - residual_mean * residual_mean
    residual_std = np.sqrt(np.maximum(residual_var, 0.0))
    posterior = anchor.copy()
    posterior[eval_idx] += residual_mean
    posterior[np.isfinite(tvt_input)] = tvt_input[np.isfinite(tvt_input)]
    std_full = np.zeros(len(hw), dtype=float)
    fault_full = np.zeros(len(hw), dtype=float)
    std_full[eval_idx] = residual_std
    fault_full[eval_idx] = fault_probability
    metadata = {
        "status": "ok",
        "config": asdict(config),
        "use_faults": bool(use_faults),
        "use_tool_response": bool(use_tool_response),
        "eval_rows": int(len(eval_idx)),
        "grid_states": int(len(grid)),
        "response_widths": [float(item["width_ft"]) for item in calibration["selected"]],
        "response_weights": [float(value) for value in calibration["weights"]],
        "prefix_rmse": [float(item["prefix_rmse"]) for item in calibration["selected"]],
        "gr_coverage": float(calibration["coverage"]),
        "neutral_emission": bool(calibration["neutral"]),
        "residual_mean_abs": float(np.mean(np.abs(residual_mean))),
        "residual_max_abs": float(np.max(np.abs(residual_mean))),
        "posterior_std_mean": float(np.mean(residual_std)),
        "posterior_std_p95": float(np.quantile(residual_std, 0.95)),
        "fault_probability_mean": float(np.mean(fault_probability)),
        "fault_probability_max": float(np.max(fault_probability)),
        "orientation": path.metadata,
    }
    return OrientationHMMResult(posterior, std_full, fault_full, path.field_path, metadata)
