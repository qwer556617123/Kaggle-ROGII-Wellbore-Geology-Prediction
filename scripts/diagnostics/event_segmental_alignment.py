"""Event-driven semi-Markov datum correction around an exact-HMM path."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class SegmentalConfig:
    residual_states: tuple[float, ...] = (
        -35.0,
        -30.0,
        -25.0,
        -20.0,
        -15.0,
        -10.0,
        0.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
    )
    min_segment_rows: int = 256
    max_changes: int = 2
    boundary_stride: int = 128
    max_event_boundaries: int = 12
    event_smooth_rows: float = 48.0
    change_penalty: float = 10.0
    magnitude_penalty: float = 10.0
    student_df: float = 4.0
    top_paths: int = 20
    temperature: float = 1.0


@dataclass
class SegmentalResult:
    pred: np.ndarray
    correction_eval: np.ndarray
    std_eval: np.ndarray
    metadata: dict[str, Any]


def _source_sigma(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    known = hw[hw["TVT_input"].notna()]
    expected = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    residual = known["GR"].fillna(0.0).to_numpy(float) - expected
    return float(np.clip(np.nanstd(residual), 10.0, 60.0))


def _emissions(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    base_eval: np.ndarray,
    states: np.ndarray,
    sigma: float,
    student_df: float,
) -> np.ndarray:
    eval_mask = hw["TVT_input"].isna().to_numpy()
    observed = (
        hw["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(float)[eval_mask]
    )
    positions = base_eval[:, None] + states[None, :]
    expected = np.interp(positions.ravel(), tw_tvt, tw_gr).reshape(positions.shape)
    z = (observed[:, None] - expected) / max(sigma, 1e-6)
    return -0.5 * (student_df + 1.0) * np.log1p(np.square(z) / student_df)


def _event_boundaries(emission: np.ndarray, zero_index: int, config: SegmentalConfig) -> list[int]:
    rows = len(emission)
    if rows < 2 * config.min_segment_rows:
        return []
    preferred = np.max(emission, axis=1) - emission[:, zero_index]
    evidence = gaussian_filter1d(preferred, config.event_smooth_rows, mode="nearest")
    change = np.abs(np.gradient(evidence))
    allowed = np.arange(config.min_segment_rows, rows - config.min_segment_rows)
    ranked = allowed[np.argsort(change[allowed])[::-1]]
    selected: list[int] = []
    minimum_gap = max(config.boundary_stride, config.min_segment_rows // 2)
    for value in ranked:
        snapped = int(round(int(value) / config.boundary_stride) * config.boundary_stride)
        if (
            config.min_segment_rows <= snapped <= rows - config.min_segment_rows
            and all(abs(snapped - current) >= minimum_gap for current in selected)
        ):
            selected.append(snapped)
        if len(selected) >= config.max_event_boundaries:
            break
    regular = list(
        range(
            config.min_segment_rows,
            rows - config.min_segment_rows + 1,
            max(config.min_segment_rows, config.boundary_stride * 2),
        )
    )
    return sorted(set(selected + regular))


def _segment_sum(cumulative: np.ndarray, state: int, start: int, stop: int) -> float:
    return float(cumulative[stop, state] - cumulative[start, state])


def _candidate_paths(
    emission: np.ndarray,
    states: np.ndarray,
    boundaries: list[int],
    config: SegmentalConfig,
) -> list[dict[str, Any]]:
    rows, state_count = emission.shape
    zero_index = int(np.argmin(np.abs(states)))
    cumulative = np.vstack([np.zeros((1, state_count)), np.cumsum(emission, axis=0)])
    paths: list[dict[str, Any]] = [
        {
            "score": _segment_sum(cumulative, zero_index, 0, rows),
            "boundaries": (),
            "states": (zero_index,),
        }
    ]
    for boundary in boundaries:
        for state in range(state_count):
            if state == zero_index:
                continue
            score = _segment_sum(cumulative, zero_index, 0, boundary)
            score += _segment_sum(cumulative, state, boundary, rows)
            score -= config.change_penalty
            score -= config.magnitude_penalty * abs(float(states[state])) / 10.0
            paths.append(
                {"score": score, "boundaries": (boundary,), "states": (zero_index, state)}
            )
    if config.max_changes >= 2:
        for first_pos, first in enumerate(boundaries):
            for second in boundaries[first_pos + 1 :]:
                if second - first < config.min_segment_rows:
                    continue
                for state1 in range(state_count):
                    if state1 == zero_index:
                        continue
                    prefix = _segment_sum(cumulative, zero_index, 0, first)
                    middle = _segment_sum(cumulative, state1, first, second)
                    for state2 in range(state_count):
                        if state2 == state1:
                            continue
                        score = prefix + middle
                        score += _segment_sum(cumulative, state2, second, rows)
                        score -= 2.0 * config.change_penalty
                        score -= config.magnitude_penalty * (
                            abs(float(states[state1])) + abs(float(states[state2]))
                        ) / 10.0
                        paths.append(
                            {
                                "score": score,
                                "boundaries": (first, second),
                                "states": (zero_index, state1, state2),
                            }
                        )
    return sorted(paths, key=lambda item: item["score"], reverse=True)[: config.top_paths]


def _path_correction(path: dict[str, Any], states: np.ndarray, rows: int) -> np.ndarray:
    correction = np.empty(rows, dtype=float)
    points = (0,) + tuple(path["boundaries"]) + (rows,)
    for index, state in enumerate(path["states"]):
        correction[points[index] : points[index + 1]] = float(states[state])
    return correction


def run_segmental(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    base_prediction: np.ndarray,
    config: SegmentalConfig | None = None,
) -> SegmentalResult:
    config = config or SegmentalConfig()
    required = {"GR", "TVT_input"}
    if not required.issubset(hw.columns):
        raise ValueError(f"Missing horizontal columns: {sorted(required - set(hw.columns))}")
    tw_valid = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    if len(tw_valid) < 20:
        raise ValueError("Typewell has too few valid rows")
    base = np.asarray(base_prediction, dtype=float)
    if len(base) != len(hw) or not np.isfinite(base).all():
        raise ValueError("Base prediction must be finite and row-aligned")
    eval_mask = hw["TVT_input"].isna().to_numpy()
    eval_rows = np.flatnonzero(eval_mask)
    if not len(eval_rows):
        return SegmentalResult(base.copy(), np.empty(0), np.empty(0), {"status": "no_eval"})
    states = np.asarray(config.residual_states, dtype=float)
    zero_index = int(np.argmin(np.abs(states)))
    tw_tvt = tw_valid["TVT"].to_numpy(float)
    tw_gr = tw_valid["GR"].to_numpy(float)
    sigma = _source_sigma(hw, tw_tvt, tw_gr)
    emission = _emissions(
        hw, tw_tvt, tw_gr, base[eval_mask], states, sigma, config.student_df
    )
    boundaries = _event_boundaries(emission, zero_index, config)
    paths = _candidate_paths(emission, states, boundaries, config)
    scores = np.asarray([path["score"] for path in paths], dtype=float)
    logits = (scores - np.max(scores)) / max(config.temperature, 1e-6)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= weights.sum()
    corrections = np.stack(
        [_path_correction(path, states, len(eval_rows)) for path in paths], axis=0
    )
    mean_correction = weights @ corrections
    second = weights @ np.square(corrections)
    std = np.sqrt(np.maximum(second - np.square(mean_correction), 0.0))
    output = base.copy()
    output[eval_mask] += mean_correction
    metadata = {
        **asdict(config),
        "status": "ok",
        "eval_rows": int(len(eval_rows)),
        "sigma": sigma,
        "candidate_boundaries": boundaries,
        "path_count": len(paths),
        "map_boundaries": list(paths[0]["boundaries"]),
        "map_residuals": [float(states[index]) for index in paths[0]["states"]],
        "map_score_gain": float(paths[0]["score"] - paths[-1]["score"]),
        "mode_weights": [float(value) for value in weights],
        "correction_rms": float(np.sqrt(np.mean(np.square(mean_correction)))),
        "correction_max_abs": float(np.max(np.abs(mean_correction))),
        "posterior_std_mean": float(np.mean(std)),
    }
    return SegmentalResult(output, mean_correction, std, metadata)
