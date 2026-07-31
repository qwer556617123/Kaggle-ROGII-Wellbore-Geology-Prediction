"""Deterministic multimodal geosteering from a known heel and a typewell log."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PosteriorConfig:
    max_shift: float = 60.0
    grid_step: float = 0.5
    mode_bucket: float = 2.0
    temperature: float = 0.08
    top_modes: int = 10
    huber_delta: float = 1.5
    min_calibration_rows: int = 40
    min_gr_rows: int = 30
    ambiguous_gap_min: float = 10.0
    ambiguous_gap_max: float = 35.0
    ambiguous_score_margin: float = 0.12
    slope_limit: float = 0.06
    min_score_gain: float = 0.03


@dataclass
class PosteriorResult:
    prior: np.ndarray
    map_path: np.ndarray
    posterior: np.ndarray
    hybrid: np.ndarray
    metadata: dict[str, Any]


def _rolling(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    roller = series.rolling(window, center=True, min_periods=1)
    if kind == "median":
        return roller.median().to_numpy(dtype=float)
    return roller.mean().to_numpy(dtype=float)


def _mad(values: np.ndarray, floor: float = 1e-6) -> float:
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
    if len(x) < 3 or np.std(x) < 1e-6:
        offset = float(np.nanmedian(y) - np.nanmedian(x)) if len(x) else 0.0
        return 1.0, offset, float("inf"), int(len(x))
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(6):
        residual = y - design @ coef
        scale = _mad(residual)
        ratio = np.abs(residual) / max(2.5 * scale, 1e-6)
        weights = 1.0 / np.maximum(1.0, ratio)
        weighted = design * np.sqrt(weights)[:, None]
        coef, *_ = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)
    slope = float(np.clip(coef[0], 0.2, 5.0))
    intercept = float(np.median(y - slope * x))
    residual = y - (slope * x + intercept)
    return slope, intercept, float(np.sqrt(np.mean(residual * residual))), int(len(x))


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) < 1e-6:
        return 0.0, float(np.nanmedian(y)) if len(y) else 0.0
    center = float(x[-1])
    design = np.column_stack([x - center, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(6):
        residual = y - design @ coef
        scale = _mad(residual)
        weights = 1.0 / (1.0 + (residual / max(2.5 * scale, 1e-6)) ** 2)
        weighted = design * np.sqrt(weights)[:, None]
        coef, *_ = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)
    return float(coef[0]), float(coef[1])


def _heel_prior(hw: pd.DataFrame, config: PosteriorConfig) -> tuple[np.ndarray, dict[str, float]]:
    md = hw["MD"].to_numpy(dtype=float)
    z = hw["Z"].to_numpy(dtype=float)
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    known_idx = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    if not len(known_idx):
        raise ValueError("No known TVT_input rows")
    last = int(known_idx[-1])
    slopes = []
    for window in (80, 240, len(known_idx)):
        idx = known_idx[-min(window, len(known_idx)) :]
        slope, _ = _robust_line(md[idx], tvt_input[idx] + z[idx])
        slopes.append(slope)
    slope = float(np.clip(np.median(slopes), -config.slope_limit, config.slope_limit))
    last_u = float(tvt_input[last] + z[last])
    prior_u = last_u + slope * (md - md[last])
    prior_tvt = prior_u - z

    replay_rmse = float("inf")
    if len(known_idx) >= 80:
        cut = max(30, int(round(len(known_idx) * 0.75)))
        fit_idx = known_idx[:cut]
        hold_idx = known_idx[cut:]
        replay_slope, replay_u = _robust_line(md[fit_idx], tvt_input[fit_idx] + z[fit_idx])
        replay_slope = float(np.clip(replay_slope, -config.slope_limit, config.slope_limit))
        replay = replay_u + replay_slope * (md[hold_idx] - md[fit_idx[-1]]) - z[hold_idx]
        replay_rmse = float(np.sqrt(np.mean((replay - tvt_input[hold_idx]) ** 2)))
    return prior_tvt, {
        "heel_u": last_u,
        "heel_md": float(md[last]),
        "heel_slope": slope,
        "prefix_replay_rmse": replay_rmse,
    }


def _candidate_corrections(n_eval: int, config: PosteriorConfig) -> list[tuple[str, np.ndarray, float]]:
    if n_eval <= 0:
        return [("base", np.empty(0, dtype=float), 0.0)]
    t = np.linspace(0.0, 1.0, n_eval)
    ramp = 1.0 - np.exp(-t / 0.08)
    ramp /= max(float(ramp[-1]), 1e-9)
    offsets = (-30.0, -20.0, -15.0, -10.0, 10.0, 15.0, 20.0, 30.0)
    candidates: list[tuple[str, np.ndarray, float]] = [("base", np.zeros(n_eval), 0.0)]
    for value in offsets:
        candidates.append((f"datum_{value:+g}", np.clip(value * ramp, -config.max_shift, config.max_shift), 0.08))
        candidates.append((f"drift_{value:+g}", np.clip(value * t, -config.max_shift, config.max_shift), 0.04))
    for hinge in (0.33, 0.66):
        shape = np.clip((t - hinge) / max(1.0 - hinge, 1e-6), 0.0, 1.0)
        for value in (-30.0, -15.0, 15.0, 30.0):
            candidates.append((f"hinge{hinge:.2f}_{value:+g}", value * shape, 0.08))
    width = max(2.0 / max(n_eval, 1), 0.004)
    for location in (0.25, 0.50, 0.75):
        shape = 1.0 / (1.0 + np.exp(-np.clip((t - location) / width, -50.0, 50.0)))
        shape -= shape[0]
        shape /= max(float(shape[-1]), 1e-9)
        for value in (-30.0, -20.0, -15.0, 15.0, 20.0, 30.0):
            candidates.append((f"fault{location:.2f}_{value:+g}", value * shape, 0.16))
    return candidates


def _huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual * residual, delta * (absolute - 0.5 * delta))


def _calibration_scales(
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    config: PosteriorConfig,
) -> tuple[list[dict[str, Any]], float, int]:
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    horizontal_gr = hw["GR"].to_numpy(dtype=float)
    known = np.isfinite(tvt_input) & np.isfinite(horizontal_gr)
    scales = []
    for name, window, kind in (("raw", 1, "mean"), ("median7", 7, "median"), ("mean21", 21, "mean")):
        hgr = horizontal_gr.copy() if window == 1 else _rolling(horizontal_gr, window, kind)
        tgr = tw_gr.copy() if window == 1 else _rolling(tw_gr, window, kind)
        expected = np.interp(tvt_input[known], tw_tvt, tgr)
        slope, intercept, rmse, rows = _huber_affine(hgr[known], expected)
        calibrated = slope * hgr + intercept
        residual = expected - calibrated[known]
        sigma = float(np.clip(_mad(residual), 6.0, 60.0))
        scales.append(
            {
                "name": name,
                "horizontal": calibrated,
                "typewell": tgr,
                "sigma": sigma,
                "slope": slope,
                "intercept": intercept,
                "rmse": rmse,
                "rows": rows,
            }
        )
    finite_rmse = [s["rmse"] for s in scales if np.isfinite(s["rmse"])]
    calibration_rmse = float(np.median(finite_rmse)) if finite_rmse else float("inf")
    return scales, calibration_rmse, int(known.sum())


def _score_candidate(
    predicted_tvt: np.ndarray,
    correction: np.ndarray,
    path_penalty: float,
    eval_idx: np.ndarray,
    tw_tvt: np.ndarray,
    scales: list[dict[str, Any]],
    config: PosteriorConfig,
) -> tuple[float, int]:
    losses = []
    rows = 0
    for scale in scales:
        observed = scale["horizontal"][eval_idx]
        valid = np.isfinite(observed) & np.isfinite(predicted_tvt)
        if int(valid.sum()) < config.min_gr_rows:
            continue
        expected = np.interp(predicted_tvt[valid], tw_tvt, scale["typewell"])
        residual = (observed[valid] - expected) / scale["sigma"]
        losses.append(float(np.mean(_huber_loss(residual, config.huber_delta))))
        rows = max(rows, int(valid.sum()))
    emission = float(np.mean(losses)) if losses else 1.0
    magnitude = 0.025 * float(np.mean(np.abs(correction))) / max(config.max_shift, 1e-6)
    roughness = 0.0
    if len(correction) >= 3:
        roughness = 0.01 * float(np.sqrt(np.mean(np.diff(correction, n=2) ** 2)))
    return emission + magnitude + roughness + path_penalty, rows


def _heuristic_alpha(metadata: dict[str, Any]) -> float:
    coverage = float(metadata.get("gr_coverage", 0.0))
    calibration = float(metadata.get("calibration_rmse", float("inf")))
    replay = float(metadata.get("prefix_replay_rmse", float("inf")))
    confidence = float(metadata.get("confidence", 0.0))
    if coverage < 0.20 or calibration > 35.0 or replay > 10.0:
        return 0.0
    if confidence >= 0.45:
        return 0.75
    if confidence >= 0.25:
        return 0.50
    if confidence >= 0.12:
        return 0.25
    return 0.0


def predict_multimodal(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    anchor: np.ndarray | None = None,
    config: PosteriorConfig | None = None,
    alpha: float | None = None,
    search_from_anchor: bool = False,
) -> PosteriorResult:
    config = config or PosteriorConfig()
    required_hw = {"MD", "Z", "GR", "TVT_input"}
    if not required_hw.issubset(hw.columns):
        raise ValueError(f"Missing horizontal columns: {sorted(required_hw - set(hw.columns))}")
    if not {"TVT", "GR"}.issubset(tw.columns):
        raise ValueError("Typewell must contain TVT and GR")

    tw_valid = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_valid = tw_valid.drop_duplicates("TVT")
    if len(tw_valid) < 20:
        raise ValueError("Typewell has too few valid rows")
    tw_tvt = tw_valid["TVT"].to_numpy(dtype=float)
    tw_gr = tw_valid["GR"].to_numpy(dtype=float)
    prior, heel_meta = _heel_prior(hw, config)
    eval_mask = hw["TVT_input"].isna().to_numpy()
    eval_idx = np.flatnonzero(eval_mask)
    if not len(eval_idx):
        known = hw["TVT_input"].to_numpy(dtype=float)
        metadata = {**asdict(config), **heel_meta, "status": "no_eval", "alpha": 0.0}
        return PosteriorResult(known.copy(), known.copy(), known.copy(), known.copy(), metadata)

    scales, calibration_rmse, calibration_rows = _calibration_scales(hw, tw_tvt, tw_gr, config)
    candidates = _candidate_corrections(len(eval_idx), config)
    scored = []
    if search_from_anchor:
        if anchor is None:
            raise ValueError("search_from_anchor requires an anchor")
        search_base = np.asarray(anchor, dtype=float)
        if len(search_base) != len(hw):
            raise ValueError("Anchor length differs from horizontal well")
        prior_eval = search_base[eval_idx]
        if not np.isfinite(prior_eval).all():
            raise ValueError("Anchor contains non-finite eval predictions")
        search_base_name = "anchor"
    else:
        prior_eval = prior[eval_idx]
        search_base_name = "heel_prior"
    for name, correction, penalty in candidates:
        prediction = prior_eval + correction
        score, gr_rows = _score_candidate(
            prediction, correction, penalty, eval_idx, tw_tvt, scales, config
        )
        datum = float(np.mean(correction[len(correction) // 2 :]))
        scored.append(
            {
                "name": name,
                "prediction": prediction,
                "correction": correction,
                "score": score,
                "datum": datum,
                "bucket": int(round(datum / config.mode_bucket)),
                "gr_rows": gr_rows,
            }
        )

    best_by_bucket: dict[int, dict[str, Any]] = {}
    for item in scored:
        current = best_by_bucket.get(item["bucket"])
        if current is None or item["score"] < current["score"]:
            best_by_bucket[item["bucket"]] = item
    modes = sorted(best_by_bucket.values(), key=lambda item: item["score"])[: config.top_modes]
    best = modes[0]
    scores = np.asarray([m["score"] for m in modes], dtype=float)
    logits = -(scores - scores.min()) / max(config.temperature, 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    posterior_eval = np.sum(
        np.stack([m["prediction"] for m in modes], axis=0) * weights[:, None], axis=0
    )
    map_eval = np.asarray(best["prediction"], dtype=float)

    entropy = 0.0
    if len(weights) > 1:
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))) / np.log(len(weights)))
    second_gap = 0.0
    second_margin = float("inf")
    if len(modes) > 1:
        second_gap = abs(float(modes[1]["datum"] - modes[0]["datum"]))
        second_margin = float(modes[1]["score"] - modes[0]["score"])
    ambiguous = bool(
        config.ambiguous_gap_min <= second_gap <= config.ambiguous_gap_max
        and second_margin <= config.ambiguous_score_margin
    )
    gr_rows = max((int(m["gr_rows"]) for m in modes), default=0)
    gr_coverage = float(gr_rows / max(len(eval_idx), 1))
    replay = float(heel_meta["prefix_replay_rmse"])
    replay_factor = np.exp(-min(replay, 50.0) / 15.0) if np.isfinite(replay) else 0.0
    calibration_factor = np.exp(-min(calibration_rmse, 100.0) / 30.0) if np.isfinite(calibration_rmse) else 0.0
    confidence = float(gr_coverage * calibration_factor * replay_factor * (1.0 - 0.35 * entropy))
    base_item = next(item for item in scored if item["name"] == "base")
    map_score_gain = float(base_item["score"] - best["score"])
    if map_score_gain < config.min_score_gain:
        confidence *= max(0.0, map_score_gain / max(config.min_score_gain, 1e-9))

    metadata: dict[str, Any] = {
        **asdict(config),
        **heel_meta,
        "status": "ok",
        "search_base": search_base_name,
        "eval_rows": int(len(eval_idx)),
        "calibration_rows": calibration_rows,
        "calibration_rmse": calibration_rmse,
        "gr_rows": gr_rows,
        "gr_coverage": gr_coverage,
        "candidate_count": int(len(scored)),
        "mode_count": int(len(modes)),
        "map_name": str(best["name"]),
        "map_score": float(best["score"]),
        "base_score": float(base_item["score"]),
        "map_score_gain": map_score_gain,
        "map_datum": float(best["datum"]),
        "second_mode_gap": second_gap,
        "second_score_margin": second_margin,
        "ambiguous": ambiguous,
        "entropy": entropy,
        "confidence": confidence,
        "mode_names": [str(m["name"]) for m in modes],
        "mode_datums": [float(m["datum"]) for m in modes],
        "mode_scores": [float(m["score"]) for m in modes],
        "mode_weights": [float(w) for w in weights],
        "calibration_scales": [
            {
                "name": str(scale["name"]),
                "slope": float(scale["slope"]),
                "intercept": float(scale["intercept"]),
                "sigma": float(scale["sigma"]),
                "rmse": float(scale["rmse"]),
                "rows": int(scale["rows"]),
            }
            for scale in scales
        ],
    }
    chosen_alpha = _heuristic_alpha(metadata) if alpha is None else float(alpha)
    chosen_alpha = float(np.clip(chosen_alpha, 0.0, 1.0))
    metadata["alpha"] = chosen_alpha

    prior_full = prior.copy()
    map_full = prior.copy()
    posterior_full = prior.copy()
    known_values = hw["TVT_input"].to_numpy(dtype=float)
    known_mask = np.isfinite(known_values)
    prior_full[known_mask] = known_values[known_mask]
    map_full[known_mask] = known_values[known_mask]
    posterior_full[known_mask] = known_values[known_mask]
    map_full[eval_idx] = map_eval
    posterior_full[eval_idx] = posterior_eval

    if anchor is None:
        anchor_full = prior_full
    else:
        anchor_full = np.asarray(anchor, dtype=float).copy()
        if len(anchor_full) != len(hw):
            raise ValueError("Anchor length differs from horizontal well")
        anchor_full[known_mask] = known_values[known_mask]
    hybrid = anchor_full.copy()
    hybrid[eval_idx] = anchor_full[eval_idx] + chosen_alpha * (
        posterior_full[eval_idx] - anchor_full[eval_idx]
    )
    return PosteriorResult(prior_full, map_full, posterior_full, hybrid, metadata)


def result_summary(result: PosteriorResult) -> dict[str, Any]:
    out = dict(result.metadata)
    for name in ("prior", "map_path", "posterior", "hybrid"):
        values = np.asarray(getattr(result, name), dtype=float)
        out[f"{name}_finite"] = bool(np.isfinite(values).all())
        out[f"{name}_min"] = float(np.nanmin(values))
        out[f"{name}_max"] = float(np.nanmax(values))
    return out
