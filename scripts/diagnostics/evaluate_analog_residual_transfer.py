"""Pseudo-hidden audit for analog residual transfer.

The goal is to test a geology-flavored regime idea without hard-coding public
well ids: describe each hidden prefix by trajectory and GR/typewell alignment,
find similar train pseudo-hidden analogs, and transfer their PF residual shape
as a smooth correction around the PF baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PF_EVAL = ROOT / "scripts" / "diagnostics" / "evaluate_pf_variants.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred.astype(float) - y_true.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(min(int(window), len(values)), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3 or float(np.ptp(x[ok])) < 1e-9:
        return 0.0
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(1e-6, 1.4826 * mad)


def make_hidden_mask(hw: pd.DataFrame, known_frac: float, min_known: int, min_eval: int) -> tuple[pd.DataFrame, np.ndarray]:
    truth_mask = hw["TVT"].notna().to_numpy()
    truth_idx = np.where(truth_mask)[0]
    if len(truth_idx) < min_known + min_eval:
        raise ValueError("Not enough TVT truth rows for requested mask")
    first = int(truth_idx[0])
    last = int(truth_idx[-1])
    cut = first + int(round((last - first + 1) * float(known_frac)))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))
    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = (np.arange(len(out)) > cut) & truth_mask
    return out, eval_mask


def typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(dtype=float)
    tw_gr = (
        pd.Series(tw_s["GR"])
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    return tw_tvt, rolling_mean(tw_gr, 15)


def fill_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:
    return (
        pd.Series(hw["GR"])
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def gr_calibration_features(hw: pd.DataFrame, tw: pd.DataFrame, base_full: np.ndarray, eval_mask: np.ndarray) -> dict[str, float]:
    known = hw["TVT_input"].notna().to_numpy()
    if int(known.sum()) < 20:
        return {"gr_sigma": 25.0, "gr_r2": 0.0, "eval_gr_loss": 1.0, "eval_gr_std": 0.0}
    tw_tvt, tw_gr = typewell_arrays(tw)
    hw_gr = fill_gr(hw, float(np.nanmean(tw_gr)))
    known_tvt = hw.loc[known, "TVT_input"].to_numpy(dtype=float)
    x = np.interp(known_tvt, tw_tvt, tw_gr)
    y = hw_gr[known]
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 20 or float(np.std(x[ok])) < 1e-6:
        return {"gr_sigma": 25.0, "gr_r2": 0.0, "eval_gr_loss": 1.0, "eval_gr_std": 0.0}
    mat = np.column_stack([x[ok], np.ones(int(ok.sum()))])
    scale, shift = np.linalg.lstsq(mat, y[ok], rcond=None)[0]
    fitted = scale * x[ok] + shift
    resid = y[ok] - fitted
    _, sigma = robust_stats(resid)
    sigma = max(5.0, sigma)
    denom = float(np.var(y[ok]))
    r2 = float(1.0 - np.var(resid) / denom) if denom > 1e-9 else 0.0
    expected = scale * np.interp(base_full[eval_mask], tw_tvt, tw_gr) + shift
    z = (hw_gr[eval_mask] - expected) / sigma
    eval_loss = float(np.mean(np.minimum(z * z, 25.0))) if len(z) else 0.0
    return {
        "gr_sigma": float(sigma),
        "gr_r2": float(r2),
        "eval_gr_loss": eval_loss,
        "eval_gr_std": float(np.nanstd(hw_gr[eval_mask])) if int(eval_mask.sum()) else 0.0,
    }


def profile_record(
    well: str,
    known_frac: float,
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    eval_mask: np.ndarray,
    base_full: np.ndarray,
) -> dict[str, object]:
    known = hw["TVT_input"].notna().to_numpy()
    eval_idx = np.where(eval_mask)[0]
    known_idx = np.where(known)[0]
    md = hw["MD"].to_numpy(dtype=float)
    z = hw["Z"].to_numpy(dtype=float)
    tvt_known = hw["TVT_input"].to_numpy(dtype=float)
    tail_idx = known_idx[-min(80, len(known_idx)):]
    eval_md = md[eval_idx]
    eval_z = z[eval_idx]
    base_eval = base_full[eval_idx]
    true_eval = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
    residual = true_eval - base_eval
    progress = np.linspace(0.0, 1.0, len(eval_idx)) if len(eval_idx) else np.array([], dtype=float)
    gr_feats = gr_calibration_features(hw, tw, base_full, eval_mask)
    return {
        "well": well,
        "known_frac": float(known_frac),
        "n_eval": float(len(eval_idx)),
        "eval_md_span": float(np.ptp(eval_md)) if len(eval_md) else 0.0,
        "eval_z_span": float(np.ptp(eval_z)) if len(eval_z) else 0.0,
        "eval_z_slope": finite_slope(eval_md, eval_z),
        "tail_tvt_slope": finite_slope(md[tail_idx], tvt_known[tail_idx]) if len(tail_idx) else 0.0,
        "tail_z_slope": finite_slope(md[tail_idx], z[tail_idx]) if len(tail_idx) else 0.0,
        "base_eval_slope": finite_slope(eval_md, base_eval),
        "base_delta_end": float(base_eval[-1] - tvt_known[known_idx[-1]]) if len(eval_idx) and len(known_idx) else 0.0,
        "base_eval_std": float(np.std(base_eval)) if len(base_eval) else 0.0,
        **gr_feats,
        "progress": progress,
        "residual": residual.astype(float),
        "base_eval": base_eval.astype(float),
        "truth": true_eval.astype(float),
    }


FEATURE_COLS = [
    "known_frac",
    "n_eval",
    "eval_md_span",
    "eval_z_span",
    "eval_z_slope",
    "tail_tvt_slope",
    "tail_z_slope",
    "base_eval_slope",
    "base_delta_end",
    "base_eval_std",
    "gr_sigma",
    "gr_r2",
    "eval_gr_loss",
    "eval_gr_std",
]


def interpolate_template(progress_src: np.ndarray, residual_src: np.ndarray, progress_dst: np.ndarray, center: bool) -> np.ndarray:
    if len(progress_src) == 0 or len(residual_src) == 0:
        return np.zeros(len(progress_dst), dtype=float)
    res = np.asarray(residual_src, dtype=float)
    if center:
        res = res - float(np.mean(res))
    smooth = rolling_mean(res, min(201, max(5, len(res) // 8 * 2 + 1)))
    return np.interp(progress_dst, progress_src, smooth).astype(float)


def analog_prediction(
    records: list[dict[str, object]],
    target_i: int,
    feature_median: pd.Series,
    feature_scale: pd.Series,
    k: int,
    center: bool,
    alpha: float,
    exclude_same_well: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    target = records[target_i]
    frame = pd.DataFrame([{c: rec[c] for c in FEATURE_COLS} for rec in records])
    z = (frame - feature_median) / feature_scale
    target_z = z.iloc[target_i].to_numpy(dtype=float)
    pool = []
    for j, rec in enumerate(records):
        if j == target_i:
            continue
        if exclude_same_well and rec["well"] == target["well"]:
            continue
        diff = z.iloc[j].to_numpy(dtype=float) - target_z
        dist = float(np.sqrt(np.mean(diff * diff)))
        pool.append((dist, j))
    pool.sort(key=lambda item: item[0])
    selected = pool[: min(k, len(pool))]
    if not selected:
        return np.asarray(target["base_eval"], dtype=float), {"analog_count": 0}
    weights = np.array([1.0 / (0.05 + d) for d, _ in selected], dtype=float)
    weights /= weights.sum()
    target_progress = np.asarray(target["progress"], dtype=float)
    correction = np.zeros(len(target_progress), dtype=float)
    analog_wells = []
    for weight, (dist, j) in zip(weights, selected):
        src = records[j]
        analog_wells.append(str(src["well"]))
        correction += weight * interpolate_template(
            np.asarray(src["progress"], dtype=float),
            np.asarray(src["residual"], dtype=float),
            target_progress,
            center=center,
        )
    correction = np.clip(correction, -25.0, 25.0)
    pred = np.asarray(target["base_eval"], dtype=float) + float(alpha) * correction
    return pred, {
        "analog_count": int(len(selected)),
        "mean_distance": float(np.average([d for d, _ in selected], weights=weights)),
        "analog_wells": ",".join(analog_wells[:5]),
        "mean_abs_correction": float(np.mean(np.abs(float(alpha) * correction))),
        "max_abs_correction": float(np.max(np.abs(float(alpha) * correction))) if len(correction) else 0.0,
    }


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant", sort=True):
        rows.append({
            "variant": variant,
            "n_masks": int(len(g)),
            "n_wells": int(g["well"].nunique()),
            "n_rows": int(g["n_eval"].sum()),
            "row_rmse": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n_eval"]))),
            "well_rmse_mean": float(g["rmse"].mean()),
            "well_rmse_median": float(g["rmse"].median()),
            "well_rmse_p75": float(g["rmse"].quantile(0.75)),
            "well_rmse_max": float(g["rmse"].max()),
            "mean_abs_correction": float(g["mean_abs_correction"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def gate_summary(summary: pd.DataFrame, min_row_gain: float, max_well_deterioration: float) -> dict[str, object]:
    indexed = summary.set_index("variant")
    base = indexed.loc["base_grid_s3_b0_h0p17"]
    candidates = indexed.drop(index="base_grid_s3_b0_h0p17")
    best_name = str(candidates["row_rmse"].idxmin())
    best = indexed.loc[best_name]
    row_gain = float(base["row_rmse"] - best["row_rmse"])
    well_deterioration = float(best["well_rmse_mean"] - base["well_rmse_mean"])
    passed = row_gain >= min_row_gain and well_deterioration <= max_well_deterioration
    return {
        "verdict": "PASS" if passed else "STOP",
        "best_variant": best_name,
        "row_gain": row_gain,
        "well_mean_deterioration": well_deterioration,
        "min_row_gain": float(min_row_gain),
        "max_well_deterioration": float(max_well_deterioration),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=80)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--min-row-gain", type=float, default=0.30)
    parser.add_argument("--max-well-deterioration", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/analog_residual_transfer_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/analog_residual_transfer_details.csv"))
    parser.add_argument("--gate-output", type=Path, default=Path("docs/analog_residual_transfer_gate.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pf_eval = load_module("evaluate_pf_variants", PF_EVAL)
    cfg = pf_eval.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    records: list[dict[str, object]] = []
    print(
        f"Analog residual transfer audit wells={len(wells)} selection={args.selection} "
        f"known_fracs={known_fracs} seeds={args.seeds} particles={args.particles}",
        flush=True,
    )
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf_eval.load_well(args.data_dir, wid, "train")
        for known_frac in known_fracs:
            try:
                hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
            except ValueError as exc:
                print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue
            print(f"[{wi:03d}/{len(wells):03d}] {wid} frac={known_frac:.2f} n_eval={eval_mask.sum()}", flush=True)
            preds, _ = pf_eval.predict_variants(hw, tw, ["grid_s3_b0_h0p17"], cfg)
            records.append(profile_record(wid, known_frac, hw, tw, eval_mask, preds["grid_s3_b0_h0p17"]))

    if not records:
        raise RuntimeError("No records produced")
    features = pd.DataFrame([{c: rec[c] for c in FEATURE_COLS} for rec in records]).astype(float)
    med = features.median()
    scale = features.apply(lambda s: robust_stats(s.to_numpy(dtype=float))[1]).replace(0, 1.0)

    rows: list[dict[str, object]] = []
    variants = [
        ("analog_bias_a0p25", False, 0.25),
        ("analog_bias_a0p50", False, 0.50),
        ("analog_shape_a0p25", True, 0.25),
        ("analog_shape_a0p50", True, 0.50),
    ]
    for i, rec in enumerate(records):
        y_true = np.asarray(rec["truth"], dtype=float)
        base = np.asarray(rec["base_eval"], dtype=float)
        rows.append({
            "well": rec["well"],
            "known_frac": float(rec["known_frac"]),
            "variant": "base_grid_s3_b0_h0p17",
            "rmse": rmse(y_true, base),
            "bias": float(np.mean(base - y_true)),
            "n_eval": int(len(y_true)),
            "candidate": "base",
            "mean_abs_correction": 0.0,
            "max_abs_correction": 0.0,
            "analog_count": 0,
            "mean_distance": 0.0,
            "analog_wells": "",
        })
        for name, center, alpha in variants:
            pred, meta = analog_prediction(records, i, med, scale, args.k, center, alpha, exclude_same_well=True)
            rows.append({
                "well": rec["well"],
                "known_frac": float(rec["known_frac"]),
                "variant": name,
                "rmse": rmse(y_true, pred),
                "bias": float(np.mean(pred - y_true)),
                "n_eval": int(len(y_true)),
                "candidate": name,
                **meta,
            })
    details = pd.DataFrame(rows)
    summary = summarize(details)
    gate = gate_summary(summary, args.min_row_gain, args.max_well_deterioration)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    details.to_csv(args.detail_output, index=False)
    args.gate_output.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nGate:")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
