"""Pseudo-hidden audit for GR-constrained path warping.

This diagnostic tests a stricter geo hypothesis than the geo path selector:
keep the PF-family path as the geometric prior, then let calibrated GR/typewell
misfit produce a small, smooth TVT warp around that prior. It intentionally does
not use the target well's formation contact columns for inference.
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


def robust_sigma(values: np.ndarray, floor: float = 5.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(floor)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return float(max(floor, 1.4826 * mad))


def robust_loss(values: np.ndarray, delta: float = 2.5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    av = np.abs(values)
    return np.where(av <= delta, 0.5 * values * values, delta * (av - 0.5 * delta))


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
    return tw_tvt, tw_gr


def fill_gr(hw: pd.DataFrame, fallback: float) -> np.ndarray:
    return (
        pd.Series(hw["GR"])
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def gr_calibration(hw: pd.DataFrame, tw: pd.DataFrame, smooth_window: int) -> dict[str, object]:
    known = hw["TVT_input"].notna().to_numpy()
    if int(known.sum()) < 20:
        raise RuntimeError("Not enough known TVT_input rows for GR calibration")
    tw_tvt, tw_gr_raw = typewell_arrays(tw)
    tw_gr = rolling_mean(tw_gr_raw, smooth_window)
    hw_gr = fill_gr(hw, float(np.nanmean(tw_gr)))
    known_tvt = hw.loc[known, "TVT_input"].to_numpy(dtype=float)
    x = np.interp(known_tvt, tw_tvt, tw_gr)
    y = hw_gr[known]
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 20 or float(np.std(x[valid])) < 1e-6:
        raise RuntimeError("Degenerate GR calibration")
    mat = np.column_stack([x[valid], np.ones(int(valid.sum()))])
    scale, shift = np.linalg.lstsq(mat, y[valid], rcond=None)[0]
    if abs(scale) < 1e-6:
        scale = 1.0
        shift = 0.0
    fitted = scale * x[valid] + shift
    resid = y[valid] - fitted
    sigma = robust_sigma(resid, floor=5.0)
    denom = float(np.var(y[valid]))
    r2 = float(1.0 - np.var(resid) / denom) if denom > 1e-9 else 0.0
    return {
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "hw_gr": hw_gr,
        "scale": float(scale),
        "shift": float(shift),
        "sigma": float(sigma),
        "r2": float(r2),
    }


def gr_path_loss(hw: pd.DataFrame, tw: pd.DataFrame, full_path: np.ndarray, eval_mask: np.ndarray, smooth_window: int) -> float:
    try:
        cal = gr_calibration(hw, tw, smooth_window)
    except RuntimeError:
        return 0.0
    tw_tvt = np.asarray(cal["tw_tvt"], dtype=float)
    tw_gr = np.asarray(cal["tw_gr"], dtype=float)
    hw_gr = np.asarray(cal["hw_gr"], dtype=float)
    expected = float(cal["scale"]) * np.interp(full_path[eval_mask], tw_tvt, tw_gr) + float(cal["shift"])
    z = (hw_gr[eval_mask] - expected) / float(cal["sigma"])
    return float(np.mean(robust_loss(z)))


def tail_jump(hw: pd.DataFrame, full_path: np.ndarray, eval_mask: np.ndarray) -> float:
    known_idx = np.where(hw["TVT_input"].notna().to_numpy())[0]
    eval_idx = np.where(eval_mask)[0]
    if len(known_idx) == 0 or len(eval_idx) == 0:
        return 0.0
    return float(abs(full_path[int(eval_idx[0])] - hw.loc[int(known_idx[-1]), "TVT_input"]))


def warp_candidates(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    base_full: np.ndarray,
    eval_mask: np.ndarray,
    max_abs: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) == 0:
        return candidates
    for gr_window in (9, 31):
        try:
            cal = gr_calibration(hw, tw, gr_window)
        except RuntimeError:
            continue
        tw_tvt = np.asarray(cal["tw_tvt"], dtype=float)
        tw_gr = np.asarray(cal["tw_gr"], dtype=float)
        hw_gr = np.asarray(cal["hw_gr"], dtype=float)
        scale = float(cal["scale"])
        shift = float(cal["shift"])
        sigma = float(cal["sigma"])
        tvt_step = np.gradient(tw_tvt)
        tvt_step = np.where(np.abs(tvt_step) < 1e-6, np.sign(tvt_step + 1e-12) * 1e-6, tvt_step)
        tw_grad = np.gradient(tw_gr) / tvt_step
        expected = scale * np.interp(base_full[eval_idx], tw_tvt, tw_gr) + shift
        denom = scale * np.interp(base_full[eval_idx], tw_tvt, tw_grad)
        residual = hw_gr[eval_idx] - expected
        reliable = np.abs(denom) >= 0.08
        raw = np.zeros(len(eval_idx), dtype=float)
        raw[reliable] = residual[reliable] / denom[reliable]
        raw = np.clip(raw, -float(max_abs), float(max_abs))
        md = hw["MD"].to_numpy(dtype=float)[eval_idx]
        ramp = 1.0 - np.exp(-(md - md[0]) / 100.0)
        ramp = np.clip(ramp, 0.0, 1.0)
        for corr_window in (31, 121, 301):
            corr = rolling_mean(raw, min(corr_window, len(raw))) * ramp
            for alpha in (0.15, 0.30, 0.50):
                full = base_full.copy()
                full[eval_idx] = base_full[eval_idx] + alpha * corr
                full[eval_idx] = np.clip(
                    full[eval_idx],
                    base_full[eval_idx] - float(max_abs),
                    base_full[eval_idx] + float(max_abs),
                )
                loss = gr_path_loss(hw, tw, full, eval_mask, gr_window)
                jump = tail_jump(hw, full, eval_mask)
                rough = float(np.std(np.diff(corr))) if len(corr) > 2 else 0.0
                mean_abs = float(np.mean(np.abs(full[eval_idx] - base_full[eval_idx])))
                score = loss + jump / 50.0 + rough / 1.5 + max(0.0, mean_abs - 12.0) / 8.0
                candidates.append({
                    "name": f"grwarp_g{gr_window}_c{corr_window}_a{alpha:.2f}".replace(".", "p"),
                    "full": full,
                    "score": float(score),
                    "gr_loss": float(loss),
                    "tail_jump": float(jump),
                    "roughness": float(rough),
                    "mean_abs_delta": mean_abs,
                    "max_abs_delta": float(np.max(np.abs(full[eval_idx] - base_full[eval_idx]))),
                    "gr_window": int(gr_window),
                    "corr_window": int(corr_window),
                    "alpha": float(alpha),
                    "gr_sigma": sigma,
                    "gr_r2": float(cal["r2"]),
                    "reliable_fraction": float(np.mean(reliable)),
                })
    return candidates


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
            "mean_score": float(g["score"].mean()),
            "mean_abs_delta": float(g["mean_abs_delta"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def gate_summary(summary: pd.DataFrame, min_row_gain: float, max_well_deterioration: float) -> dict[str, object]:
    indexed = summary.set_index("variant")
    base = indexed.loc["base_grid_s3_b0_h0p17"]
    selected = indexed.loc["gr_selected_warp"]
    conservative = indexed.loc["gr_selected_warp_0p5"]
    best_name = "gr_selected_warp" if float(selected["row_rmse"]) <= float(conservative["row_rmse"]) else "gr_selected_warp_0p5"
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
    parser.add_argument("--well-limit", type=int, default=100)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-abs-delta", type=float, default=20.0)
    parser.add_argument("--min-row-gain", type=float, default=0.30)
    parser.add_argument("--max-well-deterioration", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path, default=Path("docs/geo_warped_path_summary.csv"))
    parser.add_argument("--detail-output", type=Path, default=Path("docs/geo_warped_path_details.csv"))
    parser.add_argument("--gate-output", type=Path, default=Path("docs/geo_warped_path_gate.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pf_eval = load_module("evaluate_pf_variants", PF_EVAL)
    cfg = pf_eval.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf_eval.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)
    rows: list[dict[str, object]] = []
    print(
        f"Geo warped path audit wells={len(wells)} selection={args.selection} "
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
            base_full = preds["grid_s3_b0_h0p17"]
            y_true = hw.loc[eval_mask, "TVT"].to_numpy(dtype=float)
            base_eval = base_full[eval_mask]
            candidates = warp_candidates(hw, tw, base_full, eval_mask, args.max_abs_delta)
            if not candidates:
                print(f"[skip-warp] {wid} frac={known_frac:.2f}: no candidates", flush=True)
                continue
            selected = min(candidates, key=lambda item: float(item["score"]))
            oracle = min(candidates, key=lambda item: rmse(y_true, np.asarray(item["full"])[eval_mask]))
            selected_eval = np.asarray(selected["full"], dtype=float)[eval_mask]
            conservative_eval = 0.5 * base_eval + 0.5 * selected_eval
            variants = [
                ("base_grid_s3_b0_h0p17", base_eval, {"name": "base", "score": 0.0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0}),
                ("best_warp_oracle", np.asarray(oracle["full"], dtype=float)[eval_mask], oracle),
                ("gr_selected_warp", selected_eval, selected),
                ("gr_selected_warp_0p5", conservative_eval, selected),
            ]
            for variant, pred, meta in variants:
                rows.append({
                    "well": wid,
                    "known_frac": float(known_frac),
                    "variant": variant,
                    "rmse": rmse(y_true, pred),
                    "bias": float(np.mean(pred - y_true)),
                    "n_eval": int(eval_mask.sum()),
                    "candidate": str(meta.get("name", "")),
                    "score": float(meta.get("score", 0.0)),
                    "gr_loss": float(meta.get("gr_loss", 0.0)),
                    "tail_jump": float(meta.get("tail_jump", 0.0)),
                    "mean_abs_delta": float(meta.get("mean_abs_delta", 0.0)),
                    "max_abs_delta": float(meta.get("max_abs_delta", 0.0)),
                    "reliable_fraction": float(meta.get("reliable_fraction", 0.0)),
                    "gr_sigma": float(meta.get("gr_sigma", 0.0)),
                    "gr_r2": float(meta.get("gr_r2", 0.0)),
                })
    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No audit rows produced")
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
