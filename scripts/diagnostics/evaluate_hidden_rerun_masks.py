"""Evaluate PF variants on artificial hidden-rerun masks.

The code competition reruns notebooks with substituted hidden test data, so the
visible three test wells are not a reliable Public LB target. This diagnostic
creates pseudo-hidden prediction starts inside train wells by replacing
post-cut `TVT_input` with NaN while keeping `TVT` as truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_pf_module():
    path = Path("scripts/diagnostics/evaluate_pf_variants.py")
    spec = importlib.util.spec_from_file_location("pf_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true.astype(float) - y_pred.astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def make_hidden_mask(hw: pd.DataFrame, known_frac: float, min_known: int, min_eval: int) -> tuple[pd.DataFrame, np.ndarray]:
    if "TVT" not in hw.columns:
        raise ValueError("Train horizontal well is missing TVT truth.")
    truth_mask = hw["TVT"].notna().to_numpy()
    truth_idx = np.where(truth_mask)[0]
    if len(truth_idx) < min_known + min_eval:
        raise ValueError("Not enough TVT truth rows for requested mask.")

    first = int(truth_idx[0])
    last = int(truth_idx[-1])
    span = last - first + 1
    cut = first + int(round(span * float(known_frac)))
    cut = int(np.clip(cut, first + min_known - 1, last - min_eval))

    out = hw.copy()
    out["TVT_input"] = out["TVT"].where(np.arange(len(out)) <= cut, np.nan)
    eval_mask = np.zeros(len(out), dtype=bool)
    eval_mask[(np.arange(len(out)) > cut) & truth_mask] = True
    if eval_mask.sum() < min_eval:
        raise ValueError("Mask produced too few eval rows.")
    return out, eval_mask


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in details.groupby("variant", sort=True):
        y_true = np.concatenate([np.asarray(x, dtype=float) for x in g["y_true"]])
        y_pred = np.concatenate([np.asarray(x, dtype=float) for x in g["y_pred"]])
        rows.append({
            "variant": variant,
            "n_masks": int(len(g)),
            "n_wells": int(g["well"].nunique()),
            "n_rows": int(sum(len(x) for x in g["y_true"])),
            "row_rmse": rmse(y_true, y_pred),
            "well_rmse_mean": float(g["rmse"].mean()),
            "well_rmse_median": float(g["rmse"].median()),
            "well_rmse_p75": float(g["rmse"].quantile(0.75)),
            "well_rmse_max": float(g["rmse"].max()),
        })
    return pd.DataFrame(rows).sort_values(["row_rmse", "well_rmse_mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--selection", choices=["lb_like", "hard", "random", "all"], default="lb_like")
    parser.add_argument("--well-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particles", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--known-fracs", default="0.45,0.60,0.75")
    parser.add_argument("--min-known", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=80)
    parser.add_argument(
        "--variants",
        default="anchor,contact_physics,grid_s3_b0_h0p17,grid_s3_b0_h0p2",
        help="Comma-separated variants. 'anchor' repeats the last known TVT.",
    )
    parser.add_argument("--details-output", type=Path, default=Path("docs/hidden_rerun_mask_details.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("docs/hidden_rerun_mask_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pf = load_pf_module()
    cfg = pf.PfConfig(n_particles=args.particles, n_seeds=args.seeds)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pf_variants = [v for v in variants if v != "anchor"]
    known_fracs = [float(x.strip()) for x in args.known_fracs.split(",") if x.strip()]
    wells = pf.select_wells(args.data_dir, args.selection, args.well_limit, args.seed)

    rows = []
    for wi, wid in enumerate(wells, 1):
        hw_raw, tw = pf.load_well(args.data_dir, wid, "train")
        for known_frac in known_fracs:
            try:
                hw, eval_mask = make_hidden_mask(hw_raw, known_frac, args.min_known, args.min_eval)
            except ValueError as exc:
                print(f"[skip] {wid} frac={known_frac:.2f}: {exc}", flush=True)
                continue

            print(f"[{wi:03d}/{len(wells):03d}] {wid} frac={known_frac:.2f} eval={int(eval_mask.sum())}", flush=True)
            preds = {}
            if pf_variants:
                preds, meta = pf.predict_variants(hw, tw, pf_variants, cfg)
            else:
                meta = {}
            if "anchor" in variants:
                anchor = hw["TVT_input"].ffill().bfill().to_numpy(float)
                preds["anchor"] = anchor

            y_true = hw.loc[eval_mask, "TVT"].to_numpy(float)
            profile = pf.well_profile(hw)
            for variant in variants:
                y_pred = preds[variant][eval_mask]
                row = {
                    "well": wid,
                    "known_frac_request": float(known_frac),
                    "known_frac_actual": float((~eval_mask).mean()),
                    "variant": variant,
                    "rmse": rmse(y_true, y_pred),
                    "n_eval": int(eval_mask.sum()),
                    "y_true": y_true.tolist(),
                    "y_pred": y_pred.astype(float).tolist(),
                }
                row.update({f"profile_{k}": float(v) for k, v in profile.items()})
                row.update({f"meta_{k}": float(v) for k, v in meta.items() if isinstance(v, (int, float, np.integer, np.floating))})
                rows.append(row)

    if not rows:
        raise RuntimeError("No hidden-rerun masks were evaluated.")

    details = pd.DataFrame(rows)
    summary = summarize(details)
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.details_output}")
    print(f"Saved {args.summary_output}")


if __name__ == "__main__":
    main()
