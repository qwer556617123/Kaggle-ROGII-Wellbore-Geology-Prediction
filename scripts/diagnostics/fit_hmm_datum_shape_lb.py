"""Fit a two-axis Public-LB quadratic for HMM datum and shape components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/kernel_outputs/hmm_student_shape050_v1"),
    )
    parser.add_argument("--anchor-score", type=float, default=7.053)
    parser.add_argument("--scalar-score", type=float, default=6.909)
    parser.add_argument("--shape050-score", type=float, default=6.986)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/hmm_datum_shape_lb_fit.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.output_dir / "submission_before_exact_hmm.csv")
    hmm = pd.read_csv(args.output_dir / "exact_hmm_predictions.csv")
    base["id"] = base["id"].astype(str)
    hmm["id"] = hmm["id"].astype(str)
    merged = base[["id", "tvt"]].merge(
        hmm[["id", "hmm_tvt"]], on="id", how="left", validate="one_to_one"
    )
    if merged["hmm_tvt"].isna().any():
        raise RuntimeError("incomplete HMM vector")
    direction = merged["hmm_tvt"].to_numpy(float) - merged["tvt"].to_numpy(float)
    wells = merged["id"].str.rsplit("_", n=1).str[0]
    mean_direction = pd.Series(direction).groupby(wells, sort=False).transform("mean").to_numpy()
    shape_direction = direction - mean_direction
    cross = float(np.mean(mean_direction * shape_direction))
    mean_norm2 = float(np.mean(mean_direction**2))
    shape_norm2 = float(np.mean(shape_direction**2))

    # score(a,b)^2 = S0^2 + 2*a*pm + a^2*Mm + 2*b*ps + b^2*Ss.
    # Scored points are (a,b)=(.25,.25) and (.25,.50).
    delta_scalar = args.scalar_score**2 - args.anchor_score**2
    delta_shape = args.shape050_score**2 - args.scalar_score**2
    shape_projection = (
        delta_shape - (0.50**2 - 0.25**2) * shape_norm2
    ) / (2.0 * (0.50 - 0.25))
    projection_sum = (
        delta_scalar - 0.25**2 * (mean_norm2 + shape_norm2)
    ) / (2.0 * 0.25)
    mean_projection = projection_sum - shape_projection
    mean_optimum = -mean_projection / mean_norm2
    shape_optimum = -shape_projection / shape_norm2

    grid = []
    for mean_weight in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.25):
        for shape_weight in (0.20, 0.25, 0.30, 0.35, 0.40):
            score2 = (
                args.anchor_score**2
                + 2 * mean_weight * mean_projection
                + mean_weight**2 * mean_norm2
                + 2 * shape_weight * shape_projection
                + shape_weight**2 * shape_norm2
            )
            grid.append(
                {
                    "mean_weight": mean_weight,
                    "shape_weight": shape_weight,
                    "predicted_score": float(np.sqrt(max(score2, 0.0))),
                }
            )
    grid.sort(key=lambda row: row["predicted_score"])
    result = {
        "scores": {
            "anchor_0_0": args.anchor_score,
            "student_025_025": args.scalar_score,
            "student_025_050": args.shape050_score,
        },
        "visible_component_norm2": {"mean": mean_norm2, "shape": shape_norm2},
        "visible_component_cross": cross,
        "estimated_projection": {"mean": mean_projection, "shape": shape_projection},
        "unclamped_optimum": {"mean": mean_optimum, "shape": shape_optimum},
        "recommended_conservative": {"mean": 0.0, "shape": 0.30},
        "top_grid": grid[:10],
        "warning": "Component norms come from the visible rerun; use only as a low-dimensional probe.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
