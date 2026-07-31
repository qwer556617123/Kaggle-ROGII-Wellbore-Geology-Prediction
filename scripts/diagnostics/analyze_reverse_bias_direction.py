"""Calibrate a scored residual direction and transfer it to another anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["id", "tvt"])
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate ids in {path}")
    values = frame["tvt"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--target-anchor", type=Path, required=True)
    parser.add_argument("--base-score", type=float, required=True)
    parser.add_argument("--positive-score", type=float, required=True)
    parser.add_argument("--positive-dose", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = load(args.base)
    positive = load(args.positive)
    target = load(args.target_anchor)
    ids = base["id"].astype(str)
    for label, frame in (("positive", positive), ("target", target)):
        if not ids.equals(frame["id"].astype(str)):
            raise ValueError(f"{label} id order differs from base")

    base_values = base["tvt"].to_numpy(dtype=float)
    positive_values = positive["tvt"].to_numpy(dtype=float)
    target_values = target["tvt"].to_numpy(dtype=float)
    unit_direction = (positive_values - base_values) / args.positive_dose
    q_unit = float(np.mean(np.square(unit_direction)))
    scored_q = q_unit * args.positive_dose**2
    scored_projection = (
        args.positive_score**2 - args.base_score**2 - scored_q
    ) / 2.0
    unit_projection = scored_projection / args.positive_dose
    optimum_dose = -unit_projection / q_unit

    split = ids.str.rsplit("_", n=1, expand=True)
    audit = pd.DataFrame(
        {
            "well": split[0],
            "direction": unit_direction,
            "target_delta": target_values - base_values,
        }
    )
    by_well = (
        audit.groupby("well", sort=True)
        .agg(
            rows=("direction", "size"),
            unit_rms=("direction", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            unit_mean=("direction", "mean"),
            target_delta_rms=(
                "target_delta", lambda x: float(np.sqrt(np.mean(np.square(x))))
            ),
        )
        .reset_index()
    )

    dose_grid = sorted(
        set(
            np.round(
                np.concatenate(
                    [np.arange(-0.10, 0.005, 0.005), [optimum_dose]]
                ),
                6,
            )
        )
    )
    predictions = []
    for dose in dose_grid:
        square = (
            args.base_score**2
            + 2.0 * dose * unit_projection
            + dose**2 * q_unit
        )
        predictions.append(
            {"dose": float(dose), "predicted_lb_from_base": float(np.sqrt(max(square, 0.0)))}
        )

    summary = {
        "base": str(args.base),
        "positive": str(args.positive),
        "target_anchor": str(args.target_anchor),
        "base_sha256": sha256(args.base),
        "positive_sha256": sha256(args.positive),
        "target_anchor_sha256": sha256(args.target_anchor),
        "rows": int(len(base)),
        "base_score": args.base_score,
        "positive_score": args.positive_score,
        "positive_dose": args.positive_dose,
        "unit_direction_rms_ft": float(np.sqrt(q_unit)),
        "unit_residual_projection": unit_projection,
        "optimum_dose": optimum_dose,
        "predictions": predictions,
        "by_well": by_well.to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
